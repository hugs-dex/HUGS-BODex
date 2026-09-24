import glob
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import trimesh as tm
import viser

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from curobo.util.visualizer import Visualizer
from curobo.util.artifact_path import resolve_artifact_path
from curobo.util_file import (
    get_assets_path,
    get_manip_configs_path,
    get_robot_path,
    join_path,
    load_yaml,
)
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW


@dataclass
class GraspTypeSpec:
    type_name: str
    manip_cfg_file: str
    robot_file: str
    output_dir: str


@dataclass
class GraspRecord:
    type_name: str
    manip_cfg_file: str
    robot_file: str
    file_path: str
    scene_id: str
    grasp_count: Optional[int] = None
    type_budget: Optional[int] = None
    total_budget: Optional[int] = None


TYPE_COLORS = {
    "1_right_two": [0.941, 0.502, 0.502, 0.72],
    "2_right_three": [0.337, 0.706, 0.914, 0.72],
    "3_right_full": [0.596, 0.780, 0.376, 0.72],
    "4_both_three": [0.961, 0.729, 0.275, 0.72],
    "5_both_full": [0.686, 0.549, 0.820, 0.72],
}

ALL_SCALE_LABEL = "all"
UNKNOWN_SCALE_LABEL = "unknown"
SCALE_PATTERN = re.compile(r"(?:^|/)scale(?P<code>\d+)(?=_|/|$)")


def _cfg_get(cfg, key, default=None):
    """Read a key from an OmegaConf config with a Python default.

    Args:
        cfg: OmegaConf section or plain mapping.
        key: Key to read.
        default: Value returned when the key is absent.

    Returns:
        Config value or ``default``.
    """
    return cfg[key] if key in cfg else default


def _normalize_grasp_types(grasp_types) -> Optional[List[str]]:
    """Normalize a Hydra grasp type option into a list.

    Args:
        grasp_types: Config value containing ``all`` or type names/ids.

    Returns:
        List of requested type tokens, or ``None`` for all types.
    """
    if grasp_types is None:
        return None
    if isinstance(grasp_types, str):
        if grasp_types == "all":
            return None
        return [token for token in grasp_types.split(",") if token]
    values = [str(value) for value in grasp_types]
    if "all" in values:
        return None
    return values


def _type_matches(type_name: str, type_config: Dict, requested: Optional[Sequence[str]]) -> bool:
    """Check whether a type should be included in the viewer.

    Args:
        type_name: Runtime grasp type name.
        type_config: Runtime grasp type config.
        requested: Optional requested type names or ids.

    Returns:
        Whether the type should be used.
    """
    if requested is None:
        return True
    type_id = str(type_config.get("type_id", str(type_name).split("_", 1)[0]))
    return type_name in requested or type_id in requested


def _progress(items, desc: str, unit: str):
    """Wrap initialization work in a tqdm progress bar when tqdm is available.

    Args:
        items: Sized iterable to report progress for.
        desc: Progress bar description.
        unit: Unit name displayed by tqdm.

    Returns:
        Iterable that yields the original items.
    """
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)


def _load_suite_config(task_cfg) -> Dict:
    """Load the visualize suite config.

    Args:
        task_cfg: Visualize task config.

    Returns:
        Suite config dictionary.
    """
    suite_config = _cfg_get(task_cfg, "suite_config", None)
    if suite_config is None:
        hand = str(_cfg_get(task_cfg, "hand", "shadow"))
        suite_config = f"sim_{hand}.yml"
    return load_yaml(join_path(get_manip_configs_path(), suite_config))


def _resolve_output_dir(output_path: str, manip_cfg_file: str, exp_name: str) -> str:
    """Resolve one grasp type's ``graspdata`` output directory.

    Args:
        output_path: Root output path from the main config.
        manip_cfg_file: Manipulation config path.
        exp_name: Experiment name.

    Returns:
        Absolute or workspace-relative graspdata directory.
    """
    manip_prefix, _ = os.path.splitext(str(manip_cfg_file))
    return os.path.join(output_path, manip_prefix, exp_name, "graspdata")


def _build_type_specs(cfg, suite_config: Dict) -> List[GraspTypeSpec]:
    """Build grasp type specs from the suite config.

    Args:
        cfg: Full Hydra config.
        suite_config: Loaded suite config.

    Returns:
        Ordered list of grasp type specs.
    """
    requested = _normalize_grasp_types(_cfg_get(cfg.task, "grasp_types", None))
    specs = []
    for type_name, type_config in suite_config.get("grasp_types", {}).items():
        type_config = dict(type_config)
        if not _type_matches(type_name, type_config, requested):
            continue
        manip_cfg_file = str(type_config["manip_cfg_file"])
        manip_config_data = load_yaml(join_path(get_manip_configs_path(), manip_cfg_file))
        specs.append(
            GraspTypeSpec(
                type_name=str(type_name),
                manip_cfg_file=manip_cfg_file,
                robot_file=str(manip_config_data["robot_file"]),
                output_dir=_resolve_output_dir(str(cfg.output_path), manip_cfg_file, str(cfg.name)),
            )
        )
    if not specs:
        raise ValueError("No grasp types matched the visualize config.")
    return specs


def _scene_id_from_grasp_path(file_path: str) -> str:
    """Infer a stable scene id from one saved grasp path without loading it.

    Args:
        file_path: Path to the grasp npy file.

    Returns:
        Scene id string.
    """
    base = file_path.replace("\\", "/").split("/graspdata/", 1)[-1]
    return base[: -len("_grasp.npy")]


def _scale_label_from_scene_id(scene_id: str) -> str:
    """Infer the object scale label encoded in a scene id.

    Args:
        scene_id: Scene id such as ``object/tabletop_ur10e/scale016_pose000_0``.

    Returns:
        Decimal scale label such as ``0.16``, or ``unknown`` when no scale token
        is present.
    """
    match = SCALE_PATTERN.search(str(scene_id))
    if match is None:
        return UNKNOWN_SCALE_LABEL
    return f"{int(match.group('code')) / 100.0:.2f}"


def _scale_sort_key(scale_label: str):
    """Build a stable sorting key for GUI scale labels.

    Args:
        scale_label: Scale dropdown label.

    Returns:
        Tuple that sorts numeric scales before non-numeric fallback labels.
    """
    try:
        return (0, float(scale_label))
    except ValueError:
        return (1, str(scale_label))


def _normalize_scale_label(scale_value) -> str:
    """Normalize a config value into a scale dropdown label.

    Args:
        scale_value: User-provided scale value, such as ``0.16``, ``scale016``,
            ``all``, or ``None``.

    Returns:
        Canonical dropdown label.
    """
    if scale_value is None:
        return ALL_SCALE_LABEL
    scale_text = str(scale_value)
    if scale_text == ALL_SCALE_LABEL:
        return ALL_SCALE_LABEL
    match = re.fullmatch(r"scale(?P<code>\d+)", scale_text)
    if match is not None:
        return f"{int(match.group('code')) / 100.0:.2f}"
    try:
        return f"{float(scale_text):.2f}"
    except ValueError:
        return scale_text


def _make_lazy_record(spec: GraspTypeSpec, file_path: str) -> GraspRecord:
    """Create a grasp record from its path without loading the npy payload.

    Args:
        spec: Grasp type spec.
        file_path: Path to a saved ``*_grasp.npy`` file.

    Returns:
        Lazy grasp record with metadata loaded later on demand.
    """
    return GraspRecord(
        type_name=spec.type_name,
        manip_cfg_file=spec.manip_cfg_file,
        robot_file=spec.robot_file,
        file_path=file_path,
        scene_id=_scene_id_from_grasp_path(file_path),
    )


def _ensure_record_metadata(record: GraspRecord, grasp_data: Optional[Dict] = None) -> int:
    """Load one record's lightweight metadata only when the UI needs it.

    Args:
        record: Lazy grasp record to update in place.
        grasp_data: Optional already-loaded grasp npy dictionary.

    Returns:
        Number of saved grasps in the record.
    """
    if record.grasp_count is not None:
        return int(record.grasp_count)
    if grasp_data is None:
        grasp_data = np.load(record.file_path, allow_pickle=True).item()
    robot_pose = np.asarray(grasp_data["robot_pose"]).squeeze(0)
    record.grasp_count = int(robot_pose.shape[0])
    if "human_prior_type_budget" in grasp_data:
        record.type_budget = int(np.asarray(grasp_data["human_prior_type_budget"]).reshape(-1)[0])
    if "human_prior_type_budgets" in grasp_data:
        record.total_budget = int(np.asarray(grasp_data["human_prior_type_budgets"]).reshape(-1, 5)[0].sum())
    return int(record.grasp_count)


def _build_record_index(type_specs: Sequence[GraspTypeSpec]) -> Dict[str, Dict[str, GraspRecord]]:
    """Collect saved grasp records for all requested types.

    Args:
        type_specs: Grasp type specs.

    Returns:
        Nested mapping ``scene_id -> type_name -> GraspRecord``.
    """
    index: Dict[str, Dict[str, GraspRecord]] = {}
    for spec in _progress(list(type_specs), desc="visualize init: scan grasp dirs", unit="type"):
        file_paths = sorted(glob.glob(os.path.join(spec.output_dir, "**", "*_grasp.npy"), recursive=True))
        print(
            f"visualize init: {spec.type_name} found {len(file_paths)} grasp files in {spec.output_dir}",
            flush=True,
        )
        for file_path in file_paths:
            record = _make_lazy_record(spec, file_path)
            index.setdefault(record.scene_id, {})[record.type_name] = record
    if not index:
        searched = [spec.output_dir for spec in type_specs]
        raise FileNotFoundError(f"No grasp records found. Searched: {searched}")
    print(
        f"visualize init: indexed {sum(len(records) for records in index.values())} "
        f"record paths across {len(index)} scenes; grasp npy files will be loaded on demand",
        flush=True,
    )
    return index


def _load_object_mesh(grasp_data: Dict) -> tm.Trimesh:
    """Load the object mesh from a saved grasp record.

    Args:
        grasp_data: Loaded grasp dictionary.

    Returns:
        Transformed object mesh.
    """
    scene_path = resolve_artifact_path(str(grasp_data["scene_path"][0]))
    scene_data = np.load(scene_path, allow_pickle=True).item()
    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)
    return obj_mesh


def _scene_path_from_grasp_data(grasp_data: Dict) -> str:
    """Read the source scene path from one saved grasp record.

    Args:
        grasp_data: Loaded grasp dictionary.

    Returns:
        Source scene config path saved with the grasp result.
    """
    return resolve_artifact_path(str(grasp_data["scene_path"][0]))


def _make_visualizer(record: GraspRecord, device: str) -> Visualizer:
    """Create a robot visualizer for one grasp type.

    Args:
        record: Grasp record.
        device: Torch device string.

    Returns:
        Visualizer for the record's robot.
    """
    robot_config_data = load_yaml(join_path(get_robot_path(), record.robot_file))
    urdf_file = robot_config_data["robot_cfg"]["kinematics"]["urdf_path"]
    robot_urdf_path = join_path(get_assets_path(), urdf_file)
    mesh_dir_path = os.path.dirname(robot_urdf_path)
    return Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path, device=device)


class GraspViserApp:
    def __init__(self, cfg, type_specs: Sequence[GraspTypeSpec], records_by_scene: Dict[str, Dict[str, GraspRecord]]):
        """Create the viser app state.

        Args:
            cfg: Full Hydra config.
            type_specs: Ordered grasp type specs.
            records_by_scene: Nested scene/type record index.

        Returns:
            None.
        """
        self.cfg = cfg
        self.type_specs = list(type_specs)
        self.type_names = [spec.type_name for spec in self.type_specs]
        self.records_by_scene = records_by_scene
        self.all_scene_ids = sorted(records_by_scene)
        self.scale_by_scene = {scene_id: _scale_label_from_scene_id(scene_id) for scene_id in self.all_scene_ids}
        self.scale_labels = self._build_scale_labels()
        self.complete_scene_ids = self._build_complete_scene_ids()
        # Show every scene that has at least one saved grasp in one_scene mode.
        # Missing grasp types are reported in the status panel by _render_one_scene().
        self.scene_ids = self.all_scene_ids
        self.max_grasps = int(_cfg_get(cfg.task, "max_grasps", 5))
        self.scene_batch_size = int(_cfg_get(cfg.task, "scene_batch_size", 5))
        self.spacing = float(_cfg_get(cfg.task, "spacing", 0.55))
        self.pose_index = int(_cfg_get(cfg.task, "pose_index", -1))
        self.pose_labels = self._build_pose_labels()
        self.pose_label = (
            "pose_1"
            if (not bool(_cfg_get(cfg.task, "b_opt_process", False)) and "pose_1" in self.pose_labels)
            else self.pose_labels[-1]
        )
        self.device = str(cfg.device)
        self.mode = "one_scene"
        self.scale_label = self._initial_scale_label()
        self.scene_index = 0
        self.type_index = 0
        self.scene_ids = self._scene_ids_for_current_mode()
        self.batch_offset = 0
        self.sample_offset = 0
        self.handles = []
        self.visualizers: Dict[str, Visualizer] = {}
        self.grasp_data_cache = OrderedDict()
        self.object_mesh_cache = OrderedDict()
        self.robot_use_root_pose_cache: Dict[str, bool] = {}
        self.grasp_data_cache_size = int(_cfg_get(cfg.task, "grasp_data_cache_size", 64))
        self.object_mesh_cache_size = int(_cfg_get(cfg.task, "object_mesh_cache_size", 256))
        self.scene_dropdown_window_size = int(_cfg_get(cfg.task, "scene_dropdown_window_size", 200))
        self.random_generator = np.random.default_rng()
        self._rendering_gui = False

    def _cache_get(self, cache: OrderedDict, key):
        """Read one LRU cache entry and mark it recently used.

        Args:
            cache: Ordered dictionary used as an LRU cache.
            key: Cache key to read.

        Returns:
            Cached value, or ``None`` when the key is absent.
        """
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_put(self, cache: OrderedDict, key, value, max_size: int) -> None:
        """Store one LRU cache entry and evict old entries past the cap.

        Args:
            cache: Ordered dictionary used as an LRU cache.
            key: Cache key to write.
            value: Value to store.
            max_size: Maximum retained entries. Values less than one disable retention.

        Returns:
            None.
        """
        if max_size <= 0:
            return
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _load_grasp_data(self, record: GraspRecord) -> Dict:
        """Load a saved grasp npy once and reuse it across adjacent renders.

        Args:
            record: Grasp record whose file should be loaded.

        Returns:
            Loaded grasp dictionary.
        """
        cached = self._cache_get(self.grasp_data_cache, record.file_path)
        if cached is not None:
            return cached
        grasp_data = np.load(record.file_path, allow_pickle=True).item()
        self._cache_put(self.grasp_data_cache, record.file_path, grasp_data, self.grasp_data_cache_size)
        return grasp_data

    def _load_object_mesh_cached(self, grasp_data: Dict) -> tm.Trimesh:
        """Load one scene object mesh with an LRU cache keyed by scene path.

        Args:
            grasp_data: Loaded grasp dictionary containing the source scene path.

        Returns:
            Transformed object mesh for the scene.
        """
        scene_path = _scene_path_from_grasp_data(grasp_data)
        cached = self._cache_get(self.object_mesh_cache, scene_path)
        if cached is not None:
            return cached
        obj_mesh = _load_object_mesh(grasp_data)
        self._cache_put(self.object_mesh_cache, scene_path, obj_mesh, self.object_mesh_cache_size)
        return obj_mesh

    def _robot_use_root_pose(self, record: GraspRecord) -> bool:
        """Read and cache whether a robot config stores root pose in qpos.

        Args:
            record: Grasp record whose robot config should be inspected.

        Returns:
            Whether the saved robot pose already includes the root pose.
        """
        if record.robot_file not in self.robot_use_root_pose_cache:
            robot_config_data = load_yaml(join_path(get_robot_path(), record.robot_file))
            self.robot_use_root_pose_cache[record.robot_file] = bool(
                robot_config_data["robot_cfg"]["kinematics"]["use_root_pose"]
            )
        return self.robot_use_root_pose_cache[record.robot_file]

    def _random_scene_index(self) -> int:
        """Sample a random scene index, avoiding the current index when possible.

        Args:
            None.

        Returns:
            Random scene index in the current mode's scene list.
        """
        scene_count = len(self.scene_ids)
        if scene_count <= 1:
            return 0
        sampled_index = int(self.random_generator.integers(0, scene_count - 1))
        if sampled_index >= self.scene_index:
            sampled_index += 1
        return sampled_index

    def _build_scale_labels(self) -> List[str]:
        """Build the scale dropdown options from indexed scene ids.

        Args:
            None.

        Returns:
            Ordered scale labels, with ``all`` first.
        """
        labels = sorted(set(self.scale_by_scene.values()), key=_scale_sort_key)
        return [ALL_SCALE_LABEL] + labels

    def _initial_scale_label(self) -> str:
        """Read the optional initial object scale from the visualize config.

        Args:
            None.

        Returns:
            Initial scale dropdown label. Falls back to ``all`` when the
            requested scale is unavailable.
        """
        scale_label = _normalize_scale_label(_cfg_get(self.cfg.task, "object_scale", ALL_SCALE_LABEL))
        if scale_label in self.scale_labels:
            return scale_label
        logging.warning("Requested object_scale=%s is unavailable; using all scales.", scale_label)
        return ALL_SCALE_LABEL

    def _build_complete_scene_ids(self) -> List[str]:
        """Find scenes that have saved records for every requested grasp type.

        Args:
            None.

        Returns:
            Sorted scene ids whose record set contains all requested grasp types.
        """
        required_types = set(self.type_names)
        return [
            scene_id
            for scene_id in sorted(self.records_by_scene)
            if required_types.issubset(set(self.records_by_scene.get(scene_id, {})))
        ]

    def _scene_ids_for_current_mode(self) -> List[str]:
        """Return the scene ids that should be browsed by the active mode.

        Args:
            None.

        Returns:
            Scene ids for the current GUI mode.
        """
        # Scale filtering is applied before mode-specific filtering so both
        # visualization modes only browse scenes from the selected object scale.
        scale_scene_ids = [
            scene_id
            for scene_id in self.all_scene_ids
            if self.scale_label == ALL_SCALE_LABEL or self.scale_by_scene.get(scene_id) == self.scale_label
        ]
        if self.mode == "one_scene":
            return scale_scene_ids
        type_name = self.type_names[self.type_index]
        type_scene_ids = [
            scene_id for scene_id in scale_scene_ids if type_name in self.records_by_scene.get(scene_id, {})
        ]
        return type_scene_ids if type_scene_ids else scale_scene_ids

    def _scene_dropdown_options(self) -> List[str]:
        """Return a small scene dropdown window around the current scene.

        Args:
            None.

        Returns:
            Scene id options to show in the GUI dropdown.
        """
        if not self.scene_ids:
            return []
        if self.scene_dropdown_window_size <= 0 or len(self.scene_ids) <= self.scene_dropdown_window_size:
            return list(self.scene_ids)
        window_size = max(1, self.scene_dropdown_window_size)
        half_window = window_size // 2
        start = max(0, self.scene_index - half_window)
        start = min(start, max(0, len(self.scene_ids) - window_size))
        return list(self.scene_ids[start : start + window_size])

    def _sync_scene_controls(self, scene_handle, scene_index_handle, desired_scene_id: Optional[str] = None) -> None:
        """Synchronize scene dropdown and index controls with app state.

        Args:
            scene_handle: Viser GUI dropdown handle for nearby scene selection.
            scene_index_handle: Viser GUI number handle for direct scene index jumps.
            desired_scene_id: Scene id to preserve when it is available.

        Returns:
            None.
        """
        self.scene_ids = self._scene_ids_for_current_mode()
        if not self.scene_ids:
            scene_handle.options = []
            scene_index_handle.value = 0
            return
        if desired_scene_id in self.scene_ids:
            self.scene_index = self.scene_ids.index(desired_scene_id)
        else:
            self.scene_index = min(self.scene_index, len(self.scene_ids) - 1)
        scene_options = self._scene_dropdown_options()
        if tuple(scene_handle.options) != tuple(scene_options):
            scene_handle.options = scene_options
        next_scene_id = self.scene_ids[self.scene_index]
        if scene_handle.value != next_scene_id:
            scene_handle.value = next_scene_id
        if hasattr(scene_index_handle, "max"):
            scene_index_handle.max = max(0, len(self.scene_ids) - 1)
        if int(scene_index_handle.value) != self.scene_index:
            scene_index_handle.value = self.scene_index

    def start(self):
        """Start the viser server and block forever.

        Args:
            None.

        Returns:
            None.
        """
        host = str(_cfg_get(self.cfg.task, "host", "0.0.0.0"))
        port = int(_cfg_get(self.cfg.task, "port", 8081))
        server = viser.ViserServer(host=host, port=port, label="HUGS-BODex Grasp Visualizer")
        server.scene.world_axes.visible = True
        server.scene.world_axes.axes_length = 0.04
        server.scene.world_axes.axes_radius = 0.001
        server.scene.world_axes.origin_radius = 0.004

        mode = server.gui.add_dropdown("Visualization mode", ("one_scene", "one_grasp_type"), initial_value=self.mode)
        object_scale = server.gui.add_dropdown("Object scale", self.scale_labels, initial_value=self.scale_label)
        scene = server.gui.add_dropdown(
            "Scene",
            self._scene_dropdown_options(),
            initial_value=self.scene_ids[self.scene_index],
        )
        scene_index = server.gui.add_number(
            "Scene index",
            self.scene_index,
            min=0,
            max=max(0, len(self.scene_ids) - 1),
            step=1,
        )
        grasp_type = server.gui.add_dropdown("Grasp type", self.type_names, initial_value=self.type_names[self.type_index])
        pose = server.gui.add_dropdown("Pose", self.pose_labels, initial_value=self.pose_label)
        next_button = server.gui.add_button("Next Batch")
        next_scene_button = server.gui.add_button("Next Scene")
        status = server.gui.add_markdown("")

        def render(desired_scene_id: Optional[str] = None):
            if self._rendering_gui:
                return
            self._rendering_gui = True
            try:
                previous_scene_id = desired_scene_id if desired_scene_id is not None else str(scene.value)
                self.mode = str(mode.value)
                self.scale_label = str(object_scale.value)
                self.type_index = self.type_names.index(str(grasp_type.value))
                self._sync_scene_controls(scene, scene_index, previous_scene_id)
                self.pose_label = str(pose.value)
                self._render_scene(server, status)
            finally:
                self._rendering_gui = False

        @mode.on_update
        def _(_event):
            self.batch_offset = 0
            self.sample_offset = 0
            render()

        @object_scale.on_update
        def _(_event):
            self.scene_index = 0
            self.batch_offset = 0
            self.sample_offset = 0
            render()

        @scene.on_update
        def _(_event):
            self.batch_offset = 0
            self.sample_offset = 0
            render(desired_scene_id=str(scene.value))

        @scene_index.on_update
        def _(_event):
            target_index = int(np.clip(round(float(scene_index.value)), 0, len(self.scene_ids) - 1))
            self.scene_index = target_index
            self.batch_offset = 0
            self.sample_offset = 0
            render(desired_scene_id=self.scene_ids[target_index])

        @grasp_type.on_update
        def _(_event):
            self.batch_offset = 0
            self.sample_offset = 0
            render()

        @pose.on_update
        def _(_event):
            render()

        @next_button.on_click
        def _(_event):
            if self.mode == "one_scene":
                max_grasp_count = self._current_scene_max_grasp_count()
                if max_grasp_count > self.max_grasps:
                    self.sample_offset += self.max_grasps
                    if self.sample_offset >= max_grasp_count:
                        self.sample_offset = 0
                else:
                    self.sample_offset = 0
            else:
                self.batch_offset = (self.batch_offset + self.scene_batch_size) % len(self.scene_ids)
            render()

        @next_scene_button.on_click
        def _(_event):
            if self.mode == "one_scene":
                self.scene_index = self._random_scene_index()
                self.sample_offset = 0
                render(desired_scene_id=self.scene_ids[self.scene_index])
            else:
                self.batch_offset = self._random_scene_index()
                render()

        render()
        browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        print(f"Viser web UI is running at http://{browser_host}:{port}")
        while True:
            time.sleep(1.0)

    def _build_pose_labels(self) -> Sequence[str]:
        """Build pose choices shown in the GUI.

        Args:
            None.

        Returns:
            Pose label sequence.
        """
        if bool(_cfg_get(self.cfg.task, "b_opt_process", False)):
            opt_progress = _cfg_get(self.cfg.task, "opt_progress", [0.0, 0.8, 1.0])
            return [f"opt_{float(value):.2f}" for value in opt_progress]
        return ["pose_0", "pose_1", "pose_2"]

    def _pose_index_for_count(self, pose_count: int) -> int:
        """Resolve the selected GUI pose label for a trajectory length.

        Args:
            pose_count: Number of poses stored in one saved grasp trajectory.

        Returns:
            Valid pose index in ``[0, pose_count)``.
        """
        if pose_count <= 0:
            raise ValueError(f"pose_count must be positive, got {pose_count}")
        if self.pose_label.startswith("opt_"):
            progress = float(self.pose_label.split("_", 1)[1])
            return int(np.clip(round((pose_count - 1) * progress), 0, pose_count - 1))
        if self.pose_label.startswith("pose_"):
            pose_index = int(self.pose_label.split("_", 1)[1])
            return int(np.clip(pose_index, 0, pose_count - 1))
        pose_index = self.pose_index if self.pose_index >= 0 else pose_count + self.pose_index
        return int(np.clip(pose_index, 0, pose_count - 1))

    def _clear_scene(self):
        """Remove previously rendered scene nodes.

        Args:
            None.

        Returns:
            None.
        """
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []

    def _current_scene_max_grasp_count(self) -> int:
        """Read the largest grasp count among records in the current one-scene view.

        Args:
            None.

        Returns:
            Largest saved grasp count for the current scene.
        """
        scene_id = self.scene_ids[self.scene_index]
        records = self.records_by_scene.get(scene_id, {})
        counts = []
        for record in records.values():
            try:
                counts.append(_ensure_record_metadata(record, self._load_grasp_data(record)))
            except Exception as exc:
                logging.warning("Skip unreadable grasp file %s: %s", record.file_path, exc)
        return max(counts, default=0)

    def _normalize_sample_offset(self, max_grasp_count: int) -> None:
        """Keep the one-scene sample offset inside the current scene's grasp range.

        Args:
            max_grasp_count: Largest saved grasp count for the current scene.

        Returns:
            None.
        """
        if max_grasp_count <= 0 or self.sample_offset >= max_grasp_count:
            self.sample_offset = 0

    def _render_scene(self, server, status):
        """Render the selected view mode.

        Args:
            server: Viser server.
            status: GUI markdown handle used for captions.

        Returns:
            None.
        """
        self._clear_scene()
        if self.mode == "one_scene":
            caption = self._render_one_scene(server)
        else:
            caption = self._render_one_grasp_type(server)
        status.content = caption

    def _render_one_scene(self, server) -> str:
        """Render all grasp types for one scene.

        Args:
            server: Viser server.

        Returns:
            Markdown caption.
        """
        scene_id = self.scene_ids[self.scene_index]
        records = self.records_by_scene.get(scene_id, {})
        max_grasp_count = self._current_scene_max_grasp_count()
        self._normalize_sample_offset(max_grasp_count)
        sample_end = min(self.sample_offset + self.max_grasps, max_grasp_count)
        caption_lines = [
            f"### one_scene: `{scene_id}`",
            f"Object scale: `{self.scale_label}`",
            f"Sample batch: `{self.sample_offset}:{sample_end}` / `{max_grasp_count}`",
            "",
        ]
        for row, type_name in enumerate(self.type_names):
            record = records.get(type_name)
            if record is None:
                caption_lines.append(f"- `{type_name}`: missing")
                continue
            try:
                grasp_data = self._load_grasp_data(record)
                grasp_count = _ensure_record_metadata(record, grasp_data)
            except Exception as exc:
                logging.warning("Skip unreadable grasp file %s: %s", record.file_path, exc)
                caption_lines.append(f"- `{type_name}`: unreadable")
                continue
            if self.sample_offset >= grasp_count:
                caption_lines.append(
                    f"- `{type_name}`: no grasps in this sample batch "
                    f"({grasp_count}/{record.total_budget or '?'} grasps saved)"
                )
                continue
            type_end = min(self.sample_offset + self.max_grasps, grasp_count)
            caption_lines.append(
                f"- `{type_name}`: showing `{self.sample_offset}:{type_end}` of "
                f"{grasp_count}/{record.total_budget or '?'} grasps"
            )
            self._add_record_meshes(
                server,
                record,
                row=row,
                row_label=type_name,
                sample_offset=self.sample_offset,
                grasp_data=grasp_data,
            )
        return "\n".join(caption_lines)

    def _render_one_grasp_type(self, server) -> str:
        """Render one grasp type across a scene batch.

        Args:
            server: Viser server.

        Returns:
            Markdown caption.
        """
        type_name = self.type_names[self.type_index]
        batch = [self.scene_ids[(self.batch_offset + i) % len(self.scene_ids)] for i in range(self.scene_batch_size)]
        caption_lines = [
            f"### one_grasp_type: `{type_name}`",
            f"Object scale: `{self.scale_label}`",
            f"Batch offset: `{self.batch_offset}`",
            "",
        ]
        for row, scene_id in enumerate(batch):
            record = self.records_by_scene.get(scene_id, {}).get(type_name)
            if record is None:
                caption_lines.append(f"- `{scene_id}`: missing")
                continue
            try:
                grasp_data = self._load_grasp_data(record)
                grasp_count = _ensure_record_metadata(record, grasp_data)
            except Exception as exc:
                logging.warning("Skip unreadable grasp file %s: %s", record.file_path, exc)
                caption_lines.append(f"- `{scene_id}`: unreadable")
                continue
            caption_lines.append(f"- `{scene_id}`: {grasp_count}/{record.total_budget or '?'} grasps")
            self._add_record_meshes(
                server,
                record,
                row=row,
                row_label=os.path.basename(scene_id),
                sample_offset=0,
                grasp_data=grasp_data,
            )
        return "\n".join(caption_lines)

    def _add_record_meshes(
        self,
        server,
        record: GraspRecord,
        row: int,
        row_label: str,
        sample_offset: int,
        grasp_data: Optional[Dict] = None,
    ):
        """Add object and robot meshes for one record row.

        Args:
            server: Viser server.
            record: Grasp record to visualize.
            row: Row index in the current view.
            row_label: Text label for the row.
            sample_offset: First saved grasp sample to show in this row.
            grasp_data: Optional loaded grasp dictionary from the caller.

        Returns:
            None.
        """
        if grasp_data is None:
            grasp_data = self._load_grasp_data(record)
        _ensure_record_metadata(record, grasp_data)
        syn_pose = torch.from_numpy(np.asarray(grasp_data["robot_pose"])).to(device=self.device).squeeze(0)
        sample_count, pose_count, n_dof = syn_pose.shape[-3:]
        pose_index = self._pose_index_for_count(pose_count)
        start = min(max(int(sample_offset), 0), sample_count)
        sample_indices = list(range(start, min(start + self.max_grasps, sample_count)))
        if not sample_indices:
            return

        use_root_pose = self._robot_use_root_pose(record)
        robot_pose = syn_pose[sample_indices, pose_index, :].clone().view(-1, n_dof)
        if not use_root_pose:
            base_pose = torch.tensor([[0, 0, 0, 1, 0, 0, 0]], device=self.device, dtype=robot_pose.dtype).repeat(
                robot_pose.shape[0], 1
            )
            robot_pose = torch.cat([base_pose, robot_pose], dim=-1)

        visualizer = self.visualizers.get(record.type_name)
        if visualizer is None:
            visualizer = _make_visualizer(record, self.device)
            self.visualizers[record.type_name] = visualizer
        visualizer.set_robot_parameters(robot_pose, joint_names=grasp_data["joint_names"])
        obj_mesh = self._load_object_mesh_cached(grasp_data)
        color = TYPE_COLORS.get(record.type_name, [0.8, 0.8, 0.8, 0.72])

        label_pos = np.array([-0.25, row * self.spacing, 0.32], dtype=np.float64)
        self.handles.append(
            server.scene.add_label(
                f"/labels/{row}",
                row_label,
                position=label_pos,
                font_size_mode="screen",
                font_screen_scale=0.8,
            )
        )
        for col, sample_idx in enumerate(sample_indices):
            offset = np.array([col * self.spacing, row * self.spacing, 0.0], dtype=np.float64)
            shifted_obj = obj_mesh.copy()
            shifted_obj.apply_translation(offset)
            robot_mesh = visualizer.get_robot_trimesh_data(col, color=color)
            robot_mesh.apply_translation(offset)
            base_name = f"/view/r{row}/c{col}"
            self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/object", shifted_obj))
            self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/robot", robot_mesh))
            self.handles.append(
                server.scene.add_label(
                    f"{base_name}/label",
                    f"g{sample_idx}",
                    position=offset + np.array([0.0, 0.0, 0.24]),
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="bottom-center",
                )
            )


def task_visualize(cfg):
    """Launch a viser Web UI for saved HUGS-BODex grasp results.

    Args:
        cfg: Hydra config.

    Returns:
        None. This task blocks while the Web UI server is running.
    """
    suite_config = _load_suite_config(cfg.task)
    type_specs = _build_type_specs(cfg, suite_config)
    records_by_scene = _build_record_index(type_specs)
    app = GraspViserApp(cfg, type_specs, records_by_scene)
    app.start()

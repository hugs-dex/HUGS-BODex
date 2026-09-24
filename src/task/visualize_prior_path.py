import glob
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import trimesh as tm
import viser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from curobo.util.visualizer import Visualizer
from curobo.util.artifact_path import resolve_artifact_path
from curobo.util_file import get_assets_path, get_manip_configs_path, get_robot_path, join_path, load_yaml
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW


GRASP_TYPE_NAMES = {
    1: "1_right_two",
    2: "2_right_three",
    3: "3_right_full",
    4: "4_both_three",
    5: "5_both_full",
}
SINGLE_HAND_TYPE_IDS = {1, 2, 3}
BOTH_THREE_TYPE_ID = 4
FINGERTIP_JOINT_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
GRASP_TYPE_FINGERS = {
    1: {"right": ["thumb", "index"], "left": []},
    2: {"right": ["thumb", "index", "middle"], "left": []},
    3: {"right": ["thumb", "index", "middle", "ring", "pinky"], "left": []},
    4: {"right": ["thumb", "index", "middle"], "left": ["thumb", "index", "middle"]},
    5: {
        "right": ["thumb", "index", "middle", "ring", "pinky"],
        "left": ["thumb", "index", "middle", "ring", "pinky"],
    },
}


@dataclass
class PriorPathRecord:
    grasp_file: str
    scene_id: str
    type_id: int
    type_name: str
    robot_file: str
    manip_cfg_file: str
    root_label: str


def _cfg_get(cfg, key, default=None):
    """Read a config key with a Python default.

    Args:
        cfg: OmegaConf section or plain mapping.
        key: Key to read.
        default: Value returned when the key is absent.

    Returns:
        Config value or ``default``.
    """
    return cfg[key] if key in cfg else default


def _normalize_grasp_types(grasp_types) -> Optional[List[str]]:
    """Normalize an optional grasp-type selection.

    Args:
        grasp_types: Config value containing ``all`` or type names/ids.

    Returns:
        Requested type tokens, or ``None`` when all suite types are selected.
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
    """Check whether a suite grasp type should be indexed.

    Args:
        type_name: Suite grasp type name.
        type_config: Suite grasp type config dictionary.
        requested: Optional requested names or ids.

    Returns:
        Whether the type should be included.
    """
    if requested is None:
        return True
    type_id = str(type_config.get("type_id", str(type_name).split("_", 1)[0]))
    return type_name in requested or type_id in requested


def _target_fingertip_indices(grasp_type_id: int, side: str) -> List[int]:
    """Return MANO fingertip joints selected by the predicted grasp type.

    Args:
        grasp_type_id: Predicted grasp type id in ``[1, 5]``.
        side: Hand side, either ``right`` or ``left``.

    Returns:
        MANO joint indices for fingertips that participate in the predicted
        contact mode.
    """
    finger_names = GRASP_TYPE_FINGERS.get(int(grasp_type_id), {}).get(str(side), [])
    return [FINGERTIP_JOINT_INDICES[name] for name in finger_names]


def _resolve_path(path: str) -> str:
    """Resolve a possibly repo-relative path.

    Args:
        path: Absolute path, HUGS-BODex-relative path, or cwd-relative path.

    Returns:
        Absolute path to use for file loading.
    """
    path = resolve_artifact_path(str(path))
    if os.path.isabs(path):
        return path
    if path.startswith("assets/"):
        asset_relative = os.path.join(get_assets_path(), path[len("assets/") :])
        if os.path.exists(asset_relative):
            return asset_relative
    repo_relative = os.path.join(REPO_ROOT, path)
    if os.path.exists(repo_relative):
        return repo_relative
    asset_relative = os.path.join(get_assets_path(), path)
    if os.path.exists(asset_relative):
        return asset_relative
    return os.path.abspath(path)


def _load_suite_config(task_cfg) -> Dict:
    """Load the suite config that defines grasp types and manipulation configs.

    Args:
        task_cfg: Visualize-prior-path task config.

    Returns:
        Loaded suite config dictionary.
    """
    suite_config = str(_cfg_get(task_cfg, "suite_config", "sim_shadow.yml"))
    return load_yaml(join_path(get_manip_configs_path(), suite_config))


def _output_graspdata_dir(output_path: str, manip_cfg_file: str, exp_name: str) -> str:
    """Resolve the graspdata directory for one suite grasp type.

    Args:
        output_path: Root HUGS-BODex output path.
        manip_cfg_file: Manipulation config file from the suite.
        exp_name: Experiment name from top-level ``name``.

    Returns:
        Directory containing ``*_grasp.npy`` results for the type.
    """
    manip_prefix, _ = os.path.splitext(str(manip_cfg_file))
    return _resolve_path(os.path.join(str(output_path), manip_prefix, str(exp_name), "graspdata"))


def _scene_id_from_grasp_path(grasp_file: str) -> str:
    """Infer the DGN2k scene id from a ``*_grasp.npy`` path.

    Args:
        grasp_file: Saved HUGS-BODex grasp result path.

    Returns:
        Scene id in ``object/tabletop_ur10e/scaleXXX_poseYYY_0`` form.
    """
    rel = grasp_file.replace("\\", "/").split("/graspdata/", 1)[-1]
    return rel[: -len("_grasp.npy")]


def _load_npy_dict(path: str) -> Dict:
    """Load a numpy object dictionary.

    Args:
        path: Path to an ``.npy`` file saved as a Python dictionary.

    Returns:
        Loaded dictionary.
    """
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected an object-dict npy file, got {path}")


def _human_prior_file_from_grasp_data(grasp_data: Dict) -> str:
    """Read and resolve the human-prior source file from grasp data.

    Args:
        grasp_data: Loaded HUGS-BODex grasp result.

    Returns:
        Absolute path to the per-scene human-prior export.
    """
    value = grasp_data["human_prior_scene_file"][0]
    return _resolve_path(str(value))


def _scene_path_from_grasp_data(grasp_data: Dict) -> str:
    """Read and resolve the source scene config path.

    Args:
        grasp_data: Loaded HUGS-BODex grasp result.

    Returns:
        Absolute scene config path.
    """
    return _resolve_path(str(grasp_data["scene_path"][0]))


def _type_id_from_grasp_data(grasp_data: Dict) -> int:
    """Read the human-prior grasp type id stored in a result file.

    Args:
        grasp_data: Loaded HUGS-BODex grasp result.

    Returns:
        Integer grasp type id.
    """
    return int(np.asarray(grasp_data["human_prior_type_id"]).reshape(-1)[0])


def _type_name_from_grasp_data(grasp_data: Dict, type_id: int) -> str:
    """Read the grasp type name, falling back to the canonical name table.

    Args:
        grasp_data: Loaded HUGS-BODex grasp result.
        type_id: Integer grasp type id.

    Returns:
        Grasp type name.
    """
    if "human_prior_type_name" in grasp_data:
        return str(grasp_data["human_prior_type_name"][0])
    return GRASP_TYPE_NAMES.get(int(type_id), f"type_{type_id}")


def _load_object_mesh_from_scene(scene_path: str) -> tm.Trimesh:
    """Load the transformed object mesh for one DGN2k scene.

    Args:
        scene_path: Scene config ``.npy`` path.

    Returns:
        Object mesh transformed into the scene frame.
    """
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


def _load_scene_point_cloud(prior_data: Dict, num_points: int) -> np.ndarray:
    """Load the object point cloud saved with a human-prior export.

    Args:
        prior_data: Per-scene human-prior export dictionary.
        num_points: Number of points to display. Values <= 0 keep all points.

    Returns:
        Point cloud in the scene frame.
    """
    pc_path = _resolve_path(str(prior_data["pc_path"]))
    pc = np.asarray(np.load(pc_path, allow_pickle=True), dtype=np.float32)
    if int(num_points) > 0 and pc.shape[0] > int(num_points):
        idx = np.random.choice(pc.shape[0], int(num_points), replace=False)
        pc = pc[idx]
    elif int(num_points) > 0 and pc.shape[0] < int(num_points):
        idx = np.random.choice(pc.shape[0], int(num_points), replace=True)
        pc = pc[idx]

    scene_path = _resolve_path(str(prior_data["scene_path"]))
    scene_data = np.load(scene_path, allow_pickle=True).item()
    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = np.asarray(scene_data["scene"][obj_name]["scale"], dtype=np.float32)
    if obj_scale.ndim == 0:
        obj_scale = np.full((3,), float(obj_scale), dtype=np.float32)
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    pc = pc * obj_scale.reshape(1, 3)
    pc_h = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=np.float32)], axis=1)
    return (np.asarray(obj_transform, dtype=np.float32) @ pc_h.T).T[:, :3]


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert one WXYZ quaternion into a rotation matrix.

    Args:
        quat: Quaternion tensor with shape ``(4,)``.

    Returns:
        Rotation matrix with shape ``(3, 3)``.
    """
    quat = quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )


def _matrix_to_axis_angle(rot: torch.Tensor) -> torch.Tensor:
    """Convert one rotation matrix into an axis-angle vector.

    Args:
        rot: Rotation matrix with shape ``(3, 3)``.

    Returns:
        Axis-angle tensor with shape ``(3,)``.
    """
    cos_angle = torch.clamp((torch.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    axis = torch.stack([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    denom = torch.clamp(2.0 * torch.sin(angle), min=1e-8)
    axis = axis / denom
    return axis * angle


def _create_mano_layers(device: str, mano_root: str):
    """Create MANO layers lazily for human-prior rendering.

    Args:
        device: Torch device string.
        mano_root: Directory containing MANO model files.

    Returns:
        Mapping from ``right``/``left`` to MANO layers.
    """
    # chumpy, which is used by manopth to load MANO pickle files, still imports
    # deprecated NumPy scalar aliases. Define them before importing manopth so
    # the viewer can run in modern conda environments such as anyscalelearn.
    for alias, scalar_type in (
        ("bool", np.bool_),
        ("int", np.int_),
        ("float", np.float64),
        ("complex", np.complex128),
        ("object", np.object_),
        ("str", np.str_),
        ("unicode", np.str_),
    ):
        if alias not in np.__dict__:
            setattr(np, alias, scalar_type)

    try:
        from manopth.manolayer import ManoLayer
    except ModuleNotFoundError:
        package_root = os.path.dirname(os.path.dirname(str(mano_root)))
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from manopth.manolayer import ManoLayer

    return {
        side: ManoLayer(
            center_idx=0,
            mano_root=mano_root,
            side=side,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(device)
        for side in ("right", "left")
    }


def _wrist_translation_from_index_mcp(target_pos: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Compute MANO wrist translation from a target index-MCP position.

    Args:
        target_pos: Target index-MCP position in meters.
        joints: MANO joints in millimeters before translation.

    Returns:
        Wrist translation in meters.
    """
    index_mcp_joint = joints[5] / 1000.0
    return target_pos - index_mcp_joint


def _make_mano_meshes(
    prior_data: Dict,
    type_index: int,
    sample_index: int,
    grasp_type_id: int,
    mano_layers,
    device: str,
) -> List[tm.Trimesh]:
    """Build MANO meshes for one selected human-prior sample.

    Args:
        prior_data: Per-scene human-prior export dictionary.
        type_index: Row index for the grasp type in the prior export.
        sample_index: Sample index selected by ``human_prior_sample_indices``.
        grasp_type_id: Predicted grasp type id used to choose red fingertip
            markers.
        mano_layers: Mapping returned by ``_create_mano_layers``.
        device: Torch device string.

    Returns:
        List of trimesh objects representing active human hands and axes.
    """
    position_key = str(prior_data.get("export_position_key", "index_mcp_pos"))
    if position_key not in prior_data:
        position_key = "index_mcp_pos" if "index_mcp_pos" in prior_data else "wrist_pos"
    grasp_pos_source = str(prior_data.get("grasp_pos_source", "index_mcp"))
    positions = np.asarray(prior_data[position_key], dtype=np.float32)[type_index, sample_index]
    wrist_quats = np.asarray(prior_data["wrist_quat"], dtype=np.float32)[type_index, sample_index]
    active_mask = np.asarray(prior_data["active_hand_mask"], dtype=bool)[type_index, sample_index]

    side_names = ("right", "left")
    colors = ([180, 200, 255, 220], [210, 190, 250, 220])
    meshes: List[tm.Trimesh] = []
    for hand_idx, side in enumerate(side_names):
        if hand_idx >= len(active_mask) or not bool(active_mask[hand_idx]):
            continue
        target_pos = torch.as_tensor(positions[hand_idx], dtype=torch.float32, device=device)
        quat = torch.as_tensor(wrist_quats[hand_idx], dtype=torch.float32, device=device)
        rot = _quat_wxyz_to_matrix(quat)
        mano_params = torch.cat(
            [_matrix_to_axis_angle(rot).reshape(1, 3), torch.zeros((1, 45), dtype=torch.float32, device=device)],
            dim=-1,
        )
        mano_layer = mano_layers[side]
        verts, joints = mano_layer(mano_params, th_betas=torch.zeros((1, 10), dtype=torch.float32, device=device))
        if grasp_pos_source == "index_mcp":
            wrist_trans = _wrist_translation_from_index_mcp(target_pos, joints[0])
        else:
            wrist_trans = target_pos
        verts_np = ((verts[0] / 1000.0) + wrist_trans).detach().cpu().numpy()
        joints_np = ((joints[0] / 1000.0) + wrist_trans).detach().cpu().numpy()
        faces_np = mano_layer.th_faces.detach().cpu().numpy()
        meshes.append(tm.Trimesh(vertices=verts_np, faces=faces_np, face_colors=colors[hand_idx], process=False))
        for fingertip_index in _target_fingertip_indices(int(grasp_type_id), side):
            sphere = tm.creation.uv_sphere(radius=0.005)
            sphere.visual.face_colors = [255, 0, 0, 255]
            sphere.apply_translation(joints_np[fingertip_index])
            meshes.append(sphere)

        axis_pose = np.eye(4, dtype=np.float32)
        axis_pose[:3, :3] = rot.detach().cpu().numpy()
        axis_pose[:3, 3] = target_pos.detach().cpu().numpy()
        meshes.append(tm.creation.axis(transform=axis_pose, origin_size=0.008, axis_radius=0.002, axis_length=0.07))
    return meshes


def _type_index_in_prior(prior_data: Dict, type_id: int) -> int:
    """Find the row index for a grasp type in a prior export.

    Args:
        prior_data: Per-scene human-prior export dictionary.
        type_id: Grasp type id to locate.

    Returns:
        Row index in prior pose arrays.
    """
    type_ids = np.asarray(prior_data.get("grasp_type_ids", list(GRASP_TYPE_NAMES)), dtype=np.int64).reshape(-1)
    matches = np.where(type_ids == int(type_id))[0]
    if len(matches) == 0:
        raise KeyError(f"grasp type id {type_id} not found in prior export")
    return int(matches[0])


def _pose_index_for_type(type_id: int, pose_count: int) -> int:
    """Resolve the final robot pose index for one grasp type.

    Args:
        type_id: Grasp type id.
        pose_count: Number of saved optimization states.

    Returns:
        Pose index corresponding to opt=1.0 for single hand and opt=0.8 for
        bimanual grasps.
    """
    progress = 1.0 if int(type_id) in SINGLE_HAND_TYPE_IDS else 0.8
    return int(np.clip(round((pose_count - 1) * progress), 0, pose_count - 1))


def _add_root_pose_if_needed(robot_pose: torch.Tensor, robot_file: str) -> torch.Tensor:
    """Add an identity root pose when a robot config stores hand qpos only.

    Args:
        robot_pose: Robot qpos tensor with shape ``(N, D)``.
        robot_file: Robot config file name.

    Returns:
        Robot qpos tensor compatible with ``Visualizer``.
    """
    robot_config_data = load_yaml(join_path(get_robot_path(), robot_file))
    use_root_pose = bool(robot_config_data["robot_cfg"]["kinematics"]["use_root_pose"])
    if use_root_pose:
        return robot_pose
    base_pose = torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=robot_pose.dtype, device=robot_pose.device).repeat(
        robot_pose.shape[0], 1
    )
    return torch.cat([base_pose, robot_pose], dim=-1)


def _make_visualizer(robot_file: str, device: str) -> Visualizer:
    """Create a robot visualizer for a robot config.

    Args:
        robot_file: Robot config file name.
        device: Torch device string.

    Returns:
        Visualizer instance.
    """
    robot_config_data = load_yaml(join_path(get_robot_path(), robot_file))
    urdf_file = robot_config_data["robot_cfg"]["kinematics"]["urdf_path"]
    robot_urdf_path = join_path(get_assets_path(), urdf_file)
    mesh_dir_path = os.path.dirname(robot_urdf_path)
    return Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path, device=device)


def _offset_mesh(mesh: tm.Trimesh, offset: np.ndarray) -> tm.Trimesh:
    """Copy and translate a mesh.

    Args:
        mesh: Source mesh.
        offset: Translation vector.

    Returns:
        Translated mesh copy.
    """
    shifted = mesh.copy()
    shifted.apply_translation(offset)
    return shifted


class PriorPathViserApp:
    def __init__(self, cfg, records: Sequence[PriorPathRecord]):
        """Create app state for the prior-to-robot path viewer.

        Args:
            cfg: Hydra config.
            records: Indexed grasp result records.

        Returns:
            None.
        """
        self.cfg = cfg
        self.records = list(records)
        self.batch_size = int(_cfg_get(cfg.task, "batch_size", 5))
        self.spacing_x = float(_cfg_get(cfg.task, "spacing_x", 0.62))
        self.spacing_y = float(_cfg_get(cfg.task, "spacing_y", 0.42))
        self.num_points = int(_cfg_get(cfg.task, "num_points", 1024))
        self.seed_index = int(_cfg_get(cfg.task, "seed_index", -1))
        self.device = str(cfg.device)
        self.show_text = bool(_cfg_get(cfg.task, "show_text", False))
        self.show_caption = bool(_cfg_get(cfg.task, "show_caption", False)) and self.show_text
        self.random = random.Random(int(_cfg_get(cfg, "seed", 1)))
        self.handles = []
        self.grasp_cache = OrderedDict()
        self.prior_cache = OrderedDict()
        self.object_mesh_cache = OrderedDict()
        self.pc_cache = OrderedDict()
        self.visualizers: Dict[str, Visualizer] = {}
        self.mano_layers = None

    def _cache_read(self, cache: OrderedDict, key):
        """Read and refresh one cache entry.

        Args:
            cache: OrderedDict cache.
            key: Cache key.

        Returns:
            Cached value or ``None``.
        """
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_write(self, cache: OrderedDict, key, value, max_size: int = 128) -> None:
        """Write one bounded cache entry.

        Args:
            cache: OrderedDict cache.
            key: Cache key.
            value: Value to store.
            max_size: Maximum retained entries.

        Returns:
            None.
        """
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _load_grasp(self, record: PriorPathRecord) -> Dict:
        """Load one grasp result with caching.

        Args:
            record: Indexed grasp record.

        Returns:
            Loaded grasp dictionary.
        """
        cached = self._cache_read(self.grasp_cache, record.grasp_file)
        if cached is not None:
            return cached
        data = _load_npy_dict(record.grasp_file)
        self._cache_write(self.grasp_cache, record.grasp_file, data)
        return data

    def _load_prior(self, prior_file: str) -> Dict:
        """Load one human-prior export with caching.

        Args:
            prior_file: Human-prior file path.

        Returns:
            Loaded prior dictionary.
        """
        cached = self._cache_read(self.prior_cache, prior_file)
        if cached is not None:
            return cached
        data = _load_npy_dict(prior_file)
        self._cache_write(self.prior_cache, prior_file, data)
        return data

    def _load_object_mesh(self, scene_path: str) -> tm.Trimesh:
        """Load one object mesh with caching.

        Args:
            scene_path: Scene config path.

        Returns:
            Transformed object mesh.
        """
        cached = self._cache_read(self.object_mesh_cache, scene_path)
        if cached is not None:
            return cached
        mesh = _load_object_mesh_from_scene(scene_path)
        self._cache_write(self.object_mesh_cache, scene_path, mesh)
        return mesh

    def _load_pc(self, prior_file: str, prior_data: Dict) -> np.ndarray:
        """Load one point cloud with caching.

        Args:
            prior_file: Human-prior file path used as cache key.
            prior_data: Loaded prior dictionary.

        Returns:
            Point cloud array.
        """
        key = (prior_file, self.num_points)
        cached = self._cache_read(self.pc_cache, key)
        if cached is not None:
            return cached
        pc = _load_scene_point_cloud(prior_data, self.num_points)
        self._cache_write(self.pc_cache, key, pc)
        return pc

    def _visualizer(self, robot_file: str) -> Visualizer:
        """Return a cached robot visualizer.

        Args:
            robot_file: Robot config file name.

        Returns:
            Visualizer instance.
        """
        if robot_file not in self.visualizers:
            self.visualizers[robot_file] = _make_visualizer(robot_file, self.device)
        return self.visualizers[robot_file]

    def _ensure_mano_layers(self):
        """Create MANO layers on first render.

        Args:
            None.

        Returns:
            Mapping from side name to MANO layer.
        """
        if self.mano_layers is None:
            mano_root_value = _cfg_get(self.cfg.task, "mano_root")
            if not mano_root_value:
                raise ValueError("task.mano_root must point to a locally installed MANO models directory")
            mano_root = _resolve_path(str(mano_root_value))
            self.mano_layers = _create_mano_layers(self.device, mano_root)
        return self.mano_layers

    def _sample_records(self) -> List[PriorPathRecord]:
        """Randomly select one batch across all indexed grasp types.

        Args:
            None.

        Returns:
            Distinct-scene records for the next batch.
        """
        shuffled = list(self.records)
        self.random.shuffle(shuffled)
        selected = []
        seen_scenes = set()
        for record in shuffled:
            if record.scene_id in seen_scenes:
                continue
            selected.append(record)
            seen_scenes.add(record.scene_id)
            if len(selected) >= self.batch_size:
                break
        return selected

    def _clear_scene(self) -> None:
        """Remove previous Viser nodes.

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

    def start(self) -> None:
        """Start the Viser UI and block.

        Args:
            None.

        Returns:
            None.
        """
        host = str(_cfg_get(self.cfg.task, "host", "0.0.0.0"))
        port = int(_cfg_get(self.cfg.task, "port", 8082))
        server_label = "HUGS-BODex Prior Path Visualizer" if self.show_text else " "
        server = viser.ViserServer(host=host, port=port, label=server_label)
        server.scene.world_axes.visible = bool(_cfg_get(self.cfg.task, "show_world_axes", False))
        server.scene.world_axes.axes_length = 0.04
        server.scene.world_axes.axes_radius = 0.001
        server.scene.world_axes.origin_radius = 0.004

        next_button_label = str(_cfg_get(self.cfg.task, "next_button_label", "Next Batch" if self.show_text else " "))
        next_button = server.gui.add_button(next_button_label)
        status = server.gui.add_markdown("") if self.show_caption else None

        def render() -> None:
            self._clear_scene()
            batch = self._sample_records()
            caption = self._render_batch(server, batch)
            if status is not None:
                status.content = caption

        @next_button.on_click
        def _(_event):
            render()

        render()
        browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        print(f"Viser web UI is running at http://{browser_host}:{port}")
        while True:
            time.sleep(1.0)

    def _render_batch(self, server, batch: Sequence[PriorPathRecord]) -> str:
        """Render one randomly selected batch.

        Args:
            server: Viser server.
            batch: Records to render.

        Returns:
            Markdown status text.
        """
        caption = ["### human prior -> robot init -> robot grasp pose", ""]
        for row, record in enumerate(batch):
            try:
                line = self._render_record(server, row, record)
            except Exception as exc:
                logging.exception("Failed to render %s", record.grasp_file)
                line = f"- `{record.scene_id}` `{record.type_name}`: failed: {exc}"
            caption.append(line)
        return "\n".join(caption)

    def _render_record(self, server, row: int, record: PriorPathRecord) -> str:
        """Render one scene row with three columns.

        Args:
            server: Viser server.
            row: Row index.
            record: Grasp result record to render.

        Returns:
            Markdown line describing the rendered row.
        """
        grasp_data = self._load_grasp(record)
        prior_file = _human_prior_file_from_grasp_data(grasp_data)
        prior_data = self._load_prior(prior_file)
        scene_path = _scene_path_from_grasp_data(grasp_data)
        obj_mesh = self._load_object_mesh(scene_path)
        pc = self._load_pc(prior_file, prior_data)

        robot_pose = torch.as_tensor(np.asarray(grasp_data["robot_pose"]).squeeze(0), dtype=torch.float32, device=self.device)
        sample_count, pose_count, n_dof = robot_pose.shape
        seed_index = self.seed_index if self.seed_index >= 0 else int(self.random.randrange(sample_count))
        seed_index = int(np.clip(seed_index, 0, sample_count - 1))
        final_pose_index = _pose_index_for_type(record.type_id, pose_count)

        sample_indices = np.asarray(grasp_data["human_prior_sample_indices"]).reshape(-1)
        prior_sample_index = int(sample_indices[seed_index])
        prior_type_index = _type_index_in_prior(prior_data, record.type_id)

        if "init_seed_config" in grasp_data:
            init_pose = torch.as_tensor(
                np.asarray(grasp_data["init_seed_config"]).squeeze(0),
                dtype=torch.float32,
                device=self.device,
            )[seed_index].reshape(1, -1)
        else:
            init_pose = robot_pose[seed_index, 0, :].reshape(1, n_dof)
        final_pose = robot_pose[seed_index, final_pose_index, :].reshape(1, n_dof)
        robot_qpos = _add_root_pose_if_needed(torch.cat([init_pose, final_pose], dim=0), record.robot_file)

        visualizer = self._visualizer(record.robot_file)
        visualizer.set_robot_parameters(robot_qpos, joint_names=grasp_data["joint_names"])
        robot_init_mesh = visualizer.get_robot_trimesh_data(0, color=[0.35, 0.55, 0.95, 0.72])
        robot_final_mesh = visualizer.get_robot_trimesh_data(1, color=[0.95, 0.48, 0.32, 0.78])

        row_offset = np.array([0.0, row * self.spacing_y, 0.0], dtype=np.float64)
        col_offsets = [
            row_offset + np.array([0.0, 0.0, 0.0], dtype=np.float64),
            row_offset + np.array([self.spacing_x, 0.0, 0.0], dtype=np.float64),
            row_offset + np.array([2.0 * self.spacing_x, 0.0, 0.0], dtype=np.float64),
        ]
        base_name = f"/prior_path/r{row}"

        self.handles.append(
            server.scene.add_point_cloud(
                f"{base_name}/prior/object_pc",
                points=pc + col_offsets[0],
                colors=np.tile(np.array([[255, 70, 70]], dtype=np.uint8), (pc.shape[0], 1)),
                point_size=0.004,
            )
        )
        for idx, mesh in enumerate(
            _make_mano_meshes(
                prior_data,
                prior_type_index,
                prior_sample_index,
                record.type_id,
                self._ensure_mano_layers(),
                self.device,
            )
        ):
            self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/prior/mano_{idx}", _offset_mesh(mesh, col_offsets[0])))

        self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/init/object", _offset_mesh(obj_mesh, col_offsets[1])))
        self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/init/robot", _offset_mesh(robot_init_mesh, col_offsets[1])))
        self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/grasp/object", _offset_mesh(obj_mesh, col_offsets[2])))
        self.handles.append(server.scene.add_mesh_trimesh(f"{base_name}/grasp/robot", _offset_mesh(robot_final_mesh, col_offsets[2])))

        labels = ["human prior", "robot init", f"robot grasp opt={1.0 if record.type_id in SINGLE_HAND_TYPE_IDS else 0.8:.1f}"]
        for col, label in enumerate(labels):
            if not self.show_text:
                continue
            self.handles.append(
                server.scene.add_label(
                    f"{base_name}/label_{col}",
                    label,
                    position=col_offsets[col] + np.array([0.0, 0.0, 0.26], dtype=np.float64),
                    font_size_mode="screen",
                    font_screen_scale=0.65,
                    anchor="bottom-center",
                )
            )
        if self.show_text:
            self.handles.append(
                server.scene.add_label(
                    f"{base_name}/row_label",
                    f"{row + 1}. {record.type_name}",
                    position=row_offset + np.array([-0.22, 0.0, 0.22], dtype=np.float64),
                    font_size_mode="screen",
                    font_screen_scale=0.65,
                    anchor="bottom-center",
                )
            )
        return (
            f"- `{record.scene_id}` | `{record.type_name}` | seed `{seed_index}` -> prior sample "
            f"`{prior_sample_index}` | final pose index `{final_pose_index}/{pose_count - 1}`"
        )


def _build_records(cfg) -> List[PriorPathRecord]:
    """Scan suite-derived experiment directories for grasp records.

    Args:
        cfg: Hydra config.

    Returns:
        Indexed records across all requested grasp types.
    """
    suite_config = _load_suite_config(cfg.task)
    requested_types = _normalize_grasp_types(_cfg_get(cfg.task, "grasp_types", "all"))
    exclude_both_three = bool(_cfg_get(cfg.task, "exclude_both_three", False))
    records: List[PriorPathRecord] = []
    for type_name, type_config in suite_config.get("grasp_types", {}).items():
        type_config = dict(type_config)
        if not _type_matches(str(type_name), type_config, requested_types):
            continue
        type_id = int(type_config.get("type_id", str(type_name).split("_", 1)[0]))
        if exclude_both_three and type_id == BOTH_THREE_TYPE_ID:
            continue

        manip_cfg_file = str(type_config["manip_cfg_file"])
        manip_config = load_yaml(join_path(get_manip_configs_path(), manip_cfg_file))
        robot_file = str(manip_config["robot_file"])
        graspdata_dir = _output_graspdata_dir(str(cfg.output_path), manip_cfg_file, str(cfg.name))
        file_paths = sorted(glob.glob(os.path.join(graspdata_dir, "**", "*_grasp.npy"), recursive=True))
        print(
            f"visualize_prior_path init: {type_name} found {len(file_paths)} grasp files in {graspdata_dir}",
            flush=True,
        )
        for grasp_file in file_paths:
            try:
                records.append(
                    PriorPathRecord(
                        grasp_file=grasp_file,
                        scene_id=_scene_id_from_grasp_path(grasp_file),
                        type_id=type_id,
                        type_name=str(type_name),
                        robot_file=robot_file,
                        manip_cfg_file=manip_cfg_file,
                        root_label=str(cfg.name),
                    )
                )
            except Exception as exc:
                logging.warning("Skip unreadable prior-path grasp file %s: %s", grasp_file, exc)
    if not records:
        raise FileNotFoundError(
            "No prior-path grasp records found. "
            f"name={cfg.name}, suite_config={_cfg_get(cfg.task, 'suite_config', 'sim_shadow.yml')}, "
            f"output_path={cfg.output_path}"
        )
    print(
        f"visualize_prior_path init: indexed {len(records)} records across "
        f"{len({record.scene_id for record in records})} scenes",
        flush=True,
    )
    return records


def task_visualize_prior_path(cfg):
    """Launch a Viser UI for human-prior to robot-grasp optimization paths.

    Args:
        cfg: Hydra config.

    Returns:
        None. The Viser server blocks until interrupted.
    """
    records = _build_records(cfg)
    app = PriorPathViserApp(cfg, records)
    app.start()

from dataclasses import dataclass
from glob import glob
import copy
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curobo.geom.basic_transform import euler_angles_to_matrix, matrix_to_quaternion, torch_quaternion_to_matrix
from curobo.util.world_cfg_generator import scenecfg2worldcfg
from curobo.util_file import (
    get_assets_path,
    get_manip_configs_path,
    get_robot_configs_path,
    join_path,
    load_json,
    load_scene_cfg,
    load_yaml,
)


def _quaternion_to_matrix_np(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a real-first quaternion into a 3x3 rotation matrix.

    Args:
        quaternion: Quaternion in ``wxyz`` order.

    Returns:
        Numpy rotation matrix with shape ``(3, 3)``.
    """
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )

HUMAN_PRIOR_GRASP_TYPE_IDS = (1, 2, 3, 4, 5)
HUMAN_PRIOR_GRASP_TYPE_NAMES = (
    "1_right_two",
    "2_right_three",
    "3_right_full",
    "4_both_three",
    "5_both_full",
)
BUDGET_ROUNDING_CEIL = "ceil"
BUDGET_ROUNDING_ROUND = "round"
BUDGET_ROUNDING_MODES = {BUDGET_ROUNDING_CEIL, BUDGET_ROUNDING_ROUND}
SUPPORTED_ROOT_POSE_HUMAN_PRIOR_ROBOTS = {
    "right_shadow_hand_sim.yml",
    "leap_sp.yml",
}
SUPPORTED_DUMMY_ARM_HUMAN_PRIOR_ROBOTS = {
    "dual_dummy_arm_shadow.yml",
    "dual_dummy_arm_leap_sp.yml",
}


def _deep_update_dict(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge override values into a dictionary.

    Args:
        base: Base dictionary mutated in place.
        overrides: Override dictionary whose non-dict values replace base values.

    Returns:
        The updated base dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _normalize_budget_bound(bound, type_count: int, default_value: Optional[int]) -> Optional[np.ndarray]:
    """Normalize a scalar or per-type budget bound into an integer array.

    Args:
        bound: Optional scalar, sequence, or mapping keyed by type id/name.
        type_count: Number of human-prior grasp types.
        default_value: Value used when ``bound`` is ``None``.

    Returns:
        Integer numpy array with shape ``(type_count,)``, or ``None`` when no upper
        bound is requested.
    """
    if bound is None:
        if default_value is None:
            return None
        return np.full((type_count,), int(default_value), dtype=np.int64)
    if isinstance(bound, dict):
        values = []
        for idx, type_name in enumerate(HUMAN_PRIOR_GRASP_TYPE_NAMES):
            type_id = HUMAN_PRIOR_GRASP_TYPE_IDS[idx]
            value = bound.get(type_name, bound.get(str(type_id), bound.get(type_id, default_value)))
            if value is None:
                raise ValueError(f"Missing budget bound for type {type_name}")
            values.append(int(value))
        return np.asarray(values, dtype=np.int64)
    if isinstance(bound, (list, tuple, np.ndarray)):
        arr = np.asarray(bound, dtype=np.int64)
        if arr.shape != (type_count,):
            raise ValueError(f"Budget bound must have shape ({type_count},), got {arr.shape}")
        return arr
    return np.full((type_count,), int(bound), dtype=np.int64)


def _allocate_bounded_extra(safe_scores: np.ndarray, min_budget: np.ndarray, max_budget: np.ndarray, total_budget: int):
    """Allocate a capped budget without redistributing overflow from clipped types.

    Args:
        safe_scores: Non-negative finite preference scores.
        min_budget: Per-type lower budget bounds.
        max_budget: Per-type upper budget bounds.
        total_budget: Total integer budget to distribute.

    Returns:
        Integer per-type budget array whose sum is at most ``total_budget``.
    """
    budget = min_budget.astype(np.int64).copy()
    remaining = int(total_budget) - int(budget.sum())
    if remaining <= 0:
        return np.minimum(budget, max_budget)

    weights = safe_scores
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
    raw_extra = weights / weights.sum() * remaining
    extra = np.floor(raw_extra).astype(np.int64)
    remainder = remaining - int(extra.sum())
    if remainder > 0:
        fraction = raw_extra - extra
        order = np.lexsort((np.arange(fraction.shape[0]), -fraction))
        extra[order[:remainder]] += 1

    # max_budget is a hard per-type cap. Any overflow above the cap is unused
    # instead of being assigned to other grasp types.
    return np.minimum(budget + extra, max_budget)


def allocate_type_budget(
    scores: Sequence[float],
    total_budget: int = 40,
    min_budget=0,
    max_budget=None,
    score_threshold: float = 0.0,
    budget_resolution: int = 1,
    budget_rounding_mode: str = BUDGET_ROUNDING_CEIL,
) -> np.ndarray:
    """Allocate an integer seed budget across human-prior grasp types.

    Args:
        scores: Per-type non-negative preference scores.
        total_budget: Total integer budget to distribute.
        min_budget: Optional scalar or per-type lower bound.
        max_budget: Optional scalar or per-type upper bound.
        score_threshold: Per-scene minimum score required for a grasp type to
            receive any budget.
        budget_resolution: Positive integer resolution for ``K_t``. Positive
            budgets are snapped to resolution multiples and capped by
            ``max_budget`` when a max bound is configured.
        budget_rounding_mode: Rounding mode for ``budget_resolution``. ``ceil``
            rounds positive budgets up; ``round`` uses half-up rounding to the
            nearest resolution multiple.

    Returns:
        Integer numpy array with the same length as ``scores``. When ``max_budget``
        is set, the sum can be smaller than ``total_budget`` because clipped
        overflow is intentionally left unused. With ``ceil`` and
        ``budget_resolution > 1``, the sum can also be larger than
        ``total_budget`` because each positive ``K_t`` is rounded up for coarser
        solver batching.
    """
    score_arr = np.asarray(scores, dtype=np.float64)
    if score_arr.ndim != 1:
        raise ValueError(f"scores must be a 1D array, got shape {score_arr.shape}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}")
    score_threshold = float(score_threshold)
    if score_threshold < 0.0:
        raise ValueError(f"score_threshold must be non-negative, got {score_threshold}")
    budget_resolution = int(budget_resolution)
    if budget_resolution <= 0:
        raise ValueError(f"budget_resolution must be positive, got {budget_resolution}")
    budget_rounding_mode = normalize_budget_rounding_mode(budget_rounding_mode)

    safe_scores = np.where(np.isfinite(score_arr), score_arr, 0.0)
    safe_scores = np.maximum(safe_scores, 0.0)
    active_mask = safe_scores >= score_threshold
    safe_scores = np.where(active_mask, safe_scores, 0.0)

    min_budget_arr = _normalize_budget_bound(min_budget, score_arr.shape[0], 0)
    max_budget_arr = _normalize_budget_bound(max_budget, score_arr.shape[0], None)
    # The score threshold is a hard scene/type gate, so it also suppresses any
    # configured minimum budget for inactive grasp types.
    min_budget_arr = np.where(active_mask, min_budget_arr, 0)
    if max_budget_arr is not None:
        max_budget_arr = np.where(active_mask, max_budget_arr, 0)
    if min_budget_arr is not None and (min_budget_arr < 0).any():
        raise ValueError(f"min_budget must be non-negative, got {min_budget_arr}")
    if max_budget_arr is not None:
        if (max_budget_arr < 0).any():
            raise ValueError(f"max_budget must be non-negative, got {max_budget_arr}")
        if (max_budget_arr < min_budget_arr).any():
            raise ValueError(f"max_budget must be >= min_budget, got min={min_budget_arr}, max={max_budget_arr}")
    if int(min_budget_arr.sum()) > int(total_budget):
        raise ValueError(
            f"sum(min_budget)={int(min_budget_arr.sum())} is larger than total_budget={int(total_budget)}"
        )
    if not active_mask.any():
        return min_budget_arr.astype(np.int64)
    if safe_scores.sum() <= 0.0:
        safe_scores = active_mask.astype(np.float64)
    if max_budget_arr is not None:
        budget = _allocate_bounded_extra(safe_scores, min_budget_arr, max_budget_arr, int(total_budget))
        return quantize_type_budget_resolution(budget, budget_resolution, max_budget_arr, budget_rounding_mode)

    raw_budget = safe_scores / safe_scores.sum() * int(total_budget)
    budget = np.floor(raw_budget).astype(np.int64)
    remainder = int(total_budget) - int(budget.sum())
    if remainder > 0:
        fraction = raw_budget - budget
        order = np.lexsort((np.arange(fraction.shape[0]), -fraction))
        budget[order[:remainder]] += 1
    if min_budget_arr is not None and min_budget_arr.any():
        budget = _allocate_bounded_extra(safe_scores, min_budget_arr, np.full_like(min_budget_arr, total_budget), int(total_budget))
    return quantize_type_budget_resolution(budget, budget_resolution, max_budget_arr, budget_rounding_mode)


def normalize_budget_rounding_mode(mode: str) -> str:
    """Normalize the human-prior budget rounding mode.

    Args:
        mode: User-provided rounding mode string.

    Returns:
        Canonical rounding mode, either ``ceil`` or ``round``.
    """
    mode = str(mode)
    if mode not in BUDGET_ROUNDING_MODES:
        raise ValueError(f"Unsupported budget_rounding_mode={mode}; choices={sorted(BUDGET_ROUNDING_MODES)}")
    return mode


def quantize_type_budget_resolution(
    budget: Sequence[int],
    budget_resolution: int = 1,
    max_budget: Optional[Sequence[int]] = None,
    rounding_mode: str = BUDGET_ROUNDING_CEIL,
) -> np.ndarray:
    """Snap per-type budgets to a coarser resolution.

    Args:
        budget: Integer per-type seed budgets.
        budget_resolution: Positive integer ``K_t`` resolution. A value of ``1``
            preserves the input budgets.
        max_budget: Optional per-type maximum budget used as a hard cap after
            resolution rounding.
        rounding_mode: ``ceil`` for upward rounding, or ``round`` for half-up
            nearest-multiple rounding.

    Returns:
        Integer numpy array with quantized budgets. Configured max bounds are
        never exceeded.
    """
    budget_arr = np.asarray(budget, dtype=np.int64)
    budget_resolution = int(budget_resolution)
    if budget_resolution <= 0:
        raise ValueError(f"budget_resolution must be positive, got {budget_resolution}")
    rounding_mode = normalize_budget_rounding_mode(rounding_mode)
    if budget_resolution == 1:
        quantized = budget_arr.copy()
    elif rounding_mode == BUDGET_ROUNDING_CEIL:
        quantized = np.where(
            budget_arr > 0,
            ((budget_arr + budget_resolution - 1) // budget_resolution) * budget_resolution,
            0,
        ).astype(np.int64)
    else:
        # Half-up integer rounding avoids Python's bankers rounding and gives
        # predictable behavior at exact half-resolution boundaries.
        quantized = ((budget_arr * 2 + budget_resolution) // (2 * budget_resolution)) * budget_resolution
    if max_budget is not None:
        max_budget_arr = np.asarray(max_budget, dtype=np.int64)
        if max_budget_arr.shape != budget_arr.shape:
            raise ValueError(f"max_budget must have shape {budget_arr.shape}, got {max_budget_arr.shape}")
        quantized = np.minimum(quantized, max_budget_arr)
    return quantized.astype(np.int64)


def normalize_manip_config_path(config_path: str) -> str:
    """Normalize a manipulation config path for runtime-config matching.

    Args:
        config_path: Config path such as ``sim_shadow/tabletop_three`` or
            ``sim_shadow/tabletop_three.yml``.

    Returns:
        Normalized config path with a ``.yml`` suffix and POSIX-style separators.
    """
    original_path = str(config_path)
    config_path = original_path
    if os.path.isabs(config_path):
        manip_root = os.path.normpath(get_manip_configs_path()).replace(os.sep, "/")
        normalized = os.path.normpath(config_path).replace(os.sep, "/")
        if normalized.startswith(manip_root + "/"):
            normalized = normalized[len(manip_root) + 1 :]
    else:
        normalized = os.path.normpath(config_path).replace(os.sep, "/")
    if not normalized.endswith(".yml"):
        normalized = f"{normalized}.yml"
    if not os.path.isabs(original_path):
        normalized = normalized.lstrip("./")
    return normalized


def resolve_runtime_config_path(runtime_config: str) -> str:
    """Resolve a runtime config path under the manipulation config directory.

    Args:
        runtime_config: Absolute path or path relative to ``configs/manip``.

    Returns:
        Absolute path to the runtime config file.
    """
    if os.path.isabs(runtime_config):
        return runtime_config
    return join_path(get_manip_configs_path(), runtime_config)


def load_runtime_config(runtime_config: str) -> Dict:
    """Load a runtime config YAML file.

    Args:
        runtime_config: Absolute path or path relative to ``configs/manip``.

    Returns:
        Runtime config dictionary.
    """
    return load_yaml(resolve_runtime_config_path(runtime_config))


def grasp_type_from_runtime_config(manip_cfg_file: str, runtime_config: Dict) -> Tuple[int, str, Dict]:
    """Find the current human-prior grasp type from a runtime config.

    Args:
        manip_cfg_file: Current manipulation config path passed to ``-c``.
        runtime_config: Loaded runtime config dictionary.

    Returns:
        Tuple of ``(type_id, type_name, type_config)``.
    """
    target_path = normalize_manip_config_path(manip_cfg_file)
    grasp_types = runtime_config.get("grasp_types", {})
    for type_name, type_config in grasp_types.items():
        if isinstance(type_config, str):
            candidate_path = normalize_manip_config_path(type_config)
            candidate_config = {"type_id": None, "manip_cfg_file": candidate_path}
        else:
            candidate_path = normalize_manip_config_path(type_config.get("manip_cfg_file", ""))
            candidate_config = dict(type_config)
        if candidate_path == target_path:
            type_id = candidate_config.get("type_id")
            if type_id is None:
                type_id = _type_id_from_type_name(type_name)
            return int(type_id), str(type_name), candidate_config
    raise ValueError(f"Could not match manip config {target_path} in runtime_config.grasp_types")


def collect_scene_paths(world_config: Dict, scene_source_config: Optional[Dict] = None) -> List[str]:
    """Collect a deterministic list of scene config files.

    Args:
        world_config: Manipulation config ``world`` section.
        scene_source_config: Optional runtime ``scene_source`` section.

    Returns:
        Scene config path list after optional deterministic shuffling and range slicing.
    """
    if world_config.get("type") != "scene_cfg":
        raise NotImplementedError(f"Only world.type=scene_cfg is supported, got {world_config.get('type')}")

    scene_source_config = scene_source_config or {}
    template_path = scene_source_config.get("template_path") or world_config["template_path"]
    scene_cfg_pattern = join_path(get_assets_path(), template_path)
    all_paths = sorted(glob(scene_cfg_pattern, recursive=True))

    if bool(scene_source_config.get("shuffle_before_slice", False)):
        shuffle_seed = int(scene_source_config.get("shuffle_seed", 123))
        rng = np.random.default_rng(shuffle_seed)
        all_paths = [all_paths[index] for index in rng.permutation(len(all_paths))]

    start = world_config.get("start")
    end = world_config.get("end")
    sliced_paths = all_paths[start:end]

    # start/end define a shared global scene slice. Per-type scale filters are
    # applied afterwards so each grasp type sees the same candidate records.
    use_object_scale_list = bool(scene_source_config.get("use_object_scale_list", False))
    object_scale_list = world_config.get("object_scale_list", []) if use_object_scale_list else None
    if use_object_scale_list:
        scale_patterns = [f"scale{math.floor(scale * 100 + 0.5):03d}_" for scale in object_scale_list]
        sliced_paths = [path for path in sliced_paths if any(pattern in path for pattern in scale_patterns)]
    return sliced_paths


def scene_id_from_scene_path(scene_path: str) -> str:
    """Read the scene id from a scene config file without random reordering.

    Args:
        scene_path: Path to a per-scene ``.npy`` config file.

    Returns:
        Scene id stored in the scene config.
    """
    scene_cfg = np.load(scene_path, allow_pickle=True).item()
    return str(scene_cfg["scene_id"])


def scene_path_to_world_info(scene_path: str) -> Dict:
    """Convert one scene config file into the fields consumed by ``GraspSolver``.

    Args:
        scene_path: Path to a per-scene ``.npy`` config file.

    Returns:
        Dictionary with world config, object metadata, and save prefix.
    """
    scene_cfg = load_scene_cfg(scene_path)
    scene_id = str(scene_cfg["scene_id"])
    obj_name = scene_cfg["task"]["obj_name"]
    obj_cfg = scene_cfg["scene"][obj_name]
    obj_scale = obj_cfg["scale"]
    obj_pose = obj_cfg["pose"]

    json_data = load_json(obj_cfg["info_path"])
    obj_rot = _quaternion_to_matrix_np(obj_pose[3:])
    gravity_center = obj_pose[:3] + obj_rot @ json_data["gravity_center"] * obj_scale
    obb_length = np.linalg.norm(obj_scale * json_data["obb"]) / 2

    return {
        "scene_path": scene_path,
        "world_cfg": scenecfg2worldcfg(scene_cfg),
        "manip_name": scene_id + obj_name,
        "obj_gravity_center": gravity_center,
        "obj_obb_length": obb_length,
        "save_prefix": f"{scene_id}_",
    }


def build_world_info_batch(scene_paths: Sequence[str]) -> Dict:
    """Build a batched world-info dictionary without using random DataLoader order.

    Args:
        scene_paths: Ordered scene config paths for one solver batch.

    Returns:
        Batched world-info dictionary compatible with ``SaveHelper``.
    """
    world_infos = [scene_path_to_world_info(scene_path) for scene_path in scene_paths]
    return {
        "scene_path": [info["scene_path"] for info in world_infos],
        "world_cfg": [info["world_cfg"] for info in world_infos],
        "manip_name": [info["manip_name"] for info in world_infos],
        "obj_gravity_center": torch.as_tensor(
            np.stack([info["obj_gravity_center"] for info in world_infos], axis=0),
            dtype=torch.float32,
        ),
        "obj_obb_length": torch.as_tensor(
            np.asarray([info["obj_obb_length"] for info in world_infos], dtype=np.float32),
            dtype=torch.float32,
        ),
        "save_prefix": [info["save_prefix"] for info in world_infos],
    }


def load_human_prior_record(human_prior_root: str, scene_id: str) -> Tuple[Dict, str]:
    """Load and validate one human-prior scene record.

    Args:
        human_prior_root: Root directory containing per-scene prior ``.npy`` files.
        scene_id: Scene id, for example ``object/tabletop_ur10e/scale006_pose000_0``.

    Returns:
        Tuple of loaded record dictionary and absolute prior file path.
    """
    scene_file = os.path.join(human_prior_root, f"{scene_id}.npy")
    if not os.path.isfile(scene_file):
        raise FileNotFoundError(f"Human prior file does not exist for scene_id={scene_id}: {scene_file}")
    record = np.load(scene_file, allow_pickle=True).item()
    if "scene_id" in record and str(record["scene_id"]) != scene_id:
        raise ValueError(f"Human prior file {scene_file} stores scene_id={record['scene_id']}, expected {scene_id}")
    validate_human_prior_record(record, scene_id)
    return record, scene_file


def validate_human_prior_record(record: Dict, scene_id: Optional[str] = None) -> None:
    """Validate shape and type-order assumptions for a human-prior record.

    Args:
        record: Per-scene prior dictionary loaded from ``np.load(...).item()``.
        scene_id: Optional scene id for clearer error messages.

    Returns:
        None. Raises an exception when validation fails.
    """
    label = scene_id or str(record.get("scene_id", "<unknown>"))
    required_keys = {"budget_scores", "index_mcp_pos", "wrist_quat", "active_hand_mask"}
    missing = sorted(required_keys.difference(record.keys()))
    if missing:
        raise KeyError(f"Human prior record {label} misses keys: {missing}")

    budget_scores = np.asarray(record["budget_scores"])
    index_mcp_pos = np.asarray(record["index_mcp_pos"])
    wrist_quat = np.asarray(record["wrist_quat"])
    active_hand_mask = np.asarray(record["active_hand_mask"])

    expected_type_num = len(HUMAN_PRIOR_GRASP_TYPE_IDS)
    if budget_scores.shape != (expected_type_num,):
        raise ValueError(f"Invalid budget_scores shape for {label}: {budget_scores.shape}")
    if index_mcp_pos.ndim != 4 or index_mcp_pos.shape[0] != expected_type_num or index_mcp_pos.shape[-1] != 3:
        raise ValueError(f"Invalid index_mcp_pos shape for {label}: {index_mcp_pos.shape}")
    if wrist_quat.shape[:3] != index_mcp_pos.shape[:3] or wrist_quat.shape[-1] != 4:
        raise ValueError(f"Invalid wrist_quat shape for {label}: {wrist_quat.shape}")
    if active_hand_mask.shape != index_mcp_pos.shape[:3]:
        raise ValueError(f"Invalid active_hand_mask shape for {label}: {active_hand_mask.shape}")
    if not np.isfinite(budget_scores).all() or not np.isfinite(index_mcp_pos).all() or not np.isfinite(wrist_quat).all():
        raise ValueError(f"Non-finite human-prior value found for {label}")

    if "grasp_type_ids" in record:
        type_ids = np.asarray(record["grasp_type_ids"]).astype(int).tolist()
        if type_ids != list(HUMAN_PRIOR_GRASP_TYPE_IDS):
            raise ValueError(f"Unexpected grasp_type_ids for {label}: {type_ids}")
    if "grasp_type_names" in record:
        type_names = np.asarray(record["grasp_type_names"]).astype(str).tolist()
        if type_names != list(HUMAN_PRIOR_GRASP_TYPE_NAMES):
            raise ValueError(f"Unexpected grasp_type_names for {label}: {type_names}")

    active_quat = wrist_quat[active_hand_mask.astype(bool)]
    if active_quat.size > 0:
        quat_norm = np.linalg.norm(active_quat, axis=-1)
        if np.max(np.abs(quat_norm - 1.0)) > 1e-3:
            raise ValueError(f"Quaternion norm validation failed for {label}")


def select_human_prior_sample_indices(
    sample_count: int,
    budget: int,
    allow_replacement: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select per-type proposal indices for a requested budget.

    Args:
        sample_count: Number of proposals available for the current grasp type.
        budget: Requested integer budget for the current grasp type.
        allow_replacement: Whether to sample extra proposals with replacement.
        rng: Optional numpy random generator used for replacement samples.

    Returns:
        Tuple ``(indices, replacement_mask)`` with length ``budget``.
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    perm = rng.permutation(sample_count) if rng is not None else np.random.permutation(sample_count)
    if budget <= sample_count:
        return perm[:budget].astype(np.int64), np.zeros((budget,), dtype=bool)
    if not allow_replacement:
        raise ValueError(f"budget={budget} exceeds sample_count={sample_count}, and replacement is disabled")

    rng = rng or np.random.default_rng()
    extra = rng.choice(sample_count, size=budget - sample_count, replace=True).astype(np.int64)
    indices = np.concatenate([perm.astype(np.int64), extra], axis=0)
    replacement_mask = np.concatenate(
        [np.zeros((sample_count,), dtype=bool), np.ones((budget - sample_count,), dtype=bool)], axis=0
    )
    return indices, replacement_mask


@dataclass
class HumanPriorSceneJob:
    scene_path: str
    scene_id: str
    prior_record: Dict
    prior_file: str
    budget_scores: np.ndarray
    type_budgets: np.ndarray
    type_budget: int
    sample_indices: np.ndarray
    replacement_mask: np.ndarray


class HumanPriorSeedBuilder:
    def __init__(
        self,
        manip_config_data: Dict,
        runtime_config: Dict,
        type_id: int,
        type_name: str,
        type_config: Optional[Dict] = None,
    ):
        """Create a seed builder for Shadow human-prior initialization.

        Args:
            manip_config_data: Loaded manipulation config for the current grasp type.
            runtime_config: Loaded runtime config containing ``human_prior`` settings.
            type_id: Human-prior grasp type id in ``1..5``.
            type_name: Human-prior grasp type name.
            type_config: Optional runtime config section for the current grasp type.

        Returns:
            None.
        """
        self.manip_config_data = manip_config_data
        self.runtime_config = runtime_config
        self.type_config = dict(type_config or {})
        self.type_id = int(type_id)
        self.type_name = str(type_name)
        self.type_index = self.type_id - 1
        self.human_prior_config = dict(runtime_config.get("human_prior", {}))
        self.allow_replacement = bool(self.human_prior_config.get("replacement", True))
        self.seed = int(self.human_prior_config.get("seed", 123))
        self.min_type_budget = self.human_prior_config.get("min_type_budget", 0)
        self.max_type_budget = self.human_prior_config.get("max_type_budget")
        self.score_threshold = float(self.human_prior_config.get("score_threshold", 0.0))
        self.budget_resolution = int(self.human_prior_config.get("budget_resolution", 1))
        if self.budget_resolution <= 0:
            raise ValueError(f"human_prior.budget_resolution must be positive, got {self.budget_resolution}")
        self.budget_rounding_mode = normalize_budget_rounding_mode(
            self.human_prior_config.get("budget_rounding_mode", BUDGET_ROUNDING_CEIL)
        )
        self.jitter_config = self._resolve_jitter_config()
        self.transfer_file = str(
            self.type_config.get(
                "transfer_file",
                self.human_prior_config.get("transfer_file", "hand_pose_human_transfer/right_shadow_hand.yml"),
            )
        )
        self.transfer_path = join_path(get_robot_configs_path(), self.transfer_file)
        self.transfer_links = self._normalize_list(
            self.type_config.get(
                "transfer_links",
                self.type_config.get("transfer_link", self.human_prior_config.get("transfer_link", "rh_palm")),
            )
        )
        self.hand_indices = [
            int(index)
            for index in self._normalize_list(
                self.type_config.get(
                    "hand_indices",
                    self.type_config.get(
                        "hand_index",
                        self.human_prior_config.get("right_hand_index", 0),
                    ),
                )
            )
        ]
        if len(self.transfer_links) != len(self.hand_indices):
            raise ValueError(
                f"transfer_links and hand_indices must have the same length, got "
                f"{self.transfer_links} and {self.hand_indices}"
            )
        self.transfer_r, self.transfer_t = self._load_transfer_poses(self.transfer_path, self.transfer_links)
        self.hand_q = np.asarray(manip_config_data["seeder_cfg"]["q"], dtype=np.float32)
        self.robot_layout = None
        self._validate_supported_robot()

    def _resolve_jitter_config(self) -> Dict:
        """Merge global enable and grasp-type-level human-prior jitter ranges.

        Args:
            None.

        Returns:
            Jitter configuration dictionary. The global ``human_prior.jitter.enabled``
            switch is preserved, while ``grasp_types.<type>.jitter`` can override
            range/std parameters but cannot override the global enable switch.
        """
        jitter_cfg = copy.deepcopy(self.human_prior_config.get("jitter", {}))
        type_jitter_cfg = self.type_config.get("jitter")
        if isinstance(type_jitter_cfg, dict):
            type_jitter_cfg = {key: value for key, value in type_jitter_cfg.items() if key != "enabled"}
            _deep_update_dict(jitter_cfg, type_jitter_cfg)
        return jitter_cfg

    def _validate_supported_robot(self) -> None:
        """Validate the first implementation scope.

        Args:
            None.

        Returns:
            None. Raises ``NotImplementedError`` for unsupported robot layouts.
        """
        robot_file = self.manip_config_data.get("robot_file")
        robot_config = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]["kinematics"]
        use_root_pose = bool(robot_config.get("use_root_pose", False))
        if robot_file in SUPPORTED_ROOT_POSE_HUMAN_PRIOR_ROBOTS and use_root_pose:
            if len(self.transfer_links) != 1:
                raise ValueError(
                    f"Root-pose human-prior initialization expects one transfer link, got {self.transfer_links}"
                )
            if robot_config.get("base_link") != self.transfer_links[0]:
                raise ValueError(
                    f"Transfer link {self.transfer_links[0]} does not match robot base_link={robot_config.get('base_link')}"
                )
            self.robot_layout = "root_pose"
            return

        if robot_file in SUPPORTED_DUMMY_ARM_HUMAN_PRIOR_ROBOTS and not use_root_pose:
            expected_links = ["rh_palm", "lh_palm"]
            if "leap_sp" in str(robot_file):
                expected_links = ["rh_palm_lower", "lh_palm_lower"]
            if self.transfer_links != expected_links:
                raise ValueError(
                    f"Dual dummy-arm human-prior initialization expects transfer_links={expected_links}, "
                    f"got {self.transfer_links}"
                )
            if self.hand_indices != [0, 1]:
                raise ValueError(
                    f"Dual dummy-arm human-prior initialization expects hand_indices=[0, 1], "
                    f"got {self.hand_indices}"
                )
            self.robot_layout = "dummy_arm"
            return

        raise NotImplementedError(
            "Human-prior seed initialization currently supports "
            f"root-pose robots {sorted(SUPPORTED_ROOT_POSE_HUMAN_PRIOR_ROBOTS)} "
            f"and dummy-arm robots {sorted(SUPPORTED_DUMMY_ARM_HUMAN_PRIOR_ROBOTS)}. "
            f"Got robot_file={robot_file}, use_root_pose={use_root_pose}."
        )

    def _normalize_list(self, value) -> List:
        """Normalize a scalar or sequence config value into a Python list.

        Args:
            value: Scalar or sequence value from runtime config.

        Returns:
            List representation of the value.
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        return [value]

    def _load_transfer_poses(self, transfer_path: str, transfer_links: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load human-index-MCP to robot-base transfer poses for all active hands.

        Args:
            transfer_path: YAML path under ``configs/robot``.
            transfer_links: Link entries to read from the YAML file.

        Returns:
            Tuple ``(R, t)`` with shapes ``[hand, 3, 3]`` and ``[hand, 3]``.
        """
        transfer_r = []
        transfer_t = []
        for transfer_link in transfer_links:
            link_r, link_t = self._load_transfer_pose(transfer_path, transfer_link)
            transfer_r.append(link_r)
            transfer_t.append(link_t)
        return np.stack(transfer_r, axis=0), np.stack(transfer_t, axis=0)

    def _load_transfer_pose(self, transfer_path: str, transfer_link: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load one human-index-MCP to robot-base transfer pose.

        Args:
            transfer_path: YAML path under ``configs/robot``.
            transfer_link: Link entry to read from the YAML file.

        Returns:
            Tuple ``(R, t)`` for the transfer pose.
        """
        if not os.path.isfile(transfer_path):
            raise FileNotFoundError(f"Human transfer config does not exist: {transfer_path}")
        transfer_data = load_yaml(transfer_path)
        if transfer_link not in transfer_data:
            raise KeyError(f"Human transfer config {transfer_path} has no link entry {transfer_link}")
        entry = transfer_data[transfer_link]
        transfer_r = np.asarray(entry["r"], dtype=np.float32)
        transfer_t = np.asarray(entry["t"], dtype=np.float32)
        if transfer_r.shape != (3, 3) or transfer_t.shape != (3,):
            raise ValueError(f"Invalid transfer pose shape in {transfer_path}:{transfer_link}")
        return transfer_r, transfer_t

    def make_scene_job(
        self,
        scene_path: str,
        human_prior_root: str,
        total_budget: int,
    ) -> HumanPriorSceneJob:
        """Load one scene's human prior and compute its current-type budget.

        Args:
            scene_path: Scene config path.
            human_prior_root: Root directory containing human-prior scene files.
            total_budget: Total budget distributed across all human-prior types.

        Returns:
            ``HumanPriorSceneJob`` for later grouping by ``type_budget``.
        """
        scene_id = scene_id_from_scene_path(scene_path)
        record, prior_file = load_human_prior_record(human_prior_root, scene_id)
        budget_scores = np.asarray(record["budget_scores"], dtype=np.float32)
        type_budgets = allocate_type_budget(
            budget_scores,
            total_budget=total_budget,
            min_budget=self.min_type_budget,
            max_budget=self.max_type_budget,
            score_threshold=self.score_threshold,
            budget_resolution=self.budget_resolution,
            budget_rounding_mode=self.budget_rounding_mode,
        )
        type_budget = int(type_budgets[self.type_index])
        sample_count = int(np.asarray(record["index_mcp_pos"]).shape[1])
        sample_indices, replacement_mask = self.select_sample_indices(scene_id, sample_count, type_budget)
        return HumanPriorSceneJob(
            scene_path=scene_path,
            scene_id=scene_id,
            prior_record=record,
            prior_file=prior_file,
            budget_scores=budget_scores,
            type_budgets=type_budgets,
            type_budget=type_budget,
            sample_indices=sample_indices,
            replacement_mask=replacement_mask,
        )

    def select_sample_indices(self, scene_id: str, sample_count: int, type_budget: int) -> Tuple[np.ndarray, np.ndarray]:
        """Select deterministic human-prior proposal indices for this grasp type.

        Args:
            scene_id: Scene id used to derive the deterministic random seed.
            sample_count: Number of proposals available for the grasp type.
            type_budget: Requested proposal count for this grasp type.

        Returns:
            Tuple of selected proposal indices and replacement mask.
        """
        rng = np.random.default_rng(self.seed + _stable_scene_int(scene_id) + self.type_id)
        return select_human_prior_sample_indices(
            sample_count=sample_count,
            budget=type_budget,
            allow_replacement=self.allow_replacement,
            rng=rng,
        )

    def build_seed_tensor(
        self,
        jobs: Sequence[HumanPriorSceneJob],
        device: torch.device,
        dtype: torch.dtype,
        expected_dof: Optional[int] = None,
        seed_generator=None,
        init_source: str = "human",
    ) -> Tuple[torch.Tensor, Dict]:
        """Build a batched seed tensor and metadata for solver input.

        Args:
            jobs: Same-budget human-prior scene jobs.
            device: Torch device for the seed tensor.
            dtype: Torch dtype for the seed tensor.
            expected_dof: Optional solver DOF used for validation.
            seed_generator: Optional seed generator used to project dummy-arm
                transfer-link poses into full robot q.
            init_source: Human-prior initialization mode (``human``).

        Returns:
            Tuple ``(seed_tensor, metadata)``. ``seed_tensor`` has shape
            ``[batch, K_t, dof]``.
        """
        if not jobs:
            raise ValueError("jobs must not be empty")
        type_budget = int(jobs[0].type_budget)
        if type_budget <= 0:
            raise ValueError("build_seed_tensor requires type_budget > 0")
        if any(int(job.type_budget) != type_budget for job in jobs):
            raise ValueError("All jobs in one seed tensor must share the same type_budget")
        if init_source != "human":
            raise ValueError(f"Unsupported human-prior init_source={init_source}")

        base_trans_list = []
        base_rot_list = []
        for job in jobs:
            index_mcp_pos = np.asarray(job.prior_record["index_mcp_pos"], dtype=np.float32)
            wrist_quat = np.asarray(job.prior_record["wrist_quat"], dtype=np.float32)
            active_mask = np.asarray(job.prior_record["active_hand_mask"], dtype=bool)
            per_hand_trans = []
            per_hand_rot = []
            for local_hand_index, human_hand_index in enumerate(self.hand_indices):
                if not active_mask[self.type_index, job.sample_indices, human_hand_index].all():
                    raise ValueError(
                        f"Inactive hand sample found for scene_id={job.scene_id}, type={self.type_name}, "
                        f"hand_index={human_hand_index}"
                    )

                p_human = torch.as_tensor(
                    index_mcp_pos[self.type_index, job.sample_indices, human_hand_index],
                    device=device,
                    dtype=dtype,
                )
                q_human = torch.as_tensor(
                    wrist_quat[self.type_index, job.sample_indices, human_hand_index],
                    device=device,
                    dtype=dtype,
                )
                q_human = q_human / torch.linalg.norm(q_human, dim=-1, keepdim=True).clamp_min(1e-8)
                r_human = torch_quaternion_to_matrix(q_human)

                p_human, r_human = self._apply_optional_jitter(
                    p_human,
                    r_human,
                    scene_id=job.scene_id,
                    local_hand_index=local_hand_index,
                    human_hand_index=human_hand_index,
                )

                transfer_r = torch.as_tensor(self.transfer_r[local_hand_index], device=device, dtype=dtype)
                transfer_t = torch.as_tensor(self.transfer_t[local_hand_index], device=device, dtype=dtype)
                r_base = r_human @ transfer_r.transpose(-1, -2)
                p_base = p_human - (r_base @ transfer_t.view(3, 1)).squeeze(-1)
                per_hand_trans.append(p_base)
                per_hand_rot.append(r_base)

            base_trans_list.append(torch.stack(per_hand_trans, dim=1))
            base_rot_list.append(torch.stack(per_hand_rot, dim=1))

        base_trans = torch.stack(base_trans_list, dim=0)
        base_rot = torch.stack(base_rot_list, dim=0)
        hand_q = torch.as_tensor(self.hand_q, device=device, dtype=dtype).view(1, 1, -1)
        hand_q = hand_q.expand(len(jobs), type_budget, -1)
        if self.robot_layout == "root_pose":
            base_quat = matrix_to_quaternion(base_rot[:, :, 0])
            seed_tensor = torch.cat([base_trans[:, :, 0], base_quat, hand_q], dim=-1).contiguous()
        elif self.robot_layout == "dummy_arm":
            if seed_generator is None:
                raise ValueError("seed_generator is required for dummy-arm human-prior seed projection")
            seed_tensor = seed_generator.get_samples_from_base_pose(
                base_trans,
                base_rot,
                base_q=hand_q,
                apply_transfer=False,
            ).contiguous()
        else:
            raise NotImplementedError(f"Unsupported robot_layout={self.robot_layout}")
        if expected_dof is not None and seed_tensor.shape[-1] != expected_dof:
            raise ValueError(f"Seed dof mismatch: got {seed_tensor.shape[-1]}, expected {expected_dof}")

        metadata = {
            "human_prior_scene_file": [job.prior_file for job in jobs],
            "human_prior_budget_scores": np.stack([job.budget_scores for job in jobs], axis=0),
            "human_prior_type_budgets": np.stack([job.type_budgets for job in jobs], axis=0),
            "human_prior_type_budget": np.asarray([job.type_budget for job in jobs], dtype=np.int64),
            "human_prior_type_id": np.asarray([self.type_id for _ in jobs], dtype=np.int64),
            "human_prior_type_name": [self.type_name for _ in jobs],
            "human_prior_sample_indices": np.stack([job.sample_indices for job in jobs], axis=0),
            "human_prior_replacement_mask": np.stack([job.replacement_mask for job in jobs], axis=0),
            "human_prior_transfer_file": [self.transfer_path for _ in jobs],
            "init_seed_config": seed_tensor.detach().cpu().numpy(),
        }
        return seed_tensor, metadata

    def _apply_optional_jitter(
        self,
        index_mcp_trans: torch.Tensor,
        index_mcp_rot: torch.Tensor,
        scene_id: str,
        local_hand_index: int,
        human_hand_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply optional pose jitter in the human index MCP frame.

        Args:
            index_mcp_trans: Human index MCP translation tensor with shape ``[K, 3]``.
            index_mcp_rot: Human index MCP rotation tensor with shape ``[K, 3, 3]``.
            scene_id: Scene id used to derive deterministic per-scene randomness.
            local_hand_index: Hand index within the current robot transfer list.
            human_hand_index: Hand index in the human-prior record.

        Returns:
            Tuple of jittered index MCP translation and rotation tensors.
        """
        jitter_cfg = dict(self.jitter_config)
        if not bool(jitter_cfg.get("enabled", False)):
            return index_mcp_trans, index_mcp_rot

        generator = torch.Generator(device=index_mcp_trans.device)
        generator.manual_seed(self._jitter_seed(jitter_cfg, scene_id, local_hand_index, human_hand_index))
        trans_bounds = self._jitter_bounds_tensor(
            jitter_cfg,
            primary_key="translational_bounds_xyz",
            legacy_key="translational_bounds",
            device=index_mcp_trans.device,
            dtype=index_mcp_trans.dtype,
        )
        rot_bounds = self._jitter_bounds_tensor(
            jitter_cfg,
            primary_key="rotational_bounds_deg_xyz",
            legacy_key="rotational_bounds_deg",
            device=index_mcp_trans.device,
            dtype=index_mcp_trans.dtype,
            scale=math.pi / 180.0,
        )
        if trans_bounds is not None:
            local_noise = self._sample_uniform_jitter(index_mcp_trans.shape, trans_bounds, generator)
            index_mcp_trans = index_mcp_trans + (index_mcp_rot @ local_noise.unsqueeze(-1)).squeeze(-1)
        else:
            trans_std = self._jitter_std_tensor(
                jitter_cfg,
                primary_key="translational_std_xyz",
                legacy_key="translational_std",
                device=index_mcp_trans.device,
                dtype=index_mcp_trans.dtype,
            )
            if bool(torch.any(trans_std > 0.0)):
                local_noise = torch.randn(
                    index_mcp_trans.shape,
                    generator=generator,
                    device=index_mcp_trans.device,
                    dtype=index_mcp_trans.dtype,
                ) * trans_std.view(1, 3)
                index_mcp_trans = index_mcp_trans + (index_mcp_rot @ local_noise.unsqueeze(-1)).squeeze(-1)
        if rot_bounds is not None:
            euler_noise = self._sample_uniform_jitter(index_mcp_trans.shape, rot_bounds, generator)
            index_mcp_rot = index_mcp_rot @ euler_angles_to_matrix(euler_noise, "XYZ")
        else:
            rot_std = self._jitter_std_tensor(
                jitter_cfg,
                primary_key="rotational_std_deg_xyz",
                legacy_key="rotational_std_deg",
                device=index_mcp_trans.device,
                dtype=index_mcp_trans.dtype,
                scale=math.pi / 180.0,
            )
            if bool(torch.any(rot_std > 0.0)):
                euler_noise = torch.randn(
                    index_mcp_trans.shape,
                    generator=generator,
                    device=index_mcp_trans.device,
                    dtype=index_mcp_trans.dtype,
                ) * rot_std.view(1, 3)
                index_mcp_rot = index_mcp_rot @ euler_angles_to_matrix(euler_noise, "XYZ")
        return index_mcp_trans, index_mcp_rot

    def _sample_uniform_jitter(
        self,
        sample_shape: Sequence[int],
        bounds: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample local jitter from per-axis lower and upper bounds.

        Args:
            sample_shape: Output tensor shape, normally ``[K, 3]``.
            bounds: Tensor with shape ``[2, 3]`` storing lower and upper bounds.
            generator: Torch random generator for deterministic sampling.

        Returns:
            Jitter tensor sampled uniformly within ``bounds``.
        """
        rand = torch.rand(
            sample_shape,
            generator=generator,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        low = bounds[0].view(1, 3)
        high = bounds[1].view(1, 3)
        return low + rand * (high - low)

    def _jitter_bounds_tensor(
        self,
        jitter_cfg: Dict,
        primary_key: str,
        legacy_key: str,
        device: torch.device,
        dtype: torch.dtype,
        scale: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Normalize optional lower and upper jitter bounds into a tensor.

        Args:
            jitter_cfg: Human-prior jitter config dictionary.
            primary_key: Preferred config key for per-axis lower and upper bounds.
            legacy_key: Backward-compatible bounds config key.
            device: Torch device for the returned tensor.
            dtype: Torch dtype for the returned tensor.
            scale: Multiplicative scale applied to the configured values.

        Returns:
            Tensor with shape ``[2, 3]`` or ``None`` when no bounds are configured.
        """
        if primary_key not in jitter_cfg and legacy_key not in jitter_cfg:
            return None
        value = jitter_cfg.get(primary_key, jitter_cfg.get(legacy_key))
        if value is None:
            return None
        bounds = torch.as_tensor(value, device=device, dtype=dtype)
        if bounds.shape == (2,):
            bounds = bounds.view(2, 1).expand(2, 3)
        if bounds.shape != (2, 3):
            raise ValueError(f"{primary_key} must have shape [2, 3] or [2], got {value}")
        if bool(torch.any(bounds[1] < bounds[0])):
            raise ValueError(f"{primary_key} upper bounds must be >= lower bounds, got {value}")
        return bounds * float(scale)

    def _jitter_seed(self, jitter_cfg: Dict, scene_id: str, local_hand_index: int, human_hand_index: int) -> int:
        """Derive a deterministic jitter seed for one scene/type/hand stream.

        Args:
            jitter_cfg: Human-prior jitter config dictionary.
            scene_id: Scene id used to decorrelate batch entries.
            local_hand_index: Hand index within the current robot transfer list.
            human_hand_index: Hand index in the human-prior record.

        Returns:
            Deterministic integer seed for a torch random generator.
        """
        base_seed = int(jitter_cfg.get("seed", self.seed))
        seed = base_seed
        seed += _stable_scene_int(scene_id)
        seed += self.type_id * 1_000_003
        seed += int(local_hand_index) * 10_007
        seed += int(human_hand_index) * 1_009
        return int(seed % (2**63 - 1))

    def _jitter_std_tensor(
        self,
        jitter_cfg: Dict,
        primary_key: str,
        legacy_key: str,
        device: torch.device,
        dtype: torch.dtype,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Normalize scalar or XYZ jitter std config into a tensor.

        Args:
            jitter_cfg: Human-prior jitter config dictionary.
            primary_key: Preferred config key for per-axis std values.
            legacy_key: Backward-compatible scalar or per-axis config key.
            device: Torch device for the returned tensor.
            dtype: Torch dtype for the returned tensor.
            scale: Multiplicative scale applied to the configured values.

        Returns:
            Tensor with shape ``[3]`` containing x/y/z standard deviations.
        """
        value = jitter_cfg.get(primary_key, jitter_cfg.get(legacy_key, 0.0))
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 3:
                raise ValueError(f"{primary_key} must have length 3, got {value}")
            std = torch.as_tensor(value, device=device, dtype=dtype)
        else:
            std = torch.full((3,), float(value), device=device, dtype=dtype)
        if bool(torch.any(std < 0.0)):
            raise ValueError(f"{primary_key} must be non-negative, got {value}")
        return std * float(scale)


def chunk_sequence(items: Sequence, chunk_size: int) -> Iterable[Sequence]:
    """Yield fixed-size chunks from a sequence.

    Args:
        items: Sequence to chunk.
        chunk_size: Positive chunk size.

    Returns:
        Iterable of sequence slices.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _type_id_from_type_name(type_name: str) -> int:
    """Parse the numeric human-prior type id from a type name.

    Args:
        type_name: Type name prefixed by an integer id.

    Returns:
        Parsed integer type id.
    """
    return int(str(type_name).split("_", 1)[0])


def _stable_scene_int(scene_id: str) -> int:
    """Build a deterministic small integer hash for scene-local random choices.

    Args:
        scene_id: Scene id string.

    Returns:
        Stable non-negative integer.
    """
    value = 0
    for char in scene_id:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value

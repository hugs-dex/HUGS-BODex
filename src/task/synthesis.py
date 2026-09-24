from __future__ import annotations

import copy
import datetime
import json
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from glob import iglob
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Keep torch.compile disabled by default for this Hydra entrypoint unless the caller
# explicitly sets a different value in the environment.
os.environ.setdefault("CUROBO_TORCH_COMPILE_DISABLE", "1")

# Raise the cuRobo logger level before importing cuRobo modules so import-time
# informational messages do not get emitted by Hydra's default logging setup.
logging.getLogger("curobo").setLevel(logging.WARN)


class _TrimeshEvenSampleWarningFilter(logging.Filter):
    """Suppress the noisy trimesh even-sampling shortfall warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a logging record should be emitted.

        Args:
            record: Logging record produced by Python's logging framework.

        Returns:
            ``False`` only for the known ``sample_surface_even`` shortfall warning.
        """
        message = record.getMessage()
        return not (
            record.name == "trimesh.util"
            and message.startswith("only got ")
            and message.endswith(" samples!")
        )


class _SynthesisProgressNoiseLogFilter(logging.Filter):
    """Suppress solver/save messages that corrupt active tqdm progress bars."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a cuRobo logging record should be emitted.

        Args:
            record: Logging record produced by Python's logging framework.

        Returns:
            ``False`` only for known high-frequency synthesis progress noise.
        """
        message = record.getMessage()
        noisy_prefixes = (
            "final succ ",
            "final dist error ",
            "Save results to ",
            "save visualization to ",
        )
        return not (record.name == "curobo" and message.startswith(noisy_prefixes))


def _suppress_trimesh_even_sample_warning() -> None:
    """Install a narrow filter for trimesh even-sampling shortfall warnings.

    Args:
        None.

    Returns:
        None.
    """
    logger = logging.getLogger("trimesh.util")
    if not any(isinstance(log_filter, _TrimeshEvenSampleWarningFilter) for log_filter in logger.filters):
        logger.addFilter(_TrimeshEvenSampleWarningFilter())


def _install_synthesis_progress_noise_filter() -> Optional[logging.Filter]:
    """Install a temporary cuRobo log filter for active synthesis progress bars.

    Args:
        None.

    Returns:
        Installed filter instance, or ``None`` when an equivalent filter already exists.
    """
    logger = logging.getLogger("curobo")
    if any(isinstance(log_filter, _SynthesisProgressNoiseLogFilter) for log_filter in logger.filters):
        return None
    log_filter = _SynthesisProgressNoiseLogFilter()
    logger.addFilter(log_filter)
    return log_filter


def _remove_synthesis_progress_noise_filter(log_filter: Optional[logging.Filter]) -> None:
    """Remove a synthesis progress log filter installed by this runtime.

    Args:
        log_filter: Filter instance returned by ``_install_synthesis_progress_noise_filter``.

    Returns:
        None.
    """
    if log_filter is None:
        return
    logging.getLogger("curobo").removeFilter(log_filter)


def _suppress_warp_deprecation_warnings() -> None:
    """Install narrow filters for known noisy Warp deprecation warnings.

    Args:
        None.

    Returns:
        None.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"The namespace `warp\.torch` will soon be removed from the public API\..*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The symbol `warp\.torch\.device_from_torch` will soon be removed from the public API\..*",
        category=Warning,
    )


_suppress_warp_deprecation_warnings()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from curobo.util.human_prior_seed import (
    HumanPriorSeedBuilder,
    allocate_type_budget,
    build_world_info_batch,
    chunk_sequence,
    grasp_type_from_runtime_config as grasp_type_from_suite_config,
    load_human_prior_record,
    load_runtime_config as load_suite_config_file,
    scene_id_from_scene_path,
)
from curobo.util.logger import log_warn, setup_logger
from curobo.util_file import (
    get_assets_path,
    get_configs_path,
    get_manip_configs_path,
    get_output_path,
    join_path,
    load_scene_cfg,
    load_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

INIT_SOURCE_CHOICES = {"surface_sample", "human"}

BATCH_GROUPING_NONE = "none"
BATCH_GROUPING_OBJECT_SCALE = "object_scale"
BATCH_GROUPING_CHOICES = {BATCH_GROUPING_NONE, BATCH_GROUPING_OBJECT_SCALE}

SOLVER_REUSE_TYPE = "type"
SOLVER_REUSE_BATCH = "batch"
SOLVER_REUSE_CHOICES = {SOLVER_REUSE_TYPE, SOLVER_REUSE_BATCH}

DEFAULT_OBJECT_HEIGHT_RECORD_FILE = "tabletop_scene_object_heights.jsonl"

NPY_SAVE_KEYS = [
    "robot_pose",
    "world_cfg",
    "manip_name",
    "scene_path",
    "contact_point",
    "contact_frame",
    "contact_force",
    "grasp_error",
    "dist_error",
    "pene_error",
    "debug_info",
    "init_source",
    "human_prior_scene_file",
    "human_prior_budget_scores",
    "human_prior_type_budgets",
    "human_prior_type_budget",
    "human_prior_type_id",
    "human_prior_type_name",
    "human_prior_sample_indices",
    "human_prior_replacement_mask",
    "human_prior_transfer_file",
    "human_prior_solver_num_seeds",
    "init_seed_config",
]


class ProgressReporter:
    """Small progress wrapper used by long synthesis runs.

    Args:
        total: Total number of work units.
        desc: Human-readable progress label.
        unit: Unit name displayed by tqdm.
        enabled: Whether progress output should be emitted.
        leave: Whether the progress bar remains visible after closing.

    Returns:
        None.
    """

    def __init__(self, total: int, desc: str, unit: str = "it", enabled: bool = True, leave: bool = True):
        self.total = int(total)
        self.desc = str(desc)
        self.unit = str(unit)
        self.enabled = bool(enabled)
        self.count = 0
        self._fallback_step = max(1, self.total // 20) if self.total > 0 else 1
        self._bar = None
        if self.enabled and tqdm is not None:
            self._bar = tqdm(total=self.total, desc=self.desc, unit=self.unit, dynamic_ncols=True, leave=leave)
        elif self.enabled:
            log_warn(f"{self.desc}: 0/{self.total} {self.unit}")

    def set_postfix(self, **kwargs) -> None:
        """Update progress status without advancing the counter.

        Args:
            **kwargs: Key-value status fields displayed after the progress bar.

        Returns:
            None.
        """
        if not self.enabled or not kwargs:
            return
        if self._bar is not None:
            self._bar.set_postfix(kwargs, refresh=True)

    def update(self, n: int = 1, **kwargs) -> None:
        """Advance progress and optionally update status fields.

        Args:
            n: Number of completed work units.
            **kwargs: Optional key-value status fields displayed after the progress bar.

        Returns:
            None.
        """
        step = int(n)
        self.count += step
        if not self.enabled:
            return
        if self._bar is not None:
            if kwargs:
                self._bar.set_postfix(kwargs, refresh=False)
            self._bar.update(step)
            return
        if self.count == self.total or self.count % self._fallback_step == 0:
            suffix = " ".join(f"{key}={value}" for key, value in kwargs.items())
            log_warn(f"{self.desc}: {self.count}/{self.total} {self.unit} {suffix}".rstrip())

    def close(self) -> None:
        """Close the underlying progress display.

        Args:
            None.

        Returns:
            None.
        """
        if self._bar is not None:
            self._bar.close()


class SynthesisProfiler:
    """Accumulate coarse runtime timing for synthesis benchmark runs."""

    def __init__(self, enabled: bool = False):
        """Create a profiler accumulator.

        Args:
            enabled: Whether timing should be collected.

        Returns:
            None.
        """
        self.enabled = bool(enabled)
        self.total_seconds = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Measure wall-clock time for one synthesis stage.

        Args:
            stage: Stable stage name used in the final timing report.

        Returns:
            Context manager that records elapsed seconds when profiling is enabled.
        """
        if not self.enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.total_seconds[str(stage)] += time.perf_counter() - start_time
            self.counts[str(stage)] += 1

    def report(self, worker_label: str) -> None:
        """Print a compact timing summary for completed synthesis work.

        Args:
            worker_label: Label of the current process, such as ``main`` or ``gpu2``.

        Returns:
            None.
        """
        if not self.enabled or not self.total_seconds:
            return
        total = sum(self.total_seconds.values())
        ordered_items = sorted(self.total_seconds.items(), key=lambda item: -item[1])
        stage_text = []
        for stage, seconds in ordered_items:
            count = max(1, int(self.counts[stage]))
            stage_text.append(f"{stage}={seconds:.3f}s/{count}x avg={seconds / count:.3f}s")
        log_warn(f"Synthesis profile worker={worker_label} measured_total={total:.3f}s; " + "; ".join(stage_text))


def batch_grasp_count(batch: Dict) -> int:
    """Compute the progress unit count for one synthesis batch.

    Args:
        batch: Planned synthesis batch record.

    Returns:
        Number of requested grasps in this batch, equal to ``scene_count * K_t``.
    """
    if "grasp_count" in batch:
        return int(batch["grasp_count"])
    return int(batch["scene_count"]) * int(max(1, batch["type_budget"]))


def canonical_init_source(init_source: str) -> str:
    """Validate the requested initialization source.

    Args:
        init_source: User-provided init source.

    Returns:
        Valid init source: ``surface_sample`` or ``human``.
    """
    if init_source not in INIT_SOURCE_CHOICES:
        raise ValueError(f"Unsupported init_source={init_source}; choices={sorted(INIT_SOURCE_CHOICES)}")
    return init_source


def normalize_batch_grouping(batch_grouping: Optional[str]) -> str:
    """Normalize the scene batch grouping mode.

    Args:
        batch_grouping: User-provided grouping mode from Hydra.

    Returns:
        Canonical grouping mode.
    """
    if batch_grouping is None or str(batch_grouping) == "":
        return BATCH_GROUPING_NONE
    mode = str(batch_grouping)
    if mode not in BATCH_GROUPING_CHOICES:
        raise ValueError(f"Unsupported task.batch_grouping={mode}; choices={sorted(BATCH_GROUPING_CHOICES)}")
    return mode


def normalize_solver_reuse(solver_reuse: Optional[str]) -> str:
    """Normalize the solver reuse scope.

    Args:
        solver_reuse: User-provided solver reuse scope from Hydra.

    Returns:
        Canonical solver reuse scope, either ``type`` or ``batch``.
    """
    if solver_reuse is None or str(solver_reuse) == "":
        return SOLVER_REUSE_TYPE
    mode = str(solver_reuse)
    if mode not in SOLVER_REUSE_CHOICES:
        raise ValueError(f"Unsupported task.solver_reuse={mode}; choices={sorted(SOLVER_REUSE_CHOICES)}")
    return mode


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a config key while preserving a Python default for missing values.

    Args:
        cfg: OmegaConf config section or plain mapping.
        key: Key to read from the config.
        default: Value returned when the key is absent.

    Returns:
        Config value or ``default``.
    """
    if cfg is None:
        return default
    return cfg[key] if key in cfg else default


def _to_plain_value(value: Any) -> Any:
    """Convert OmegaConf containers into plain Python values.

    Args:
        value: Raw value read from a Hydra config.

    Returns:
        Plain Python value with interpolations resolved.
    """
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _to_optional_str(value: Any) -> Optional[str]:
    """Normalize optional string config values.

    Args:
        value: Raw config value.

    Returns:
        String value, or ``None`` when the config value is empty.
    """
    value = _to_plain_value(value)
    if value is None:
        return None
    return str(value)


def _to_optional_int(value: Any) -> Optional[int]:
    """Normalize optional integer config values.

    Args:
        value: Raw config value.

    Returns:
        Integer value, or ``None`` when the config value is empty.
    """
    value = _to_plain_value(value)
    if value is None:
        return None
    return int(value)


def _to_optional_float(value: Any) -> Optional[float]:
    """Normalize optional floating-point config values.

    Args:
        value: Raw config value.

    Returns:
        Floating-point value, or ``None`` when the config value is empty.
    """
    value = _to_plain_value(value)
    if value is None:
        return None
    return float(value)


def _to_optional_str_list(value: Any) -> Optional[List[str]]:
    """Normalize Hydra list or comma-separated string options.

    Args:
        value: Raw config value containing ``None``, a string, or a list.

    Returns:
        List of string tokens, or ``None`` when no list was provided.
    """
    value = _to_plain_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            return None
        return [token for token in value.split(",") if token]
    if isinstance(value, (int, float)):
        return [str(value)]
    return [str(item) for item in value]


def _to_optional_int_list(value: Any) -> Optional[List[int]]:
    """Normalize optional integer-list config values.

    Args:
        value: Raw config value containing ``None``, an integer, or a list.

    Returns:
        List of integers, or ``None`` when no list was provided.
    """
    value = _to_plain_value(value)
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        if value == "":
            return None
        return [int(token) for token in value.split(",") if token]
    return [int(item) for item in value]


def _drop_none_values(value: Any) -> Any:
    """Remove unset values from a plain config container.

    Args:
        value: Plain scalar, list, or dictionary converted from an OmegaConf node.

    Returns:
        Value with ``None`` entries recursively removed from dictionaries. Empty
        dictionaries are omitted by their parent.
    """
    value = _to_plain_value(value)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned_item = _drop_none_values(item)
            if cleaned_item is None:
                continue
            if isinstance(cleaned_item, dict) and not cleaned_item:
                continue
            cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def _deep_update(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge overrides into a dictionary.

    Args:
        base: Base dictionary mutated in place.
        overrides: Override dictionary whose non-dict values replace base values.

    Returns:
        The updated base dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _format_requirement_value(value: Any) -> str:
    """Format a baseline requirement value for error messages.

    Args:
        value: Value that should be rendered in a concise diagnostic message.

    Returns:
        String representation suitable for requirement mismatch errors.
    """
    return repr(value)


def _requirement_values_match(required: Any, actual: Any) -> bool:
    """Check whether an actual runtime value satisfies a required value.

    Args:
        required: Value declared by ``baseline_config.requires``.
        actual: Value resolved from the active task configuration.

    Returns:
        True when the values are considered equal.
    """
    if isinstance(required, bool) or isinstance(actual, bool):
        return isinstance(required, bool) and isinstance(actual, bool) and required == actual
    if isinstance(required, (int, float)) and isinstance(actual, (int, float)):
        return float(required) == float(actual)
    return required == actual


def _collect_requirement_mismatches(prefix: str, required: Any, actual: Any) -> List[str]:
    """Collect nested baseline requirement mismatches.

    Args:
        prefix: Dotted config path for the current value.
        required: Required value or nested mapping from a baseline config.
        actual: Actual runtime value or nested mapping.

    Returns:
        List of formatted mismatch messages.
    """
    if isinstance(required, dict):
        if not isinstance(actual, dict):
            return [
                f"{prefix} requires a mapping, got {_format_requirement_value(actual)}"
            ]
        mismatches = []
        for key, value in required.items():
            child_prefix = f"{prefix}.{key}"
            mismatches.extend(_collect_requirement_mismatches(child_prefix, value, actual.get(key)))
        return mismatches
    if not _requirement_values_match(required, actual):
        return [
            f"{prefix} requires {_format_requirement_value(required)}, got {_format_requirement_value(actual)}"
        ]
    return []


def _build_scene_source_overrides(task_cfg: Any) -> Dict[str, Any]:
    """Build Hydra-managed scene source configuration.

    Args:
        task_cfg: Hydra task config section.

    Returns:
        Scene source dictionary with unset fields removed.
    """
    return dict(_drop_none_values(_cfg_get(task_cfg, "scene_source", {})) or {})


def _build_human_prior_overrides(task_cfg: Any) -> Dict[str, Any]:
    """Build Hydra-managed human-prior configuration.

    Args:
        task_cfg: Hydra task config section.

    Returns:
        Human-prior dictionary with unset fields removed.
    """
    return dict(_drop_none_values(_cfg_get(task_cfg, "human_prior", {})) or {})


def get_baseline_configs_path() -> str:
    """Return the baseline override config directory.

    Args:
        None.

    Returns:
        Absolute path to ``src/curobo/content/configs/baselines``.
    """
    return os.path.join(get_configs_path(), "baselines")


def resolve_baseline_config_path(baseline_config: str) -> str:
    """Resolve a baseline config path supplied by Hydra.

    Args:
        baseline_config: Baseline config path, absolute or relative to the
            baseline config directory.

    Returns:
        Absolute path to the baseline YAML file.
    """
    if os.path.isabs(baseline_config):
        return baseline_config
    return join_path(get_baseline_configs_path(), baseline_config)


def load_baseline_config_file(baseline_config: str) -> Tuple[Dict, str]:
    """Load one baseline override config file.

    Args:
        baseline_config: Baseline config path, absolute or relative to
            ``src/curobo/content/configs/baselines``.

    Returns:
        Tuple containing the loaded baseline dictionary and absolute path.
    """
    baseline_path = resolve_baseline_config_path(baseline_config)
    return load_yaml(baseline_path), baseline_path


def _suite_name_from_config(suite_config: Dict, suite_config_path: Optional[str]) -> Optional[str]:
    """Infer a stable suite label from the loaded suite config.

    Args:
        suite_config: Loaded suite config dictionary.
        suite_config_path: Optional suite config path supplied by Hydra.

    Returns:
        Suite label from ``robot_name`` or the suite config file name.
    """
    robot_name = suite_config.get("robot_name")
    if robot_name:
        return str(robot_name)
    if suite_config_path is None:
        return None
    suite_file = os.path.splitext(os.path.basename(str(suite_config_path)))[0]
    return suite_file[4:] if suite_file.startswith("sim_") else suite_file


def _build_synthesis_args(cfg: DictConfig) -> SimpleNamespace:
    """Build normalized synthesis arguments from a Hydra config.

    Args:
        cfg: Full Hydra config from ``example_grasp/main.py``.

    Returns:
        Namespace used by the native synthesis runtime.
    """
    task_cfg = cfg.task
    init_source = canonical_init_source(str(_cfg_get(task_cfg, "init_source", "surface_sample")))
    save_mode = str(_cfg_get(task_cfg, "save_mode", "npy"))
    if save_mode not in {"npy", "none"}:
        raise ValueError(f"Unsupported task.save_mode={save_mode}; choices=['none', 'npy']")
    suite_config = _to_optional_str(_cfg_get(task_cfg, "suite_config", None))
    exp_name = _to_optional_str(_cfg_get(task_cfg, "exp_name", None))
    if exp_name is None:
        exp_name = _to_optional_str(_cfg_get(cfg, "name", None))
    return SimpleNamespace(
        manip_cfg_file=str(_cfg_get(cfg, "manip_cfg_file", "sim_shadow/tabletop_three.yml")),
        suite_config=suite_config,
        baseline_config=_to_optional_str(_cfg_get(task_cfg, "baseline_config", None)),
        init_source=init_source,
        grasp_suite=_to_optional_str(_cfg_get(task_cfg, "grasp_suite", None)),
        grasp_types=_to_optional_str_list(_cfg_get(task_cfg, "grasp_types", None)),
        scene_source_overrides=_build_scene_source_overrides(task_cfg),
        human_prior_overrides=_build_human_prior_overrides(task_cfg),
        dry_run=bool(_cfg_get(task_cfg, "dry_run", False)),
        dry_run_count=int(_cfg_get(task_cfg, "dry_run_count", 5)),
        save_folder=_to_optional_str(_cfg_get(task_cfg, "save_folder", None)),
        save_mode=save_mode,
        save_data=str(_cfg_get(task_cfg, "save_data", "all")),
        save_id=_to_optional_int_list(_cfg_get(task_cfg, "save_id", None)),
        obj_sample_hull_cache_mode=_to_optional_str(_cfg_get(task_cfg, "obj_sample_hull_cache_mode", None)),
        contact_obb_cache_mode=_to_optional_str(_cfg_get(task_cfg, "contact_obb_cache_mode", None)),
        contact_surface_points_mode=_to_optional_str(_cfg_get(task_cfg, "contact_surface_points_mode", None)),
        contact_surface_sample_num=_to_optional_int(_cfg_get(task_cfg, "contact_surface_sample_num", None)),
        warp_mesh_cache_max_entries=_to_optional_int(_cfg_get(task_cfg, "warp_mesh_cache_max_entries", None)),
        warp_mesh_cache_max_mb=_to_optional_int(_cfg_get(task_cfg, "warp_mesh_cache_max_mb", None)),
        save_debug=bool(_cfg_get(task_cfg, "save_debug", False)),
        parallel_world=int(_cfg_get(task_cfg, "parallel_world", 20)),
        batch_grouping=normalize_batch_grouping(_to_optional_str(_cfg_get(task_cfg, "batch_grouping", BATCH_GROUPING_NONE))),
        solver_reuse=normalize_solver_reuse(_to_optional_str(_cfg_get(task_cfg, "solver_reuse", SOLVER_REUSE_TYPE))),
        progress=bool(_cfg_get(task_cfg, "progress", True)),
        skip=bool(_cfg_get(task_cfg, "skip", True)),
        exp_name=exp_name,
        start=_to_optional_int(_cfg_get(task_cfg, "start", None)),
        end=_to_optional_int(_cfg_get(task_cfg, "end", None)),
        gpus=_to_optional_int_list(_cfg_get(task_cfg, "gpus", None)),
        profile=bool(_cfg_get(task_cfg, "profile", False)),
    )


def _resolve_hydra_output_dir() -> str:
    """Resolve the active Hydra run output directory.

    Args:
        None.

    Returns:
        Absolute path to the current Hydra output directory, or the current
        working directory when synthesis is invoked outside Hydra.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        return str(HydraConfig.get().runtime.output_dir)
    except Exception:
        return os.getcwd()


def _redirect_worker_output(log_path: str):
    """Redirect process stdout and stderr to a worker log file.

    Args:
        log_path: Destination file path for this GPU worker.

    Returns:
        Open file object that must stay alive for the worker process lifetime.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())
    return log_file


def _gpu_worker_entry(args_dict: Dict, batches: Sequence[Dict], gpu_id: int, log_path: Optional[str] = None) -> None:
    """Run assigned synthesis batches inside one GPU worker process.

    Args:
        args_dict: Serializable normalized synthesis arguments.
        batches: Batch-plan dictionaries assigned to this worker.
        gpu_id: Physical GPU id exposed to this worker as local cuda:0.
        log_path: Optional per-GPU log path used to capture stdout and stderr.

    Returns:
        None.
    """
    log_file = None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        if log_path:
            log_file = _redirect_worker_output(str(log_path))
            print(
                f"[synthesis-worker] gpu={gpu_id} pid={os.getpid()} "
                f"batches={len(batches)} solver_reuse={args_dict.get('solver_reuse')} log_path={log_path}",
                flush=True,
            )
        setup_logger("warn")
        _suppress_trimesh_even_sample_warning()
        _suppress_warp_deprecation_warnings()
        runtime = SynthesisRuntime(SimpleNamespace(**args_dict))
        runtime.execute_batches(list(batches), worker_label=f"gpu{gpu_id}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()


class SynthesisRuntime:
    """Native Hydra synthesis runtime with planning, scheduling, and execution.

    Args:
        args: Normalized synthesis arguments.

    Returns:
        None.
    """

    def __init__(self, args: SimpleNamespace):
        self.args = args
        self._plan_base_scene_paths_cache = {}
        self._plan_scene_paths_cache = {}
        self._plan_scene_id_cache = {}
        self._plan_scene_group_key_cache = {}
        self._plan_scene_height_cache = {}
        self._plan_human_prior_cache = {}
        self.profiler = SynthesisProfiler(enabled=bool(getattr(args, "profile", False)))

    def run(self) -> Dict:
        """Plan synthesis, print the plan, and execute it unless this is a dry-run.

        Args:
            None.

        Returns:
            Summary dictionary with scene, batch, and GPU assignment counts.
        """
        plan = self.build_plan()
        if not self.args.dry_run and not self.args.skip:
            self.confirm_and_cleanup_previous_outputs(plan)
        self.print_plan(plan)
        if self.args.dry_run:
            return {"dry_run": True, "type_count": len(plan["types"]), "batch_count": len(plan["batches"])}

        gpus = self.args.gpus or []
        if len(gpus) > 1:
            return self.run_multi_gpu(plan, gpus)
        if len(gpus) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        self.execute_batches(plan["batches"], worker_label="main")
        return {"dry_run": False, "type_count": len(plan["types"]), "batch_count": len(plan["batches"])}

    def build_plan(self) -> Dict:
        """Build a complete deterministic synthesis plan before solver creation.

        Args:
            None.

        Returns:
            Plan dictionary containing type summaries and executable batch records.
        """
        type_specs, suite_config = self.resolve_type_specs()
        suite_name = _suite_name_from_config(suite_config, self.args.suite_config)
        all_batches = []
        type_plans = []
        for type_spec in type_specs:
            type_plan = self.plan_one_type(type_spec, suite_config)
            type_plans.append(type_plan)
            all_batches.extend(type_plan["batches"])

        # First-version cost follows the user request: scene_count * K_t.
        all_batches.sort(key=lambda item: (-item["scene_count"], -item["estimated_cost"], item["type_name"]))
        for batch_index, batch in enumerate(all_batches, start=1):
            batch["global_batch_index"] = batch_index
        return {
            "init_source": self.args.init_source,
            "suite_config": self.args.suite_config,
            "baseline_config": getattr(self.args, "baseline_config", None),
            "baseline": suite_config.get("_baseline_config"),
            "suite_name": suite_name,
            "types": type_plans,
            "batches": all_batches,
        }

    def confirm_and_cleanup_previous_outputs(self, plan: Dict) -> None:
        """Ask whether stale output directories should be removed before execution.

        Args:
            plan: Plan dictionary returned by ``build_plan``.

        Returns:
            None.
        """
        cleanup_dirs = self.collect_existing_output_dirs_for_cleanup(plan)
        if not cleanup_dirs:
            log_warn("skip=false cleanup check: no previous related output folders were found.")
            return

        log_warn("skip=false cleanup check found previous related output folders:")
        for cleanup_dir in cleanup_dirs:
            log_warn(f"  {cleanup_dir}")
        try:
            answer = input("Delete these previous output folders before synthesis? (y/n): ").strip().lower()
        except EOFError:
            log_warn("No interactive input is available; previous output folders will be kept.")
            return
        if answer != "y":
            log_warn("Previous output folders will be kept.")
            return
        for cleanup_dir in cleanup_dirs:
            self.remove_output_dir(cleanup_dir)

    def collect_existing_output_dirs_for_cleanup(self, plan: Dict) -> List[str]:
        """Collect existing experiment output directories related to the current plan.

        Args:
            plan: Plan dictionary returned by ``build_plan``.

        Returns:
            Deduplicated absolute directories that currently exist under cuRobo's
            output root and can be offered for deletion.
        """
        candidates = []
        for type_plan in plan["types"]:
            candidates.append(self.output_experiment_dir_from_save_folder(type_plan["save_folder"]))
        candidates.append(self.plan_output_dir(plan))

        seen = set()
        cleanup_dirs = []
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.isdir(candidate):
                self.validate_cleanup_dir(candidate)
                cleanup_dirs.append(candidate)
        return cleanup_dirs

    def output_experiment_dir_from_save_folder(self, save_folder: str) -> str:
        """Resolve the experiment directory that owns one save folder.

        Args:
            save_folder: Save folder path relative to cuRobo's output root.

        Returns:
            Absolute experiment output directory. For the standard
            ``.../<exp_name>/graspdata`` layout this is ``.../<exp_name>``.
        """
        save_dir = os.path.abspath(os.path.join(get_output_path(), save_folder))
        if os.path.basename(save_dir) == "graspdata":
            return os.path.dirname(save_dir)
        return save_dir

    def plan_output_dir(self, plan: Dict) -> str:
        """Resolve the synthesis plan directory for the current experiment.

        Args:
            plan: Plan dictionary returned by ``build_plan``.

        Returns:
            Absolute directory that contains ``synthesis_plan.md``.
        """
        if self.args.save_folder is not None:
            return os.path.abspath(os.path.join(get_output_path(), self.args.save_folder))
        exp_label = self.args.exp_name or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        plan_group = plan.get("suite_name") or self.args.grasp_suite or "single"
        return os.path.abspath(
            os.path.join(get_output_path(), "synthesis_plans", str(plan_group), str(exp_label))
        )

    def validate_cleanup_dir(self, cleanup_dir: str) -> None:
        """Ensure a cleanup directory is scoped to cuRobo's output root.

        Args:
            cleanup_dir: Absolute directory proposed for deletion.

        Returns:
            None.
        """
        output_root = os.path.abspath(get_output_path())
        common_path = os.path.commonpath([output_root, cleanup_dir])
        if common_path != output_root or cleanup_dir == output_root:
            raise ValueError(f"Refusing to delete path outside output root: {cleanup_dir}")

    def remove_output_dir(self, cleanup_dir: str) -> None:
        """Delete one validated output directory.

        Args:
            cleanup_dir: Absolute output directory to remove.

        Returns:
            None.
        """
        self.validate_cleanup_dir(cleanup_dir)
        shutil.rmtree(cleanup_dir)
        log_warn(f"Deleted previous output folder: {cleanup_dir}")

    def is_suite_mode(self, suite_config: Dict) -> bool:
        """Decide whether the current run should select types from suite config.

        Args:
            suite_config: Loaded suite config dictionary.

        Returns:
            ``True`` when legacy ``task.grasp_suite`` is set, or when a
            ``task.suite_config`` plus ``task.grasp_types`` request selects
            one or more suite-defined grasp types.
        """
        if self.args.grasp_suite is not None:
            return True
        return self.args.suite_config is not None and self.args.grasp_types is not None and bool(suite_config.get("grasp_types"))

    def resolve_type_specs(self) -> tuple[List[Dict], Dict]:
        """Resolve selected grasp types and their manipulation configs.

        Args:
            None.

        Returns:
            Tuple of selected type specs and the loaded suite config.
        """
        suite_config = self.load_suite_config()
        if not self.is_suite_mode(suite_config):
            type_name = os.path.splitext(os.path.basename(self.args.manip_cfg_file))[0]
            type_config = {}
            if suite_config.get("grasp_types"):
                try:
                    _, type_name, type_config = grasp_type_from_suite_config(self.args.manip_cfg_file, suite_config)
                except ValueError:
                    if self.args.init_source == "human":
                        raise
            return [
                {
                    "type_name": str(type_name),
                    "manip_cfg_file": self.args.manip_cfg_file,
                    "type_config": dict(type_config or {}),
                }
            ], suite_config

        grasp_types = suite_config.get("grasp_types", {})
        if not grasp_types:
            raise ValueError("suite_config.grasp_types is required for suite synthesis")
        requested = self.args.grasp_types
        selected_names = []
        if requested is None or "all" in requested:
            selected_names = list(grasp_types.keys())
        else:
            for token in requested:
                matched_name = None
                for type_name, type_config in grasp_types.items():
                    type_id = type_config.get("type_id") if isinstance(type_config, dict) else str(type_name).split("_", 1)[0]
                    if token == type_name or token == str(type_id):
                        matched_name = type_name
                        break
                if matched_name is None:
                    raise ValueError(f"Unknown grasp type {token}; available types: {list(grasp_types.keys())}")
                selected_names.append(matched_name)
        specs = []
        for type_name in selected_names:
            type_config = grasp_types[type_name]
            manip_cfg_file = type_config if isinstance(type_config, str) else type_config["manip_cfg_file"]
            spec_config = {"manip_cfg_file": str(type_config)} if isinstance(type_config, str) else dict(type_config)
            specs.append({"type_name": str(type_name), "manip_cfg_file": str(manip_cfg_file), "type_config": spec_config})
        return specs, suite_config

    def load_baseline_config(self) -> Tuple[Optional[Dict], Optional[str]]:
        """Load the optional baseline override config.

        Args:
            None.

        Returns:
            Tuple of loaded baseline config and absolute path. Both values are
            ``None`` when ``task.baseline_config`` is unset.
        """
        baseline_config = getattr(self.args, "baseline_config", None)
        if baseline_config is None:
            return None, None
        baseline_data, baseline_path = load_baseline_config_file(str(baseline_config))
        if not isinstance(baseline_data, dict):
            raise ValueError(f"baseline_config must be a mapping, got {type(baseline_data)} from {baseline_path}")
        overrides = baseline_data.get("overrides", {})
        if overrides is None:
            baseline_data["overrides"] = {}
        elif not isinstance(overrides, dict):
            raise ValueError(f"baseline_config.overrides must be a mapping in {baseline_path}")
        return baseline_data, baseline_path

    def apply_baseline_config(self, suite_config: Dict, baseline_data: Optional[Dict], baseline_path: Optional[str]) -> Dict:
        """Apply baseline overrides to a loaded suite config.

        Args:
            suite_config: Suite config dictionary mutated in place.
            baseline_data: Loaded baseline config dictionary, or ``None``.
            baseline_path: Absolute baseline config path, or ``None``.

        Returns:
            Suite config after baseline overrides and metadata recording.
        """
        if baseline_data is None:
            return suite_config
        suite_robot_name = suite_config.get("robot_name")
        baseline_robot_name = baseline_data.get("robot_name")
        if suite_robot_name and baseline_robot_name and str(suite_robot_name) != str(baseline_robot_name):
            raise ValueError(
                f"baseline_config.robot_name={baseline_robot_name} does not match "
                f"suite_config.robot_name={suite_robot_name}"
            )
        baseline_overrides = copy.deepcopy(baseline_data.get("overrides", {}))
        explicit_suite_overrides = baseline_overrides.pop("suite", None)
        # Manipulation-config overrides are applied when each manip file is
        # loaded, so they should not become top-level suite keys.
        baseline_overrides.pop("manip", None)
        if explicit_suite_overrides is not None:
            if not isinstance(explicit_suite_overrides, dict):
                raise ValueError(f"baseline_config.overrides.suite must be a mapping in {baseline_path}")
            # Legacy top-level overrides are merged first for compatibility;
            # explicit suite overrides win when both specify the same key.
            suite_overrides = baseline_overrides
            _deep_update(suite_overrides, explicit_suite_overrides)
        else:
            suite_overrides = baseline_overrides
        _deep_update(suite_config, suite_overrides)
        suite_config["_baseline_config"] = {
            "path": baseline_path,
            "name": baseline_data.get("name"),
            "baseline": baseline_data.get("baseline"),
            "robot_name": baseline_robot_name,
        }
        return suite_config

    def validate_baseline_requirements(
        self,
        baseline_data: Optional[Dict],
        baseline_path: Optional[str],
        suite_config: Dict,
    ) -> None:
        """Validate runtime task settings required by a baseline config.

        Args:
            baseline_data: Loaded baseline config dictionary, or ``None``.
            baseline_path: Absolute baseline config path, or ``None``.
            suite_config: Suite config after baseline overrides have been
                applied.

        Returns:
            None. Raises ``ValueError`` when a declared requirement is not met.
        """
        if baseline_data is None:
            return
        requires = baseline_data.get("requires", {})
        if requires is None:
            return
        if not isinstance(requires, dict):
            raise ValueError(f"baseline_config.requires must be a mapping in {baseline_path}")
        task_requires = requires.get("task", {})
        if task_requires is None:
            return
        if not isinstance(task_requires, dict):
            raise ValueError(f"baseline_config.requires.task must be a mapping in {baseline_path}")
        supported_task_keys = {"init_source", "parallel_world", "scene_source"}
        unknown_keys = sorted(set(task_requires.keys()) - supported_task_keys)
        if unknown_keys:
            raise ValueError(
                f"Unsupported baseline_config.requires.task keys in {baseline_path}: {unknown_keys}; "
                f"supported keys are {sorted(supported_task_keys)}"
            )

        actual_task_values = {
            "init_source": self.args.init_source,
            "parallel_world": self.args.parallel_world,
            "scene_source": self.resolve_scene_source(suite_config),
        }
        mismatches = []
        for key, required_value in task_requires.items():
            mismatches.extend(
                _collect_requirement_mismatches(
                    f"task.{key}",
                    required_value,
                    actual_task_values.get(key),
                )
            )
        if mismatches:
            baseline_name = baseline_data.get("name") or baseline_data.get("baseline") or baseline_path
            detail = "; ".join(mismatches)
            raise ValueError(f"baseline_config {baseline_name} requirement mismatch: {detail}")

    def baseline_manip_overrides(self, manip_cfg_file: str) -> Dict:
        """Read baseline overrides for one manipulation config.

        Args:
            manip_cfg_file: Manipulation config path relative to
                ``src/curobo/content/configs/manip``.

        Returns:
            Override dictionary for this manipulation config, or an empty
            dictionary when the active baseline does not override it.
        """
        baseline_data, _ = self.load_baseline_config()
        if baseline_data is None:
            return {}
        manip_overrides = baseline_data.get("overrides", {}).get("manip", {})
        if manip_overrides is None:
            return {}
        if not isinstance(manip_overrides, dict):
            raise ValueError("baseline_config.overrides.manip must be a mapping")
        normalized_target = os.path.normpath(str(manip_cfg_file)).replace(os.sep, "/")
        for override_path, override_value in manip_overrides.items():
            normalized_override = os.path.normpath(str(override_path)).replace(os.sep, "/")
            if normalized_override == normalized_target:
                if override_value is None:
                    return {}
                if not isinstance(override_value, dict):
                    raise ValueError(f"baseline manip override for {override_path} must be a mapping")
                return copy.deepcopy(override_value)
        return {}

    def load_suite_config(self) -> Dict:
        """Load suite config and apply Hydra-managed defaults.

        Args:
            None.

        Returns:
            Suite config dictionary with optional baseline overrides and
            Hydra-managed human-prior overrides.
        """
        if self.args.suite_config is None:
            if (
                self.args.init_source == "human"
                or self.args.grasp_suite is not None
                or self.args.grasp_types is not None
            ):
                raise ValueError("task.suite_config is required for suite and human-prior synthesis")
            suite_config = {}
        else:
            suite_config = copy.deepcopy(load_suite_config_file(self.args.suite_config))
        baseline_data, baseline_path = self.load_baseline_config()
        suite_config = self.apply_baseline_config(suite_config, baseline_data, baseline_path)
        self.validate_baseline_requirements(baseline_data, baseline_path, suite_config)
        suite_human_prior = copy.deepcopy(suite_config.get("human_prior", {}))
        suite_config["human_prior"] = _deep_update(suite_human_prior, self.args.human_prior_overrides)
        if self.args.init_source == "human" and not suite_config["human_prior"].get("root"):
            raise ValueError(f"task.human_prior.root is required for {self.args.init_source} mode")
        return suite_config

    def resolve_scene_source(self, suite_config: Dict) -> Dict:
        """Resolve the scene selection policy for synthesis planning.

        Args:
            suite_config: Loaded suite config dictionary.

        Returns:
            Scene source dictionary from Hydra, or a suite-local fallback when a
            caller still provides ``scene_policy`` in a suite config.
        """
        suite_scene_source = suite_config.get(
            "scene_policy",
            suite_config.get("scene_source"),
        )
        if suite_scene_source is None:
            scene_source = {"use_object_scale_list": True}
        else:
            scene_source = copy.deepcopy(suite_scene_source)
        scene_source.update(self.args.scene_source_overrides)
        return scene_source

    def resolve_object_scale_list_for_type(
        self,
        type_config: Dict,
        manip_config_data: Dict,
        scene_source: Dict,
    ) -> Optional[Tuple[Any, ...]]:
        """Resolve the object scale filter for one grasp type.

        Args:
            type_config: Resolved suite grasp-type config after baseline
                overrides.
            manip_config_data: Loaded manipulation config dictionary.
            scene_source: Runtime scene source configuration.

        Returns:
            Tuple of object scales to keep, or ``None`` when no scale filter
            should be applied.
        """
        if not bool(scene_source.get("use_object_scale_list", False)):
            return None
        if isinstance(type_config, dict) and "object_scale_list" in type_config:
            return tuple(type_config.get("object_scale_list") or [])
        world_config = manip_config_data.get("world", {})
        if "object_scale_list" in world_config:
            return tuple(world_config.get("object_scale_list") or [])
        return None

    def collect_scene_paths_for_plan(
        self,
        manip_config_data: Dict,
        scene_source: Dict,
        object_scale_list: Optional[Tuple[Any, ...]] = None,
    ) -> List[str]:
        """Collect scene paths with per-runtime caching for repeated suite types.

        Args:
            manip_config_data: Loaded manipulation config dictionary.
            scene_source: Runtime scene source configuration.
            object_scale_list: Optional per-type object scale filter resolved
                from suite/baseline config, or legacy manip config fallback.

        Returns:
            Deterministic list of scene config paths for the current manipulation config.
        """
        world_config = manip_config_data["world"]
        if world_config.get("type") != "scene_cfg":
            raise NotImplementedError(f"Only world.type=scene_cfg is supported, got {world_config.get('type')}")

        template_path = scene_source.get("template_path") or world_config.get("template_path")
        shuffle_before_slice = bool(scene_source.get("shuffle_before_slice", False))
        shuffle_seed = int(scene_source.get("shuffle_seed", 123))
        start = world_config.get("start")
        end = world_config.get("end")
        object_scale_list = tuple(object_scale_list) if object_scale_list is not None else None
        cache_key = (
            template_path,
            shuffle_before_slice,
            shuffle_seed,
            start,
            end,
            object_scale_list,
        )
        if cache_key not in self._plan_scene_paths_cache:
            all_paths = self.collect_base_scene_paths_for_plan(str(template_path))
            if shuffle_before_slice:
                rng = np.random.default_rng(shuffle_seed)
                all_paths = [all_paths[index] for index in rng.permutation(len(all_paths))]
            # The runtime start/end range is a global scene-record slice. Grasp
            # type filters, such as object_scale_list, must only decide which
            # records from that shared slice belong to the current type.
            sliced_paths = all_paths[start:end]
            if object_scale_list is not None:
                scale_patterns = [f"scale{math.floor(scale * 100 + 0.5):03d}_" for scale in object_scale_list]
                sliced_paths = [path for path in sliced_paths if any(pattern in path for pattern in scale_patterns)]
            self._plan_scene_paths_cache[cache_key] = sliced_paths
        return list(self._plan_scene_paths_cache[cache_key])

    def collect_base_scene_paths_for_plan(self, template_path: str) -> List[str]:
        """Collect and cache scene paths for one template without per-type filters.

        Args:
            template_path: Scene glob path relative to cuRobo assets, or absolute.

        Returns:
            Sorted scene config paths matching the template.
        """
        scene_cfg_pattern = join_path(get_assets_path(), template_path)
        cache_key = os.path.abspath(scene_cfg_pattern)
        if cache_key not in self._plan_base_scene_paths_cache:
            scene_iter = iglob(scene_cfg_pattern, recursive=True)
            if bool(getattr(self.args, "progress", True)) and tqdm is not None:
                scene_iter = tqdm(
                    scene_iter,
                    desc=f"plan: scan {template_path}",
                    unit="scene",
                    dynamic_ncols=True,
                    leave=False,
                )
            self._plan_base_scene_paths_cache[cache_key] = sorted(scene_iter)
        return list(self._plan_base_scene_paths_cache[cache_key])

    def min_object_height_from_scene_source(self, scene_source: Dict) -> Optional[float]:
        """Read the optional scene height threshold from a scene source config.

        Args:
            scene_source: Runtime scene source configuration.

        Returns:
            Minimum allowed object height in meters, or ``None`` when disabled.
        """
        min_height = _to_optional_float(
            scene_source.get(
                "min_object_height",
                scene_source.get("minimum_object_height", scene_source.get("object_min_height")),
            )
        )
        if min_height is not None and min_height < 0.0:
            raise ValueError(f"scene_source.min_object_height must be non-negative, got {min_height}")
        return min_height

    def object_height_record_path(self, asset_root: str, scene_source: Dict) -> str:
        """Resolve the object-height JSONL file for one DGN asset root.

        Args:
            asset_root: Object asset folder that contains ``scene_cfg``.
            scene_source: Runtime scene source configuration.

        Returns:
            Absolute path to the JSONL file with per-scene object heights.
        """
        configured_path = scene_source.get("object_height_record_path") or scene_source.get("height_record_path")
        if configured_path:
            configured_path = str(configured_path)
            if os.path.isabs(configured_path):
                return configured_path
            return os.path.join(asset_root, configured_path)
        record_file = str(scene_source.get("object_height_record_file", DEFAULT_OBJECT_HEIGHT_RECORD_FILE))
        return os.path.join(asset_root, record_file)

    def scene_asset_root_and_key(self, scene_path: str) -> Tuple[str, str]:
        """Split a scene path into its DGN asset root and height-record key.

        Args:
            scene_path: Absolute path to one scene config file.

        Returns:
            Tuple of ``(asset_root, scene_key)`` where ``scene_key`` matches the
            relative ``scene_path`` field in ``tabletop_scene_object_heights.jsonl``.
        """
        normalized_path = os.path.abspath(scene_path)
        parts = normalized_path.split(os.sep)
        try:
            scene_cfg_index = parts.index("scene_cfg")
        except ValueError as exc:
            raise ValueError(f"Scene path does not contain a scene_cfg directory: {scene_path}") from exc
        asset_root = os.sep.join(parts[:scene_cfg_index])
        scene_key = os.path.relpath(normalized_path, asset_root).replace(os.sep, "/")
        return asset_root, scene_key

    def load_scene_height_map(self, asset_root: str, scene_source: Dict) -> Dict[str, float]:
        """Load cached per-scene object heights for one DGN asset root.

        Args:
            asset_root: Object asset folder that contains the height JSONL file.
            scene_source: Runtime scene source configuration.

        Returns:
            Mapping from relative scene config path to posed and scaled object height.
        """
        height_record_path = self.object_height_record_path(asset_root, scene_source)
        cache_key = os.path.abspath(height_record_path)
        if cache_key not in self._plan_scene_height_cache:
            if not os.path.exists(height_record_path):
                raise FileNotFoundError(
                    f"Object height record not found: {height_record_path}. "
                    "Run scripts/filter_tabletop_scene_cfg_by_height.py first."
                )
            height_map = {}
            with open(height_record_path, "r", encoding="utf-8") as file_obj:
                line_iter = file_obj
                if bool(getattr(self.args, "progress", True)) and tqdm is not None:
                    line_iter = tqdm(
                        file_obj,
                        desc=f"plan: load heights {os.path.basename(asset_root)}",
                        unit="line",
                        dynamic_ncols=True,
                        leave=False,
                    )
                for line_index, line in enumerate(line_iter, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    try:
                        height_map[str(record["scene_path"])] = float(record["height"])
                    except KeyError as exc:
                        raise KeyError(f"Missing {exc} in {height_record_path}:{line_index}") from exc
            self._plan_scene_height_cache[cache_key] = height_map
        return self._plan_scene_height_cache[cache_key]

    def filter_scene_paths_by_object_height(
        self,
        scene_paths: Sequence[str],
        scene_source: Dict,
        progress_desc: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        """Remove scenes whose saved object height is below the suite threshold.

        Args:
            scene_paths: Candidate scene config paths collected for planning.
            scene_source: Runtime scene source configuration containing
                ``min_object_height`` and optional height record path settings.
            progress_desc: Optional tqdm description used while checking heights.

        Returns:
            Tuple of filtered scene paths and the number of removed scenes.
        """
        min_height = self.min_object_height_from_scene_source(scene_source)
        if min_height is None:
            return list(scene_paths), 0

        filtered_paths = []
        filtered_count = 0
        for scene_path in self.iter_plan_progress(scene_paths, progress_desc or "plan: filter scene heights", unit="scene"):
            asset_root, scene_key = self.scene_asset_root_and_key(scene_path)
            height_map = self.load_scene_height_map(asset_root, scene_source)
            if scene_key not in height_map:
                raise KeyError(
                    f"Missing object height for scene {scene_key} in "
                    f"{self.object_height_record_path(asset_root, scene_source)}"
                )
            if height_map[scene_key] < min_height:
                filtered_count += 1
                continue
            filtered_paths.append(scene_path)
        return filtered_paths, filtered_count

    def scene_id_for_plan(self, scene_path: str) -> str:
        """Read a scene id once during planning and reuse it across grasp types.

        Args:
            scene_path: Path to one scene config file.

        Returns:
            Scene id stored in the scene config.
        """
        if scene_path not in self._plan_scene_id_cache:
            self._plan_scene_id_cache[scene_path] = scene_id_from_scene_path(scene_path)
        return self._plan_scene_id_cache[scene_path]

    def iter_plan_progress(self, items: Sequence, desc: str, unit: str = "it") -> Iterable:
        """Wrap planning work in a tqdm progress bar when progress output is enabled.

        Args:
            items: Sized sequence of planning items.
            desc: Progress bar description.
            unit: Unit name displayed by tqdm.

        Returns:
            Iterable over the original items, optionally wrapped by tqdm.
        """
        if not bool(getattr(self.args, "progress", True)) or tqdm is None:
            return items
        return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True, leave=False)

    def scene_group_key_for_plan(self, scene_path: str) -> Tuple[str, Tuple[float, float, float]]:
        """Read the object/scale grouping key for one scene during planning.

        Args:
            scene_path: Path to one scene config file.

        Returns:
            Tuple containing an object identity string and normalized xyz scale.
        """
        if scene_path not in self._plan_scene_group_key_cache:
            scene_cfg = load_scene_cfg(scene_path)
            obj_name = scene_cfg["task"]["obj_name"]
            obj_cfg = scene_cfg["scene"][obj_name]
            self._plan_scene_group_key_cache[scene_path] = (
                self.object_identity_for_grouping(obj_name, obj_cfg),
                self.scale_tuple_for_grouping(obj_cfg.get("scale", 1.0)),
            )
        return self._plan_scene_group_key_cache[scene_path]

    def object_identity_for_grouping(self, obj_name: str, obj_cfg: Dict) -> str:
        """Build a stable object identity for batch locality grouping.

        Args:
            obj_name: Object name from the scene task section.
            obj_cfg: Rigid object config from the scene config.

        Returns:
            String identity that is independent of scene pose.
        """
        for key in ("file_path", "urdf_path", "info_path"):
            value = obj_cfg.get(key)
            if value:
                object_path = os.path.normpath(str(value))
                if os.path.isabs(object_path):
                    try:
                        return os.path.relpath(object_path, REPO_ROOT)
                    except ValueError:
                        return object_path
                return object_path
        return str(obj_name)

    def scale_tuple_for_grouping(self, scale: Any) -> Tuple[float, float, float]:
        """Normalize scalar or xyz scale into an xyz tuple.

        Args:
            scale: Scene object scale stored as a scalar, sequence, or numpy array.

        Returns:
            Three-float scale tuple used for sorting and grouping.
        """
        scale_array = np.asarray(scale, dtype=np.float64).reshape(-1)
        if scale_array.size == 0:
            return (1.0, 1.0, 1.0)
        if scale_array.size == 1:
            scale_value = float(scale_array[0])
            return (scale_value, scale_value, scale_value)
        if scale_array.size != 3:
            raise ValueError(f"Object scale must be scalar or xyz, got shape={np.asarray(scale).shape}")
        return tuple(float(value) for value in scale_array.tolist())

    def format_scene_group_key(self, group_key: Tuple[str, Tuple[float, float, float]]) -> str:
        """Convert a scene grouping key into a compact plan string.

        Args:
            group_key: Object identity and xyz scale tuple.

        Returns:
            Human-readable grouping key for plans and batch records.
        """
        object_identity, scale = group_key
        scale_text = ",".join(f"{value:.9g}" for value in scale)
        return f"{object_identity}|scale={scale_text}"

    def order_scene_paths_for_batching(self, scene_paths: Sequence[str], progress_desc: Optional[str] = None) -> List[str]:
        """Apply optional planning-only scene grouping before chunking batches.

        Args:
            scene_paths: Pending scene config paths in legacy deterministic order.
            progress_desc: Optional tqdm description used while loading group keys.

        Returns:
            Scene paths ordered for batch chunking.
        """
        if self.args.batch_grouping == BATCH_GROUPING_NONE:
            return list(scene_paths)
        if self.args.batch_grouping == BATCH_GROUPING_OBJECT_SCALE:
            scene_paths = list(scene_paths)
            if progress_desc:
                for scene_path in self.iter_plan_progress(scene_paths, progress_desc, unit="scene"):
                    self.scene_group_key_for_plan(scene_path)
            return sorted(scene_paths, key=self.scene_group_key_for_plan)
        raise ValueError(
            f"Unsupported task.batch_grouping={self.args.batch_grouping}; choices={sorted(BATCH_GROUPING_CHOICES)}"
        )

    def batch_group_label_for_plan(self, scene_paths: Sequence[str]) -> Optional[str]:
        """Summarize grouping keys represented by one planned batch.

        Args:
            scene_paths: Scene config paths assigned to one solver batch.

        Returns:
            Group label string, or ``None`` when grouping is disabled.
        """
        if self.args.batch_grouping == BATCH_GROUPING_NONE:
            return None
        unique_keys = []
        seen_keys = set()
        for scene_path in scene_paths:
            group_key = self.scene_group_key_for_plan(scene_path)
            if group_key not in seen_keys:
                unique_keys.append(group_key)
                seen_keys.add(group_key)
        if not unique_keys:
            return "empty"
        if len(unique_keys) == 1:
            return self.format_scene_group_key(unique_keys[0])
        return f"mixed:{len(unique_keys)} groups"

    def batch_group_counts_for_plan(self, scene_paths: Sequence[str]) -> Dict[str, int]:
        """Count object/scale group membership for one planned batch.

        Args:
            scene_paths: Scene config paths assigned to one solver batch.

        Returns:
            Mapping from formatted object/scale group key to scene count.
        """
        group_counts = defaultdict(int)
        for scene_path in scene_paths:
            group_key = self.format_scene_group_key(self.scene_group_key_for_plan(scene_path))
            group_counts[group_key] += 1
        return dict(sorted(group_counts.items(), key=lambda item: (-item[1], item[0])))

    def primary_batch_group_key_for_plan(self, group_counts: Dict[str, int]) -> Optional[str]:
        """Return the dominant object/scale group for one planned batch.

        Args:
            group_counts: Batch scene-group counts produced by ``batch_group_counts_for_plan``.

        Returns:
            Dominant formatted group key, or ``None`` when the batch is empty.
        """
        if not group_counts:
            return None
        return min(group_counts.items(), key=lambda item: (-item[1], item[0]))[0]

    def human_prior_plan_record(
        self,
        scene_path: str,
        scene_id: str,
        prior_root: str,
        total_budget: int,
        seed_builder: HumanPriorSeedBuilder,
    ) -> Dict:
        """Load and cache the planning-only human-prior budget metadata for a scene.

        Args:
            scene_path: Path to one scene config file.
            scene_id: Scene id for the scene config.
            prior_root: Root directory containing per-scene prior files.
            total_budget: Total budget distributed across all human-prior grasp types.
            seed_builder: Current seed builder providing budget bounds.

        Returns:
            Cached planning metadata with scores, per-type budgets, and sample count.
        """
        cache_key = (
            prior_root,
            int(total_budget),
            repr(seed_builder.min_type_budget),
            repr(seed_builder.max_type_budget),
            float(seed_builder.score_threshold),
            int(seed_builder.budget_resolution),
            seed_builder.budget_rounding_mode,
            scene_id,
        )
        if cache_key not in self._plan_human_prior_cache:
            record, prior_file = load_human_prior_record(prior_root, scene_id)
            budget_scores = np.asarray(record["budget_scores"], dtype=np.float32)
            type_budgets = allocate_type_budget(
                budget_scores,
                total_budget=total_budget,
                min_budget=seed_builder.min_type_budget,
                max_budget=seed_builder.max_type_budget,
                score_threshold=seed_builder.score_threshold,
                budget_resolution=seed_builder.budget_resolution,
                budget_rounding_mode=seed_builder.budget_rounding_mode,
            )
            self._plan_human_prior_cache[cache_key] = {
                "scene_path": scene_path,
                "scene_id": scene_id,
                "prior_file": prior_file,
                "budget_scores": budget_scores,
                "type_budgets": type_budgets,
                "sample_count": int(np.asarray(record["index_mcp_pos"]).shape[1]),
            }
        return self._plan_human_prior_cache[cache_key]

    def plan_one_type(self, type_spec: Dict, suite_config: Dict) -> Dict:
        """Plan one grasp type from scene collection through solver batches.

        Args:
            type_spec: Selected type entry containing type name and manip config path.
            suite_config: Loaded suite config dictionary.

        Returns:
            Type plan dictionary with stats, dry-run samples, and batch records.
        """
        manip_cfg_file = type_spec["manip_cfg_file"]
        manip_config_data = self.load_manip_config(manip_cfg_file)
        scene_source = self.resolve_scene_source(suite_config)
        type_name_for_progress = str(type_spec.get("type_name", os.path.splitext(os.path.basename(manip_cfg_file))[0]))
        object_scale_list = self.resolve_object_scale_list_for_type(
            type_spec.get("type_config", {}),
            manip_config_data,
            scene_source,
        )
        scene_paths = self.collect_scene_paths_for_plan(manip_config_data, scene_source, object_scale_list)
        scene_paths, height_filtered_count = self.filter_scene_paths_by_object_height(
            scene_paths,
            scene_source,
            progress_desc=f"plan {type_name_for_progress}: filter heights",
        )
        save_folder = self.resolve_save_folder(manip_cfg_file, manip_config_data)
        stats = {
            "total": len(scene_paths) + int(height_filtered_count),
            "height_filtered": int(height_filtered_count),
            "skipped": 0,
            "zero_budget": 0,
            "pending": 0,
        }
        dry_samples = []
        budget_summary = defaultdict(int)
        batches = []

        if self.args.init_source == "surface_sample":
            pending_scene_paths = []
            for scene_path in self.iter_plan_progress(
                scene_paths,
                desc=f"plan {type_name_for_progress}: scan scenes",
                unit="scene",
            ):
                scene_id = self.scene_id_for_plan(scene_path)
                if self.args.skip and self.output_exists(save_folder, f"{scene_id}_"):
                    stats["skipped"] += 1
                    continue
                pending_scene_paths.append(scene_path)
                if len(dry_samples) < self.args.dry_run_count:
                    dry_samples.append({"scene_id": scene_id, "scene_path": scene_path})
            seed_count = int(manip_config_data.get("seed_num", 1))
            budget_summary[seed_count] = len(pending_scene_paths)
            ordered_scene_paths = self.order_scene_paths_for_batching(
                pending_scene_paths,
                progress_desc=f"plan {type_name_for_progress}: group scenes",
            )
            for local_batch_index, scene_batch in enumerate(chunk_sequence(ordered_scene_paths, self.args.parallel_world), start=1):
                batches.append(
                    self.make_batch_record(type_spec, manip_cfg_file, save_folder, local_batch_index, scene_batch, seed_count)
                )
            stats["pending"] = len(pending_scene_paths)
        else:
            type_id, type_name, type_config = grasp_type_from_suite_config(manip_cfg_file, suite_config)
            type_spec["type_name"] = type_name
            seed_builder = HumanPriorSeedBuilder(
                manip_config_data,
                suite_config,
                type_id=type_id,
                type_name=type_name,
                type_config=type_config,
            )
            total_budget = int(suite_config["human_prior"].get("total_budget", 40))
            grouped_scene_paths = defaultdict(list)
            prior_root = suite_config["human_prior"]["root"]
            for scene_path in self.iter_plan_progress(
                scene_paths,
                desc=f"plan {type_name_for_progress}: scan budgets",
                unit="scene",
            ):
                scene_id = self.scene_id_for_plan(scene_path)
                if self.args.skip and self.output_exists(save_folder, f"{scene_id}_"):
                    stats["skipped"] += 1
                    continue
                plan_record = self.human_prior_plan_record(
                    scene_path,
                    scene_id,
                    prior_root,
                    total_budget,
                    seed_builder,
                )
                type_budget = int(plan_record["type_budgets"][seed_builder.type_index])
                if len(dry_samples) < self.args.dry_run_count:
                    sample_indices, replacement_mask = seed_builder.select_sample_indices(
                        scene_id,
                        int(plan_record["sample_count"]),
                        type_budget,
                    )
                    dry_samples.append(
                        {
                            "scene_id": scene_id,
                            "scores": plan_record["budget_scores"].tolist(),
                            "budgets": plan_record["type_budgets"].tolist(),
                            "K_t": type_budget,
                            "sample_indices": sample_indices.tolist(),
                            "replacement_mask": replacement_mask.astype(int).tolist(),
                        }
                    )
                if type_budget == 0:
                    stats["zero_budget"] += 1
                    continue
                stats["pending"] += 1
                budget_summary[type_budget] += 1
                grouped_scene_paths[type_budget].append(scene_path)

            for type_budget, budget_scene_paths in sorted(
                grouped_scene_paths.items(), key=lambda item: (-len(item[1]), int(item[0]))
            ):
                parallel_world = self.compute_adaptive_parallel_world(int(type_budget), suite_config)
                ordered_scene_paths = self.order_scene_paths_for_batching(
                    budget_scene_paths,
                    progress_desc=f"plan {type_name_for_progress}: group K={int(type_budget)}",
                )
                for scene_batch in chunk_sequence(ordered_scene_paths, parallel_world):
                    batches.append(
                        self.make_batch_record(
                            type_spec,
                            manip_cfg_file,
                            save_folder,
                            len(batches) + 1,
                            scene_batch,
                            int(type_budget),
                        )
                    )

        return {
            "type_name": type_spec["type_name"],
            "manip_cfg_file": manip_cfg_file,
            "robot_file": manip_config_data["robot_file"],
            "save_folder": save_folder,
            "object_scale_list": list(object_scale_list) if object_scale_list is not None else None,
            "stats": stats,
            "budget_summary": dict(sorted((int(k), int(v)) for k, v in budget_summary.items())),
            "dry_samples": dry_samples,
            "batches": batches,
        }

    def make_batch_record(
        self,
        type_spec: Dict,
        manip_cfg_file: str,
        save_folder: str,
        local_batch_index: int,
        scene_batch: Sequence[str],
        type_budget: int,
    ) -> Dict:
        """Create one executable batch record.

        Args:
            type_spec: Current type spec.
            manip_cfg_file: Manipulation config path.
            save_folder: Output folder relative to cuRobo output root.
            local_batch_index: Batch index within this type.
            scene_batch: Scene paths assigned to this batch.
            type_budget: Seed count used per scene for cost and solver sizing.

        Returns:
            Serializable batch record dictionary.
        """
        scene_paths = list(scene_batch)
        scene_count = len(scene_paths)
        grasp_count = int(scene_count) * int(max(1, type_budget))
        record = {
            "type_name": type_spec["type_name"],
            "type_config": type_spec.get("type_config", {}),
            "manip_cfg_file": manip_cfg_file,
            "save_folder": save_folder,
            "local_batch_index": int(local_batch_index),
            "scene_paths": scene_paths,
            "scene_count": int(scene_count),
            "type_budget": int(type_budget),
            "grasp_count": int(grasp_count),
            "estimated_cost": int(grasp_count),
        }
        group_counts = self.batch_group_counts_for_plan(scene_paths)
        if group_counts:
            record["scene_group_counts"] = group_counts
            record["primary_group_key"] = self.primary_batch_group_key_for_plan(group_counts)
        batch_group_key = self.batch_group_label_for_plan(scene_paths)
        if batch_group_key is not None:
            record["batch_group_key"] = batch_group_key
        return record

    def compute_adaptive_parallel_world(self, type_budget: int, suite_config: Dict) -> int:
        """Compute scene batch size for a fixed per-scene budget.

        Args:
            type_budget: Per-scene seed budget.
            suite_config: Suite config containing human-prior batch settings.

        Returns:
            Positive scene count for the solver batch.
        """
        human_prior_cfg = suite_config.get("human_prior", {})
        target_batch_grasps = human_prior_cfg.get("target_batch_grasps")
        max_parallel_world = human_prior_cfg.get("max_parallel_world")
        if target_batch_grasps is None:
            return int(self.args.parallel_world)
        if int(target_batch_grasps) <= 0:
            raise ValueError(f"human_prior.target_batch_grasps must be positive, got {target_batch_grasps}")
        parallel_world = max(1, int(target_batch_grasps) // int(type_budget))
        if max_parallel_world is not None:
            if int(max_parallel_world) <= 0:
                raise ValueError(f"human_prior.max_parallel_world must be positive, got {max_parallel_world}")
            parallel_world = min(parallel_world, int(max_parallel_world))
        return max(1, int(parallel_world))

    def print_plan(self, plan: Dict) -> None:
        """Write the full plan to markdown and print a compact console summary.

        Args:
            plan: Plan dictionary returned by ``build_plan``.

        Returns:
            None.
        """
        total_scene = sum(type_plan["stats"]["total"] for type_plan in plan["types"])
        total_height_filtered = sum(type_plan["stats"].get("height_filtered", 0) for type_plan in plan["types"])
        total_pending = sum(type_plan["stats"]["pending"] for type_plan in plan["types"])
        total_skipped = sum(type_plan["stats"]["skipped"] for type_plan in plan["types"])
        total_zero = sum(type_plan["stats"]["zero_budget"] for type_plan in plan["types"])
        total_cost = sum(batch["estimated_cost"] for batch in plan["batches"])
        plan_path = self.write_plan_markdown(
            plan,
            {
                "scene_records": total_scene,
                "height_filtered": total_height_filtered,
                "pending": total_pending,
                "skipped": total_skipped,
                "zero_budget": total_zero,
                "estimated_cost": total_cost,
            },
        )
        log_warn(
            f"Synthesis plan analyzed scene_records={total_scene} pending={total_pending} "
            f"height_filtered={total_height_filtered} skipped={total_skipped} zero_budget={total_zero} "
            f"batches={len(plan['batches'])} estimated_cost={total_cost}; saved markdown to {plan_path}"
        )

    def write_plan_markdown(self, plan: Dict, summary: Dict) -> str:
        """Write the detailed synthesis plan to a markdown file under output.

        Args:
            plan: Plan dictionary returned by ``build_plan``.
            summary: Precomputed summary counters for the whole plan.

        Returns:
            Absolute markdown path.
        """
        if self.args.save_folder is not None:
            plan_path = os.path.join(get_output_path(), self.args.save_folder, "synthesis_plan.md")
        else:
            exp_label = self.args.exp_name or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            plan_group = plan.get("suite_name") or self.args.grasp_suite or "single"
            plan_path = os.path.join(get_output_path(), "synthesis_plans", str(plan_group), str(exp_label), "synthesis_plan.md")
        try:
            os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        except OSError:
            # Hydra dry-runs in restricted environments can make the source output
            # root read-only. Fall back to the run directory instead of aborting
            # before the user sees the generated plan.
            exp_label = self.args.exp_name or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            plan_group = plan.get("suite_name") or self.args.grasp_suite or "single"
            plan_path = os.path.join("/tmp", "bimanbodex_synthesis_plans", str(plan_group), str(exp_label), "synthesis_plan.md")
            os.makedirs(os.path.dirname(plan_path), exist_ok=True)

        lines = [
            "# Synthesis Plan",
            "",
            f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Init source: `{plan['init_source']}`",
            f"- Suite config: `{plan['suite_config']}`",
            f"- Baseline config: `{plan.get('baseline_config')}`",
            f"- Baseline name: `{(plan.get('baseline') or {}).get('name')}`",
            f"- Baseline robot: `{(plan.get('baseline') or {}).get('robot_name')}`",
            f"- GPUs: `{self.args.gpus or []}`",
            f"- Batch grouping: `{self.args.batch_grouping}`",
            f"- Solver reuse: `{self.args.solver_reuse}`",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Types | {len(plan['types'])} |",
            f"| Scene records | {summary['scene_records']} |",
            f"| Height filtered | {summary['height_filtered']} |",
            f"| Pending | {summary['pending']} |",
            f"| Skipped | {summary['skipped']} |",
            f"| Zero budget | {summary['zero_budget']} |",
            f"| Batches | {len(plan['batches'])} |",
            f"| Estimated cost | {summary['estimated_cost']} |",
            "",
            "## Type Summary",
            "",
            "| Type | Object Scale List | Total | Height Filtered | Pending | Skipped | Zero Budget | Batches | Budget Distribution |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for type_plan in plan["types"]:
            stats = type_plan["stats"]
            budget_text = ", ".join(
                f"K_t={budget}: scenes={count}" for budget, count in type_plan["budget_summary"].items()
            )
            if not budget_text:
                budget_text = "empty"
            object_scale_list = type_plan.get("object_scale_list")
            scale_text = "none" if object_scale_list is None else ", ".join(str(scale) for scale in object_scale_list)
            if scale_text == "":
                scale_text = "empty"
            lines.append(
                f"| `{type_plan['type_name']}` | `{scale_text}` | {stats['total']} | {stats.get('height_filtered', 0)} | "
                f"{stats['pending']} | {stats['skipped']} | {stats['zero_budget']} | {len(type_plan['batches'])} | "
                f"{budget_text} |"
            )

        if self.args.batch_grouping != BATCH_GROUPING_NONE:
            grouping_summary = defaultdict(lambda: {"batches": 0, "scenes": 0, "cost": 0})
            for batch in plan["batches"]:
                group_key = batch.get("batch_group_key", "untracked")
                grouping_summary[group_key]["batches"] += 1
                grouping_summary[group_key]["scenes"] += int(batch["scene_count"])
                grouping_summary[group_key]["cost"] += int(batch["estimated_cost"])
            lines.extend(
                [
                    "",
                    "## Batch Grouping Summary",
                    "",
                    "| Group Key | Batches | Scenes | Estimated Cost |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for group_key, item in sorted(
                grouping_summary.items(),
                key=lambda group: (-group[1]["cost"], str(group[0])),
            ):
                lines.append(f"| `{group_key}` | {item['batches']} | {item['scenes']} | {item['cost']} |")

        gpus = self.args.gpus or []
        if len(gpus) > 1:
            lines.extend(
                [
                    "",
                    "## Multi-GPU Assignment",
                    "",
                    "| GPU | Assigned Batches | Assigned Scenes | Estimated Cost |",
                    "| ---: | ---: | ---: | ---: |",
                ]
            )
            assignments = self.assign_batches_to_gpus(plan["batches"], gpus)
            for gpu_id in gpus:
                gpu_batches = assignments[gpu_id]
                gpu_cost = sum(batch["estimated_cost"] for batch in gpu_batches)
                gpu_scenes = sum(batch["scene_count"] for batch in gpu_batches)
                lines.append(f"| {gpu_id} | {len(gpu_batches)} | {gpu_scenes} | {gpu_cost} |")
            lines.extend(["", "### Assignment Details", ""])
            for gpu_id in gpus:
                gpu_batches = assignments[gpu_id]
                lines.extend(
                    [
                        f"#### GPU {gpu_id}",
                        "",
                        "| Type | K_t | Batches | Scenes | Cost |",
                        "| --- | ---: | ---: | ---: | ---: |",
                    ]
                )
                assignment_groups = defaultdict(lambda: {"batches": 0, "scenes": 0, "cost": 0})
                for batch in gpu_batches:
                    key = (batch["type_name"], int(batch["type_budget"]))
                    assignment_groups[key]["batches"] += 1
                    assignment_groups[key]["scenes"] += int(batch["scene_count"])
                    assignment_groups[key]["cost"] += int(batch["estimated_cost"])
                for (type_name, type_budget), item in sorted(
                    assignment_groups.items(),
                    key=lambda group: (-group[1]["cost"], group[0][0], group[0][1]),
                ):
                    lines.append(
                        f"| `{type_name}` | {type_budget} | {item['batches']} | {item['scenes']} | {item['cost']} |"
                    )
                lines.append("")

        try:
            with open(plan_path, "w", encoding="utf-8") as file:
                file.write("\n".join(lines).rstrip() + "\n")
        except OSError:
            exp_label = self.args.exp_name or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            plan_group = plan.get("suite_name") or self.args.grasp_suite or "single"
            plan_path = os.path.join("/tmp", "bimanbodex_synthesis_plans", str(plan_group), str(exp_label), "synthesis_plan.md")
            os.makedirs(os.path.dirname(plan_path), exist_ok=True)
            with open(plan_path, "w", encoding="utf-8") as file:
                file.write("\n".join(lines).rstrip() + "\n")
        return plan_path

    def run_multi_gpu(self, plan: Dict, gpus: Sequence[int]) -> Dict:
        """Launch one worker process per requested GPU and wait for completion.

        Args:
            plan: Planned synthesis work.
            gpus: GPU ids requested by the user.

        Returns:
            Summary dictionary for the multi-GPU run.
        """
        assignments = self.assign_batches_to_gpus(plan["batches"], gpus)
        args_dict = vars(self.args).copy()
        worker_log_dir = os.path.join(_resolve_hydra_output_dir(), "gpu_logs")
        os.makedirs(worker_log_dir, exist_ok=True)
        log_warn(f"Multi-GPU worker logs will be saved to {worker_log_dir}")
        workers = []
        for gpu_id in gpus:
            gpu_batches = assignments[gpu_id]
            if not gpu_batches:
                continue
            worker_log_path = os.path.join(worker_log_dir, f"gpu_{int(gpu_id)}.log")
            process = mp.Process(target=_gpu_worker_entry, args=(args_dict, gpu_batches, int(gpu_id), worker_log_path))
            process.start()
            workers.append((gpu_id, process))
            log_warn(
                f"Started synthesis worker pid={process.pid} gpu={gpu_id} "
                f"batches={len(gpu_batches)} log={worker_log_path}"
            )
        failed = []
        for gpu_id, process in workers:
            process.join()
            if process.exitcode != 0:
                failed.append((gpu_id, process.exitcode))
        if failed:
            raise RuntimeError(f"Multi-GPU synthesis failed: {failed}")
        return {"dry_run": False, "type_count": len(plan["types"]), "batch_count": len(plan["batches"]), "gpus": list(gpus)}

    def assign_batches_to_gpus(self, batches: Sequence[Dict], gpus: Sequence[int]) -> Dict[int, List[Dict]]:
        """Assign planned batches to GPUs using locality-aware greedy balancing.

        Args:
            batches: Planned batch records.
            gpus: Available GPU ids.

        Returns:
            Mapping from GPU id to assigned batch records.
        """
        assignments = {int(gpu): [] for gpu in gpus}
        loads = {int(gpu): 0 for gpu in gpus}
        gpu_group_counts = {int(gpu): defaultdict(int) for gpu in gpus}
        sorted_batches = sorted(batches, key=lambda item: (-item["estimated_cost"], -item["scene_count"], item["type_name"]))
        gpu_count = max(1, len(gpus))
        for batch in sorted_batches:
            batch_group_counts = batch.get("scene_group_counts", {})
            primary_group_key = batch.get("primary_group_key")
            min_load = min(loads.values())
            # Allow locality to steer placement when the target GPU is not too far
            # from the lightest current worker. The slack scales with one batch's
            # cost amortized across workers, and never drops below the scene count.
            load_slack = max(int(batch["scene_count"]), int(batch["estimated_cost"]) // gpu_count)
            candidate_gpus = [gpu for gpu in gpus if loads[int(gpu)] <= min_load + load_slack]
            if not candidate_gpus:
                candidate_gpus = list(gpus)
            if batch_group_counts:
                def locality_score(gpu: int) -> tuple[int, int, int, int]:
                    gpu = int(gpu)
                    shared_scene_count = sum(
                        int(count) for key, count in batch_group_counts.items() if gpu_group_counts[gpu].get(key, 0) > 0
                    )
                    primary_match = int(primary_group_key is not None and gpu_group_counts[gpu].get(primary_group_key, 0) > 0)
                    return (primary_match, shared_scene_count, -loads[gpu], -gpu)

                gpu_id = max(candidate_gpus, key=locality_score)
            else:
                gpu_id = min(candidate_gpus, key=lambda gpu: (loads[int(gpu)], int(gpu)))
            gpu_id = int(gpu_id)
            assignments[gpu_id].append(batch)
            loads[gpu_id] += int(batch["estimated_cost"])
            for group_key, count in batch_group_counts.items():
                gpu_group_counts[gpu_id][group_key] += int(count)
        return assignments

    def execute_batches(self, batches: Sequence[Dict], worker_label: str = "main") -> None:
        """Execute planned batches, reusing one solver per contiguous type group.

        Args:
            batches: Batch records assigned to this process.
            worker_label: Label shown in progress messages.

        Returns:
            None.
        """
        batches = list(batches)
        if not batches:
            return
        batches_by_type = defaultdict(list)
        for batch in batches:
            batches_by_type[(batch["type_name"], batch["manip_cfg_file"])].append(batch)
        total_grasps = sum(batch_grasp_count(batch) for batch in batches)
        worker_start_time = time.perf_counter()
        progress = ProgressReporter(
            total_grasps,
            desc=f"{worker_label} synthesis",
            unit="grasp",
            enabled=bool(self.args.progress),
        )
        progress_log_filter = None
        if self.args.progress:
            progress_log_filter = _install_synthesis_progress_noise_filter()
        try:
            for (type_name, manip_cfg_file), type_batches in batches_by_type.items():
                self.execute_type_batches(type_name, manip_cfg_file, type_batches, progress)
        finally:
            elapsed_seconds = time.perf_counter() - worker_start_time
            grasps_per_second = float(total_grasps) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
            _remove_synthesis_progress_noise_filter(progress_log_filter)
            progress.close()
            log_warn(
                f"Synthesis summary worker={worker_label} total_time={elapsed_seconds:.3f}s "
                f"total_grasps={total_grasps} avg_grasps_per_sec={grasps_per_second:.3f}"
            )
            self.profiler.report(worker_label)

    def execute_type_batches(
        self,
        type_name: str,
        manip_cfg_file: str,
        batches: Sequence[Dict],
        progress: ProgressReporter,
    ) -> None:
        """Execute all batches for one grasp type on the current process.

        Args:
            type_name: Grasp type name.
            manip_cfg_file: Manipulation config path.
            batches: Batch records for this type.
            progress: Worker-level progress reporter counting requested grasps.

        Returns:
            None.
        """
        manip_config_data = self.load_manip_config(manip_cfg_file)
        suite_config = self.load_suite_config()
        save_helper = self.make_save_helper(manip_config_data, batches[0]["save_folder"])
        seed_builder = None
        if self.args.init_source == "human":
            type_id, resolved_type_name, type_config = grasp_type_from_suite_config(manip_cfg_file, suite_config)
            type_name = resolved_type_name
            seed_builder = HumanPriorSeedBuilder(
                manip_config_data,
                suite_config,
                type_id=type_id,
                type_name=type_name,
                type_config=type_config,
            )

        grasp_solver = None
        for batch in batches:
            completed_grasps = batch_grasp_count(batch)
            batch_index = batch.get("global_batch_index", batch["local_batch_index"])
            progress.set_postfix(
                type=type_name,
                batch=batch_index,
                K_t=batch["type_budget"],
                scenes=batch["scene_count"],
                grasps=completed_grasps,
                status="solving",
            )
            if self.args.init_source == "surface_sample":
                batch_solver = None if self.args.solver_reuse == SOLVER_REUSE_BATCH else grasp_solver
                grasp_solver = self.run_surface_batch(batch_solver, batch, manip_config_data, save_helper)
            else:
                batch_solver = None if self.args.solver_reuse == SOLVER_REUSE_BATCH else grasp_solver
                grasp_solver = self.run_human_batch(batch_solver, batch, manip_config_data, suite_config, seed_builder, save_helper)
            progress.update(
                completed_grasps,
                type=type_name,
                batch=batch_index,
                K_t=batch["type_budget"],
                scenes=batch["scene_count"],
                grasps=completed_grasps,
                status="saved",
            )

    def run_surface_batch(self, grasp_solver, batch: Dict, manip_config_data: Dict, save_helper):
        """Run one surface-sample solver batch.

        Args:
            grasp_solver: Existing solver, or ``None`` for the first batch.
            batch: Executable batch record.
            manip_config_data: Loaded manipulation config dictionary.
            save_helper: Save helper or ``None``.

        Returns:
            Updated grasp solver.
        """
        with self.profiler.measure("surface.build_world_info"):
            world_info_dict = build_world_info_batch(batch["scene_paths"])
        with self.profiler.measure("surface.create_or_update_solver"):
            grasp_solver = self.create_or_update_solver(grasp_solver, world_info_dict, manip_config_data)
        self.attach_common_metadata(world_info_dict)
        with self.profiler.measure("surface.solve_batch_env"):
            result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)
        with self.profiler.measure("surface.attach_result"):
            self.attach_result_to_world_info(world_info_dict, result, grasp_solver, manip_config_data)
        if save_helper is not None:
            with self.profiler.measure("surface.save_piece"):
                save_helper.save_piece(world_info_dict)
        return grasp_solver

    def run_human_batch(self, grasp_solver, batch: Dict, manip_config_data: Dict, suite_config: Dict, seed_builder, save_helper):
        """Run one same-budget human-prior solver batch.

        Args:
            grasp_solver: Existing solver, or ``None`` for the first batch.
            batch: Executable batch record.
            manip_config_data: Loaded manipulation config dictionary.
            suite_config: Loaded suite config dictionary.
            seed_builder: Human-prior seed builder for the current type.
            save_helper: Save helper or ``None``.

        Returns:
            Updated grasp solver.
        """
        total_budget = int(suite_config["human_prior"].get("total_budget", 40))
        with self.profiler.measure("human.make_scene_jobs"):
            jobs = [
                seed_builder.make_scene_job(scene_path, suite_config["human_prior"]["root"], total_budget)
                for scene_path in batch["scene_paths"]
            ]
        with self.profiler.measure("human.build_world_info"):
            world_info_dict = build_world_info_batch(batch["scene_paths"])
        with self.profiler.measure("human.create_or_update_solver"):
            grasp_solver = self.create_or_update_solver(grasp_solver, world_info_dict, manip_config_data)
        with self.profiler.measure("human.build_seed_tensor"):
            seed_config, metadata = seed_builder.build_seed_tensor(
                jobs,
                device=grasp_solver.tensor_args.device,
                dtype=grasp_solver.tensor_args.dtype,
                expected_dof=grasp_solver.dof,
                seed_generator=grasp_solver.q_sample_gen,
                init_source=self.args.init_source,
            )
        solver_num_seeds = int(batch["type_budget"])
        metadata["human_prior_solver_num_seeds"] = np.asarray([solver_num_seeds for _ in jobs], dtype=np.int64)
        self.attach_common_metadata(world_info_dict)
        for key, value in metadata.items():
            world_info_dict[key] = value
        with self.profiler.measure("human.solve_batch_env"):
            result = grasp_solver.solve_batch_env(
                return_seeds=solver_num_seeds,
                num_seeds=solver_num_seeds,
                seed_config=seed_config,
                use_nn_seed=False,
            )
        with self.profiler.measure("human.attach_result"):
            self.attach_result_to_world_info(world_info_dict, result, grasp_solver, manip_config_data)
        if save_helper is not None:
            with self.profiler.measure("human.save_piece"):
                save_helper.save_piece(world_info_dict)
        return grasp_solver

    def create_or_update_solver(self, grasp_solver, world_info_dict: Dict, manip_config_data: Dict):
        """Create a new solver or update an existing solver's world batch.

        Args:
            grasp_solver: Existing solver, or ``None`` for the first batch.
            world_info_dict: Batched world-info dictionary.
            manip_config_data: Loaded manipulation config dictionary.

        Returns:
            Ready-to-run ``GraspSolver``.
        """
        from curobo.geom.sdf.world import WorldConfig
        from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict["world_cfg"],
                manip_name_list=world_info_dict["manip_name"],
                manip_config_data=manip_config_data,
                obj_gravity_center=world_info_dict["obj_gravity_center"],
                obj_obb_length=world_info_dict["obj_obb_length"],
                use_cuda_graph=False,
                store_debug=bool(self.args.save_debug),
                pregrasp_stage=manip_config_data["grasp_contact_strategy"]["pregrasp_stage"],
                grasp_stage=manip_config_data["grasp_contact_strategy"]["grasp_stage"],
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
            return grasp_solver
        world_model = [WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]]
        world_info_dict["world_model"] = world_model
        grasp_solver.update_world(
            world_model,
            world_info_dict["obj_gravity_center"],
            world_info_dict["obj_obb_length"],
            world_info_dict["manip_name"],
        )
        return grasp_solver

    def attach_common_metadata(self, world_info_dict: Dict) -> None:
        """Attach initialization metadata shared by all init sources.

        Args:
            world_info_dict: Batched world-info dictionary to mutate.

        Returns:
            None.
        """
        world_info_dict["init_source"] = [self.args.init_source for _ in world_info_dict["save_prefix"]]

    def attach_result_to_world_info(self, world_info_dict: Dict, result, grasp_solver, manip_config_data: Dict) -> None:
        """Attach solver outputs to a world-info dictionary before saving.

        Args:
            world_info_dict: Batched world-info dictionary to mutate.
            result: Solver result returned by ``GraspSolver``.
            grasp_solver: Active solver.
            manip_config_data: Loaded manipulation config dictionary.

        Returns:
            None.
        """
        if self.args.save_debug:
            robot_pose, debug_info = self.process_grasp_result(result, manip_config_data)
            world_info_dict["debug_info"] = debug_info
            world_info_dict["robot_pose"] = robot_pose.reshape(
                (len(world_info_dict["world_model"]), -1) + robot_pose.shape[1:]
            )
            return
        if grasp_solver.rollout_fn.kinematics.use_root_pose:
            squeeze_pose_qpos = torch.cat(
                [
                    result.solution[..., 1, :7],
                    result.solution[..., 1, 7:] * 2 - result.solution[..., 0, 7:],
                ],
                dim=-1,
            )
        else:
            squeeze_pose_qpos = result.solution[..., 1, :] * 2 - result.solution[..., 0, :]
        world_info_dict["robot_pose"] = torch.cat([result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2)
        world_info_dict["contact_point"] = result.contact_point
        world_info_dict["contact_frame"] = result.contact_frame
        world_info_dict["contact_force"] = result.contact_force
        world_info_dict["grasp_error"] = result.grasp_error
        world_info_dict["dist_error"] = result.dist_error

    def process_grasp_result(self, result, manip_config_data: Dict):
        """Select debug trajectory slices in the same format as the legacy CLI.

        Args:
            result: Solver result with debug trajectories.
            manip_config_data: Loaded manipulation config dictionary.

        Returns:
            Tuple of selected robot poses and optional debug tensors.
        """
        traj = result.debug_info["solver"]["steps"][0]
        all_traj = torch.cat(traj, dim=1)
        batch, horizon = all_traj.shape[:2]
        save_data = self.args.save_data
        if save_data == "all":
            select_horizon_lst = list(range(0, horizon))
        elif "select_" in save_data:
            part_num = int(save_data.split("select_")[-1])
            select_horizon_lst = list(range(0, horizon, horizon // (part_num - 1)))
            select_horizon_lst[-1] = horizon - 1
        elif save_data == "init":
            select_horizon_lst = [0]
        elif save_data in ("final", "final_and_mid"):
            select_horizon_lst = [-1]
        elif save_data == "pregrasp_and_grasp":
            contact_stages = torch.stack(result.debug_info["solver"]["contact_stage"][0], dim=-1)[0]
            pregrasp_stage = manip_config_data["grasp_contact_strategy"]["pregrasp_stage"]
            grasp_stage = manip_config_data["grasp_contact_strategy"]["grasp_stage"]
            pregrasp_step = torch.where(contact_stages == pregrasp_stage)[0][-1]
            grasp_step = torch.where(contact_stages == grasp_stage)[0][-1]
            select_horizon_lst = torch.stack([pregrasp_step, grasp_step], dim=0)
        else:
            raise NotImplementedError(f"Unsupported save_data={save_data}")

        save_id_lst = list(range(0, batch)) if self.args.save_id is None else list(self.args.save_id)
        save_traj = all_traj[:, select_horizon_lst][save_id_lst, :]
        if not self.args.save_debug:
            if save_data == "final_and_mid":
                mid_robot_pose = torch.cat(result.debug_info["solver"]["mid_result"][0], dim=1)
                save_traj = torch.cat([mid_robot_pose[save_id_lst, :], save_traj], dim=-2)
            return save_traj, None

        n_num = torch.stack(result.debug_info["solver"]["hp"][0]).shape[-2]
        o_num = torch.stack(result.debug_info["solver"]["op"][0]).shape[-2]
        debug_info = {
            "hp": torch.stack(result.debug_info["solver"]["hp"][0], dim=1).view(-1, n_num, 3),
            "grad": torch.stack(result.debug_info["solver"]["grad"][0], dim=1).view(-1, n_num, 3) * 100,
            "op": torch.stack(result.debug_info["solver"]["op"][0], dim=1).view(-1, o_num, 3),
            "debug_posi": torch.stack(result.debug_info["solver"]["debug_posi"][0], dim=1).view(-1, o_num, 3),
            "debug_normal": torch.stack(result.debug_info["solver"]["debug_normal"][0], dim=1).view(-1, o_num, 3),
            "contact_stage": torch.stack(result.debug_info["solver"]["contact_stage"][0], dim=-1).view(-1),
        }
        for key, value in debug_info.items():
            debug_info[key] = value.view((all_traj.shape[0], -1) + value.shape[1:])[:, select_horizon_lst]
            debug_info[key] = debug_info[key][save_id_lst, :].view((-1,) + value.shape[1:])
        return save_traj, debug_info

    def load_manip_config(self, manip_cfg_file: str) -> Dict:
        """Load one manipulation config and apply Hydra overrides.

        Args:
            manip_cfg_file: Manipulation config path relative to ``configs/manip``.

        Returns:
            Manipulation config dictionary.
        """
        manip_config_data = load_yaml(join_path(get_manip_configs_path(), manip_cfg_file))
        _deep_update(manip_config_data, self.baseline_manip_overrides(manip_cfg_file))
        if self.args.exp_name is not None:
            manip_config_data["exp_name"] = self.args.exp_name
        if self.args.start is not None:
            manip_config_data["world"]["start"] = self.args.start
        if self.args.end is not None:
            manip_config_data["world"]["end"] = self.args.end
        if self.args.obj_sample_hull_cache_mode is not None:
            manip_config_data.setdefault("seeder_cfg", {}).setdefault("obj_sample", {})[
                "hull_cache_mode"
            ] = self.args.obj_sample_hull_cache_mode
        if self.args.contact_obb_cache_mode is not None:
            manip_config_data.setdefault("seeder_cfg", {}).setdefault("obj_sample", {})[
                "contact_obb_cache_mode"
            ] = self.args.contact_obb_cache_mode
        if self.args.contact_surface_points_mode is not None:
            manip_config_data.setdefault("seeder_cfg", {}).setdefault("obj_sample", {})[
                "contact_surface_points_mode"
            ] = self.args.contact_surface_points_mode
        if self.args.contact_surface_sample_num is not None:
            manip_config_data.setdefault("seeder_cfg", {}).setdefault("obj_sample", {})[
                "contact_surface_sample_num"
            ] = self.args.contact_surface_sample_num
        if self.args.warp_mesh_cache_max_entries is not None:
            manip_config_data["warp_mesh_cache_max_entries"] = max(
                0, int(self.args.warp_mesh_cache_max_entries)
            )
        if self.args.warp_mesh_cache_max_mb is not None:
            manip_config_data["warp_mesh_cache_max_bytes"] = (
                max(0, int(self.args.warp_mesh_cache_max_mb)) * 1024 * 1024
            )
        return manip_config_data

    def resolve_save_folder(self, manip_cfg_file: str, manip_config_data: Dict) -> str:
        """Resolve the output folder in the same style as the legacy CLI.

        Args:
            manip_cfg_file: Manipulation config path.
            manip_config_data: Loaded manipulation config dictionary.

        Returns:
            Save folder path relative to cuRobo's output root.
        """
        if self.args.save_folder is not None:
            return os.path.join(self.args.save_folder, "graspdata")
        if manip_config_data["exp_name"] is not None:
            return os.path.join(manip_cfg_file[:-4], manip_config_data["exp_name"], "graspdata")
        return os.path.join(
            manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "graspdata",
        )

    def make_save_helper(self, manip_config_data: Dict, save_folder: str):
        """Construct a SaveHelper with human-prior metadata keys enabled.

        Args:
            manip_config_data: Loaded manipulation config dictionary.
            save_folder: Save folder path relative to cuRobo output root.

        Returns:
            Configured SaveHelper or ``None`` when save mode is disabled.
        """
        if self.args.save_mode == "none":
            return None
        from curobo.util.save_helper import SaveHelper

        return SaveHelper(
            robot_file=manip_config_data["robot_file"],
            save_folder=save_folder,
            task_name="grasp",
            mode=self.args.save_mode,
            npy_save_key=NPY_SAVE_KEYS,
        )

    def output_exists(self, save_folder: str, save_prefix: str) -> bool:
        """Check whether all requested output files already exist.

        Args:
            save_folder: Save folder relative to cuRobo output root.
            save_prefix: Scene save prefix such as ``<scene_id>_``.

        Returns:
            True when skip mode can skip this scene.
        """
        if self.args.save_mode == "none":
            return False
        full_folder = os.path.join(get_output_path(), save_folder)
        if not os.path.exists(os.path.join(full_folder, save_prefix + "grasp.npy")):
            return False
        return True


def task_synthesis(cfg: DictConfig):
    """Run grasp synthesis from Hydra-managed parameters.

    Args:
        cfg: Full Hydra config from ``example_grasp/main.py``.

    Returns:
        Summary dictionary for the synthesis run.
    """
    setup_logger("warn")
    _suppress_trimesh_even_sample_warning()
    _suppress_warp_deprecation_warnings()
    args = _build_synthesis_args(cfg)
    return SynthesisRuntime(args).run()

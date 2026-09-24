from __future__ import annotations

# Standard Library
import argparse
import copy
import datetime
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Sequence

# Third Party
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# CuRobo
from curobo.util.human_prior_seed import (
    HumanPriorSceneJob,
    HumanPriorSeedBuilder,
    build_world_info_batch,
    chunk_sequence,
    collect_scene_paths,
    grasp_type_from_runtime_config,
    load_runtime_config,
    scene_id_from_scene_path,
)
from curobo.util.logger import log_warn, setup_logger
from curobo.util_file import get_manip_configs_path, join_path, load_yaml
from curobo.util_file import get_output_path

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


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

INIT_SOURCE_ALIASES = {
    "heuristic": "surface_sample",
    "surface_sample": "surface_sample",
    "human_prior": "human",
    "human": "human",
}


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
        self.desc = desc
        self.unit = unit
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
            n: Number of work units completed.
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


def progress_enabled(args: argparse.Namespace) -> bool:
    """Read the progress flag from an argparse-compatible namespace.

    Args:
        args: Parsed command-line or Hydra-generated arguments.

    Returns:
        True when progress output should be shown.
    """
    return bool(getattr(args, "progress", True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for per-type grasp synthesis.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default="sim_shadow/tabletop_three.yml",
        help="Manipulation config path relative to configs/manip.",
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help="Runtime config path relative to configs/manip, e.g. sim_shadow/config.yaml.",
    )
    parser.add_argument(
        "--init-source",
        choices=sorted(INIT_SOURCE_ALIASES.keys()),
        default="surface_sample",
        help="Seed initialization source. 'heuristic' and 'human_prior' are legacy aliases.",
    )
    parser.add_argument(
        "--grasp-suite",
        type=str,
        default=None,
        help="Run all selected grasp types from the runtime config, e.g. shadow.",
    )
    parser.add_argument(
        "--grasp-types",
        type=str,
        nargs="+",
        default=None,
        help="Suite grasp types to run. Use names, ids, or 'all'. Defaults to all runtime-config types.",
    )
    parser.add_argument(
        "--human-prior-root",
        type=str,
        default=None,
        help="Override human_prior.root from the runtime config.",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Override human_prior.total_budget from the runtime config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scene and human-prior seed information without constructing the solver.",
    )
    parser.add_argument(
        "--dry-run-count",
        type=int,
        default=5,
        help="Number of scene records to print during --dry-run.",
    )
    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="If None, use join_path(manip_cfg_file[:-4], $TIME) as save_folder.",
    )
    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["npy", "none"],
        default="npy",
        help="Method to save results.",
    )
    parser.add_argument(
        "-d",
        "--save_data",
        default="all",
        help="Which results to save.",
    )
    parser.add_argument(
        "-i",
        "--save_id",
        type=int,
        nargs="+",
        default=None,
        help="Which results to save.",
    )
    parser.add_argument(
        "-debug",
        "--save_debug",
        action="store_true",
        help="Save contact normal debug information.",
    )
    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="Parallel world count.",
    )
    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )
    parser.add_argument(
        "-p",
        "--exp_name",
        type=str,
        default=None,
        help="If None, use exp_name in manip_cfg_file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Override world.start from the manipulation config.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Override world.end from the manipulation config.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm-style progress bars for long synthesis runs.",
    )
    parser.set_defaults(progress=True)
    return parser.parse_args()


def canonical_init_source(init_source: str) -> str:
    """Normalize legacy initialization source names.

    Args:
        init_source: User-provided init source or legacy alias.

    Returns:
        Canonical init source: ``surface_sample`` or ``human``.
    """
    if init_source not in INIT_SOURCE_ALIASES:
        raise ValueError(f"Unsupported init_source={init_source}")
    return INIT_SOURCE_ALIASES[init_source]


def apply_cli_overrides(manip_config_data: Dict, args: argparse.Namespace) -> Dict:
    """Apply command-line overrides to the loaded manipulation config.

    Args:
        manip_config_data: Loaded manipulation config dictionary.
        args: Parsed command-line arguments.

    Returns:
        Updated manipulation config dictionary.
    """
    if args.exp_name is not None:
        manip_config_data["exp_name"] = args.exp_name
    if args.start is not None:
        manip_config_data["world"]["start"] = args.start
    if args.end is not None:
        manip_config_data["world"]["end"] = args.end
    return manip_config_data


def resolve_save_folder(args: argparse.Namespace, manip_config_data: Dict) -> str:
    """Resolve the output folder in the same style as ``plan_batch_env.py``.

    Args:
        args: Parsed command-line arguments.
        manip_config_data: Loaded manipulation config dictionary.

    Returns:
        Save folder path relative to cuRobo's output root.
    """
    if args.save_folder is not None:
        return os.path.join(args.save_folder, "graspdata")
    if manip_config_data["exp_name"] is not None:
        return os.path.join(args.manip_cfg_file[:-4], manip_config_data["exp_name"], "graspdata")
    return os.path.join(
        args.manip_cfg_file[:-4],
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        "graspdata",
    )


def load_manip_config(args: argparse.Namespace) -> Dict:
    """Load the selected manipulation config.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Manipulation config dictionary with CLI overrides applied.
    """
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))
    return apply_cli_overrides(manip_config_data, args)


def prepare_runtime(args: argparse.Namespace, manip_config_data: Dict) -> Dict:
    """Load runtime config and apply CLI overrides needed by human-prior mode.

    Args:
        args: Parsed command-line arguments.
        manip_config_data: Loaded manipulation config dictionary.

    Returns:
        Runtime config dictionary.
    """
    if args.runtime_config is None:
        if args.init_source == "human":
            raise ValueError(f"--runtime-config is required when --init-source {args.init_source}")
        return {}

    runtime_config = load_runtime_config(args.runtime_config)
    runtime_config.setdefault("human_prior", {})
    if args.human_prior_root is not None:
        runtime_config["human_prior"]["root"] = args.human_prior_root
    if args.total_budget is not None:
        runtime_config["human_prior"]["total_budget"] = args.total_budget
    if args.init_source == "human":
        grasp_type_from_runtime_config(args.manip_cfg_file, runtime_config)
        if not runtime_config["human_prior"].get("root"):
            raise ValueError(f"runtime_config.human_prior.root is required for {args.init_source} mode")
    return runtime_config


def make_save_helper(args: argparse.Namespace, manip_config_data: Dict, save_folder: str):
    """Construct a SaveHelper with extra human-prior metadata keys enabled.

    Args:
        args: Parsed command-line arguments.
        manip_config_data: Loaded manipulation config dictionary.
        save_folder: Save folder path relative to cuRobo's output root.

    Returns:
        Configured ``SaveHelper`` instance.
    """
    if args.save_mode == "none":
        return None
    from curobo.util.save_helper import SaveHelper

    return SaveHelper(
        robot_file=manip_config_data["robot_file"],
        save_folder=save_folder,
        task_name="grasp",
        mode=args.save_mode,
        npy_save_key=NPY_SAVE_KEYS,
    )


def create_or_update_solver(
    grasp_solver: Optional[object],
    world_info_dict: Dict,
    manip_config_data: Dict,
    save_debug: bool,
):
    """Create a new solver or update the existing solver's world batch.

    Args:
        grasp_solver: Existing solver, or ``None`` for the first batch.
        world_info_dict: Batched world-info dictionary.
        manip_config_data: Loaded manipulation config dictionary.
        save_debug: Whether debug data will be stored.

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
            store_debug=save_debug,
            pregrasp_stage=manip_config_data["grasp_contact_strategy"]["pregrasp_stage"],
            grasp_stage=manip_config_data["grasp_contact_strategy"]["grasp_stage"],
        )
        grasp_solver = GraspSolver(grasp_config)
        world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
    else:
        world_model = [WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]]
        world_info_dict["world_model"] = world_model
        grasp_solver.update_world(
            world_model,
            world_info_dict["obj_gravity_center"],
            world_info_dict["obj_obb_length"],
            world_info_dict["manip_name"],
        )
    return grasp_solver


def attach_common_metadata(world_info_dict: Dict, init_source: str) -> None:
    """Attach initialization metadata shared by all init sources.

    Args:
        world_info_dict: Batched world-info dictionary to mutate.
        init_source: Initialization source name.

    Returns:
        None.
    """
    batch_size = len(world_info_dict["save_prefix"])
    world_info_dict["init_source"] = [init_source for _ in range(batch_size)]


def attach_result_to_world_info(
    world_info_dict: Dict,
    result,
    grasp_solver,
    args: argparse.Namespace,
    manip_config_data: Dict,
) -> None:
    """Attach solver outputs to the batched world-info dictionary.

    Args:
        world_info_dict: Batched world-info dictionary to mutate.
        result: ``GraspResult`` returned by the solver.
        grasp_solver: Active grasp solver.
        args: Parsed command-line arguments.
        manip_config_data: Loaded manipulation config dictionary.

    Returns:
        None.
    """
    if args.save_debug:
        from plan_batch_env import process_grasp_result

        robot_pose, debug_info = process_grasp_result(
            result, args.save_debug, args.save_data, args.save_id, manip_config_data
        )
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


def attach_human_prior_metadata(world_info_dict: Dict, metadata: Dict) -> None:
    """Attach human-prior metadata to the batched world-info dictionary.

    Args:
        world_info_dict: Batched world-info dictionary to mutate.
        metadata: Metadata returned by ``HumanPriorSeedBuilder.build_seed_tensor``.

    Returns:
        None.
    """
    for key, value in metadata.items():
        world_info_dict[key] = value


def run_solver_batch(
    grasp_solver: Optional[object],
    scene_paths: Sequence[str],
    manip_config_data: Dict,
    save_helper,
    args: argparse.Namespace,
    seed_config: Optional[torch.Tensor] = None,
    human_prior_metadata: Optional[Dict] = None,
    num_seeds: Optional[int] = None,
):
    """Run one solver batch and save the result.

    Args:
        grasp_solver: Existing solver, or ``None`` for the first batch.
        scene_paths: Scene config paths for this batch.
        manip_config_data: Loaded manipulation config dictionary.
        save_helper: Save helper used for output.
        args: Parsed command-line arguments.
        seed_config: Optional human-prior seed tensor.
        human_prior_metadata: Optional metadata aligned with ``seed_config``.
        num_seeds: Optional per-scene seed count override.

    Returns:
        Updated ``GraspSolver`` instance.
    """
    sst = time.time()
    world_info_dict = build_world_info_batch(scene_paths)
    grasp_solver = create_or_update_solver(grasp_solver, world_info_dict, manip_config_data, args.save_debug)
    attach_common_metadata(world_info_dict, args.init_source)
    if human_prior_metadata is not None:
        attach_human_prior_metadata(world_info_dict, human_prior_metadata)

    if seed_config is None:
        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)
    else:
        if num_seeds is None:
            num_seeds = int(seed_config.shape[1])
        effective_num_seeds = int(num_seeds)
        if effective_num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive for human-prior synthesis, got {effective_num_seeds}")
        result = grasp_solver.solve_batch_env(
            return_seeds=effective_num_seeds,
            num_seeds=effective_num_seeds,
            seed_config=seed_config,
            use_nn_seed=False,
        )

    attach_result_to_world_info(world_info_dict, result, grasp_solver, args, manip_config_data)
    log_warn(f"Sinlge Time: {time.time() - sst}")
    if save_helper is not None:
        save_helper.save_piece(world_info_dict)
    return grasp_solver


def run_heuristic(
    scene_paths: Sequence[str],
    manip_config_data: Dict,
    save_helper,
    args: argparse.Namespace,
) -> None:
    """Run the existing heuristic seed path over deterministic scene batches.

    Args:
        scene_paths: Ordered scene config paths.
        manip_config_data: Loaded manipulation config dictionary.
        save_helper: Save helper used for output.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    grasp_solver = None
    progress = ProgressReporter(
        len(scene_paths),
        desc=f"heuristic {args.manip_cfg_file}",
        unit="scene",
        enabled=progress_enabled(args),
    )
    try:
        for batch_idx, scene_batch in enumerate(chunk_sequence(scene_paths, args.parallel_world), start=1):
            world_info_dict = build_world_info_batch(scene_batch)
            if save_helper is not None and args.skip and save_helper.exist_piece(world_info_dict["save_prefix"]):
                log_warn(f"skip {world_info_dict['save_prefix']}")
                progress.update(len(scene_batch), batch=batch_idx, status="skipped")
                continue
            progress.set_postfix(batch=batch_idx, batch_size=len(scene_batch), status="solving")
            grasp_solver = run_solver_batch(grasp_solver, scene_batch, manip_config_data, save_helper, args)
            progress.update(len(scene_batch), batch=batch_idx, batch_size=len(scene_batch), status="saved")
    finally:
        progress.close()


def run_human_prior(
    scene_paths: Sequence[str],
    manip_config_data: Dict,
    runtime_config: Dict,
    save_helper,
    args: argparse.Namespace,
) -> None:
    """Run human-prior seed synthesis grouped by current-type budget.

    Args:
        scene_paths: Ordered scene config paths.
        manip_config_data: Loaded manipulation config dictionary.
        runtime_config: Loaded runtime config dictionary.
        save_helper: Save helper used for output.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    type_id, type_name, type_config = grasp_type_from_runtime_config(args.manip_cfg_file, runtime_config)
    human_prior_root = runtime_config["human_prior"]["root"]
    total_budget = int(runtime_config["human_prior"].get("total_budget", 40))
    seed_builder = HumanPriorSeedBuilder(
        manip_config_data,
        runtime_config,
        type_id=type_id,
        type_name=type_name,
        type_config=type_config,
    )
    target_batch_grasps = runtime_config["human_prior"].get("target_batch_grasps")
    if target_batch_grasps is not None:
        target_batch_grasps = int(target_batch_grasps)
        if target_batch_grasps <= 0:
            raise ValueError(f"human_prior.target_batch_grasps must be positive, got {target_batch_grasps}")
    max_parallel_world = runtime_config["human_prior"].get("max_parallel_world")
    if max_parallel_world is not None:
        max_parallel_world = int(max_parallel_world)
        if max_parallel_world <= 0:
            raise ValueError(f"human_prior.max_parallel_world must be positive, got {max_parallel_world}")

    pending_jobs = defaultdict(list)
    budget_summary = defaultdict(int)
    grasp_solver = None
    prepare_progress = ProgressReporter(
        len(scene_paths),
        desc=f"{type_name} prepare",
        unit="scene",
        enabled=progress_enabled(args),
        leave=False,
    )
    solve_progress = ProgressReporter(
        len(scene_paths),
        desc=f"{type_name} solve",
        unit="scene",
        enabled=progress_enabled(args),
    )
    batch_idx = 0
    try:
        for scene_path in scene_paths:
            scene_id = None
            try:
                scene_id = scene_id_from_scene_path(scene_path)
                if save_helper is not None and args.skip and save_helper.exist_piece([f"{scene_id}_"]):
                    log_warn(f"skip {scene_id}_")
                    prepare_progress.update(1, status="skipped")
                    solve_progress.update(1, status="skipped")
                    continue
                job = seed_builder.make_scene_job(scene_path, human_prior_root, total_budget)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to prepare human prior for scene_path={scene_path}, scene_id={scene_id}"
                ) from exc

            if job.type_budget == 0:
                log_warn(f"skip {job.scene_id}_ because {type_name} budget is 0")
                prepare_progress.update(1, K_t=0, status="zero_budget")
                solve_progress.update(1, K_t=0, status="zero_budget")
                continue

            budget_summary[job.type_budget] += 1
            pending_jobs[job.type_budget].append(job)
            prepare_progress.update(1, K_t=job.type_budget, status="queued")
            adaptive_parallel_world = compute_adaptive_parallel_world(
                job.type_budget,
                target_batch_grasps,
                args.parallel_world,
                max_parallel_world=max_parallel_world,
            )
            if len(pending_jobs[job.type_budget]) >= adaptive_parallel_world:
                ready_jobs = pending_jobs[job.type_budget][:adaptive_parallel_world]
                pending_jobs[job.type_budget] = pending_jobs[job.type_budget][adaptive_parallel_world:]
                batch_idx += 1
                log_human_prior_budget_batch(type_name, job.type_budget, len(ready_jobs), adaptive_parallel_world)
                solve_progress.set_postfix(
                    batch=batch_idx,
                    K_t=job.type_budget,
                    batch_size=len(ready_jobs),
                    status="solving",
                )
                grasp_solver = run_human_prior_job_batch(
                    grasp_solver, ready_jobs, seed_builder, manip_config_data, save_helper, args
                )
                solve_progress.update(
                    len(ready_jobs),
                    batch=batch_idx,
                    K_t=job.type_budget,
                    batch_size=len(ready_jobs),
                    status="saved",
                )

        log_human_prior_budget_summary(type_name, budget_summary, label="Run")
        for type_budget in sorted(pending_jobs):
            jobs = pending_jobs[type_budget]
            if jobs:
                adaptive_parallel_world = compute_adaptive_parallel_world(
                    type_budget,
                    target_batch_grasps,
                    args.parallel_world,
                    max_parallel_world=max_parallel_world,
                )
                batch_idx += 1
                log_human_prior_budget_batch(type_name, type_budget, len(jobs), adaptive_parallel_world)
                solve_progress.set_postfix(
                    batch=batch_idx,
                    K_t=type_budget,
                    batch_size=len(jobs),
                    status="solving",
                )
                grasp_solver = run_human_prior_job_batch(
                    grasp_solver, jobs, seed_builder, manip_config_data, save_helper, args
                )
                solve_progress.update(
                    len(jobs),
                    batch=batch_idx,
                    K_t=type_budget,
                    batch_size=len(jobs),
                    status="saved",
                )
    finally:
        prepare_progress.close()
        solve_progress.close()


def log_human_prior_budget_batch(
    type_name: str,
    type_budget: int,
    scene_count: int,
    adaptive_parallel_world: int,
) -> None:
    """Log the budget group used by one human-prior solver batch.

    Args:
        type_name: Current human-prior grasp type name.
        type_budget: Per-scene seed budget for this batch.
        scene_count: Number of scenes included in this solver batch.
        adaptive_parallel_world: Maximum scene count for this budget group.

    Returns:
        None.
    """
    log_warn(
        f"Run human-prior budget batch type={type_name} "
        f"type_budget={int(type_budget)} scene_count={int(scene_count)} "
        f"parallel_world={int(adaptive_parallel_world)}"
    )


def compute_adaptive_parallel_world(
    type_budget: int,
    target_batch_grasps: Optional[int],
    fallback_parallel_world: int,
    max_parallel_world: Optional[int] = None,
) -> int:
    """Compute the scene batch size for a fixed per-scene human-prior budget.

    Args:
        type_budget: Number of human-prior seeds used per scene.
        target_batch_grasps: Optional target for ``scene_count * type_budget``.
        fallback_parallel_world: Static scene count used when no target is configured.
        max_parallel_world: Optional hard cap for the adaptive scene count.

    Returns:
        Positive scene count used for this budget group.
    """
    if type_budget <= 0:
        raise ValueError(f"type_budget must be positive, got {type_budget}")
    if fallback_parallel_world <= 0:
        raise ValueError(f"fallback_parallel_world must be positive, got {fallback_parallel_world}")
    if target_batch_grasps is None:
        return int(fallback_parallel_world)
    if target_batch_grasps <= 0:
        raise ValueError(f"target_batch_grasps must be positive, got {target_batch_grasps}")
    adaptive_parallel_world = max(1, int(target_batch_grasps) // int(type_budget))
    if max_parallel_world is not None:
        if max_parallel_world <= 0:
            raise ValueError(f"max_parallel_world must be positive, got {max_parallel_world}")
        adaptive_parallel_world = min(adaptive_parallel_world, int(max_parallel_world))
    return max(1, int(adaptive_parallel_world))


def run_human_prior_job_batch(
    grasp_solver: Optional[object],
    jobs: Sequence[HumanPriorSceneJob],
    seed_builder: HumanPriorSeedBuilder,
    manip_config_data: Dict,
    save_helper,
    args: argparse.Namespace,
):
    """Build seed tensors for same-budget jobs and run one solver batch.

    Args:
        grasp_solver: Existing solver, or ``None`` for the first batch.
        jobs: Same-budget human-prior jobs.
        seed_builder: Human-prior seed builder.
        manip_config_data: Loaded manipulation config dictionary.
        save_helper: Save helper used for output.
        args: Parsed command-line arguments.

    Returns:
        Updated ``GraspSolver`` instance.
    """
    scene_paths = [job.scene_path for job in jobs]
    world_info_dict = build_world_info_batch(scene_paths)
    # Update the solver world before building human-surface seeds so q_sample_gen
    # exposes surface candidates for the current batch instead of the previous one.
    grasp_solver = create_or_update_solver(grasp_solver, world_info_dict, manip_config_data, args.save_debug)
    seed_config, metadata = seed_builder.build_seed_tensor(
        jobs,
        device=grasp_solver.tensor_args.device,
        dtype=grasp_solver.tensor_args.dtype,
        expected_dof=grasp_solver.dof,
        seed_generator=grasp_solver.q_sample_gen,
        init_source=args.init_source,
    )
    solver_num_seeds = int(jobs[0].type_budget)
    if solver_num_seeds <= 0:
        raise ValueError(f"type_budget must be positive for human-prior synthesis, got {solver_num_seeds}")
    metadata["human_prior_solver_num_seeds"] = np.asarray([solver_num_seeds for _ in jobs], dtype=np.int64)
    attach_common_metadata(world_info_dict, args.init_source)
    attach_human_prior_metadata(world_info_dict, metadata)
    result = grasp_solver.solve_batch_env(
        return_seeds=solver_num_seeds,
        num_seeds=solver_num_seeds,
        seed_config=seed_config,
        use_nn_seed=False,
    )
    attach_result_to_world_info(world_info_dict, result, grasp_solver, args, manip_config_data)
    if save_helper is not None:
        save_helper.save_piece(world_info_dict)
    return grasp_solver


def print_dry_run(
    scene_paths: Sequence[str],
    manip_config_data: Dict,
    runtime_config: Dict,
    args: argparse.Namespace,
) -> None:
    """Print dry-run information for heuristic or human-prior mode.

    Args:
        scene_paths: Ordered scene config paths.
        manip_config_data: Loaded manipulation config dictionary.
        runtime_config: Loaded runtime config dictionary.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    print(f"Dry run scene count: {len(scene_paths)}")
    print(f"Init source: {args.init_source}")
    if args.init_source == "surface_sample":
        for scene_path in scene_paths[: args.dry_run_count]:
            print(f"scene_path={scene_path}")
        return

    type_id, type_name, type_config = grasp_type_from_runtime_config(args.manip_cfg_file, runtime_config)
    total_budget = int(runtime_config["human_prior"].get("total_budget", 40))
    seed_builder = HumanPriorSeedBuilder(
        manip_config_data,
        runtime_config,
        type_id=type_id,
        type_name=type_name,
        type_config=type_config,
    )
    budget_summary = defaultdict(int)
    for scene_path in scene_paths[: args.dry_run_count]:
        job = seed_builder.make_scene_job(scene_path, runtime_config["human_prior"]["root"], total_budget)
        budget_summary[job.type_budget] += 1
        print(
            "scene_id={scene_id} type={type_name} scores={scores} budgets={budgets} "
            "K_t={budget} sample_indices={indices} replacement_mask={replacement}".format(
                scene_id=job.scene_id,
                type_name=type_name,
                scores=np.array2string(job.budget_scores, precision=4),
                budgets=job.type_budgets.tolist(),
                budget=job.type_budget,
                indices=job.sample_indices.tolist(),
                replacement=job.replacement_mask.astype(int).tolist(),
            )
        )
        if job.type_budget > 0:
            if args.init_source == "human" and seed_builder.robot_layout == "root_pose":
                seed_tensor, _ = seed_builder.build_seed_tensor(
                    [job],
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    expected_dof=None,
                    init_source=args.init_source,
                )
                print(f"seed_tensor_shape={tuple(seed_tensor.shape)}")
            else:
                print(f"seed_tensor_shape=(1, {job.type_budget}, solver_dof) requires solver surface matching/projection")
    log_human_prior_budget_summary(type_name, budget_summary, label="Dry-run")


def log_human_prior_budget_summary(type_name: str, budget_summary: Dict[int, int], label: str) -> None:
    """Log a compact distribution of human-prior type budgets.

    Args:
        type_name: Current human-prior grasp type name.
        budget_summary: Mapping from per-scene type budget to scene count.
        label: Prefix describing where the summary was produced.

    Returns:
        None.
    """
    if not budget_summary:
        log_warn(f"{label} type budget summary type={type_name}: empty")
        return
    summary = ", ".join(
        f"K_t={int(type_budget)}: scenes={int(scene_count)}"
        for type_budget, scene_count in sorted(budget_summary.items())
    )
    log_warn(f"{label} type budget summary type={type_name}: {summary}")


def apply_runtime_overrides(args: argparse.Namespace, runtime_config: Dict) -> Dict:
    """Apply command-line overrides to a loaded runtime config.

    Args:
        args: Parsed command-line arguments.
        runtime_config: Loaded runtime config dictionary.

    Returns:
        Runtime config with CLI overrides applied.
    """
    runtime_config = copy.deepcopy(runtime_config)
    runtime_config.setdefault("human_prior", {})
    if args.human_prior_root is not None:
        runtime_config["human_prior"]["root"] = args.human_prior_root
    if args.total_budget is not None:
        runtime_config["human_prior"]["total_budget"] = args.total_budget
    return runtime_config


def resolve_suite_type_names(args: argparse.Namespace, runtime_config: Dict) -> Sequence[str]:
    """Resolve suite type names requested by ``--grasp-types``.

    Args:
        args: Parsed command-line arguments.
        runtime_config: Loaded runtime config dictionary.

    Returns:
        Ordered list of grasp type names to execute.
    """
    grasp_types = runtime_config.get("grasp_types", {})
    if not grasp_types:
        raise ValueError("runtime_config.grasp_types is required for --grasp-suite")
    requested = args.grasp_types
    if requested is None or "all" in requested:
        return list(grasp_types.keys())

    selected = []
    for token in requested:
        matched_name = None
        for type_name, type_config in grasp_types.items():
            type_id = type_config.get("type_id") if isinstance(type_config, dict) else str(type_name).split("_", 1)[0]
            if token == type_name or token == str(type_id):
                matched_name = type_name
                break
        if matched_name is None:
            raise ValueError(f"Unknown grasp type {token}; available types: {list(grasp_types.keys())}")
        selected.append(matched_name)
    return selected


def manip_cfg_file_from_type_config(type_name: str, type_config) -> str:
    """Read a manipulation config path from one runtime grasp type entry.

    Args:
        type_name: Runtime grasp type name for error messages.
        type_config: Runtime grasp type config entry.

    Returns:
        Manipulation config path relative to ``configs/manip``.
    """
    if isinstance(type_config, str):
        return type_config
    if "manip_cfg_file" not in type_config:
        raise KeyError(f"runtime_config.grasp_types.{type_name}.manip_cfg_file is required")
    return str(type_config["manip_cfg_file"])


def resolve_suite_manifest_path(args: argparse.Namespace) -> str:
    """Resolve the suite manifest path.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Absolute path for the JSON suite manifest.
    """
    if args.save_folder is not None:
        return os.path.join(get_output_path(), args.save_folder, "suite_manifest.json")
    exp_label = args.exp_name or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    suite_name = args.grasp_suite or "suite"
    return os.path.join(get_output_path(), "suite_manifests", suite_name, exp_label, "suite_manifest.json")


def write_suite_manifest(path: str, manifest: Dict) -> None:
    """Write a suite manifest JSON file.

    Args:
        path: Absolute manifest path.
        manifest: JSON-serializable manifest dictionary.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def run_single_type(args: argparse.Namespace) -> Dict:
    """Run one manipulation config in heuristic or human-prior mode.

    Args:
        args: Parsed command-line arguments for one grasp type.

    Returns:
        Summary dictionary for suite manifests.
    """
    if args.parallel_world <= 0:
        raise ValueError(f"--parallel_world must be positive, got {args.parallel_world}")

    manip_config_data = load_manip_config(args)
    runtime_config = prepare_runtime(args, manip_config_data)
    scene_source_config = runtime_config.get("scene_source", {"use_object_scale_list": True})
    scene_paths = collect_scene_paths(manip_config_data["world"], scene_source_config)
    log_warn(f"Collected {len(scene_paths)} deterministic scene cfgs.")

    if args.dry_run:
        print_dry_run(scene_paths, manip_config_data, runtime_config, args)
        return {
            "manip_cfg_file": args.manip_cfg_file,
            "robot_file": manip_config_data["robot_file"],
            "scene_count": len(scene_paths),
            "dry_run": True,
        }

    save_folder = resolve_save_folder(args, manip_config_data)
    save_helper = make_save_helper(args, manip_config_data, save_folder)
    tst = time.time()
    if args.init_source == "surface_sample":
        run_heuristic(scene_paths, manip_config_data, save_helper, args)
    elif args.init_source == "human":
        run_human_prior(scene_paths, manip_config_data, runtime_config, save_helper, args)
    else:
        raise NotImplementedError(f"Unsupported init_source={args.init_source}")
    elapsed = time.time() - tst
    log_warn(f"Total Time: {elapsed}")
    return {
        "manip_cfg_file": args.manip_cfg_file,
        "robot_file": manip_config_data["robot_file"],
        "save_folder": save_folder,
        "scene_count": len(scene_paths),
        "elapsed_sec": elapsed,
        "dry_run": False,
    }


def run_suite(args: argparse.Namespace) -> None:
    """Run multiple grasp types from one runtime config.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    if args.runtime_config is None:
        raise ValueError("--runtime-config is required when --grasp-suite is used")
    runtime_config = apply_runtime_overrides(args, load_runtime_config(args.runtime_config))
    if args.init_source == "human" and not runtime_config.get("human_prior", {}).get("root"):
        raise ValueError(f"runtime_config.human_prior.root is required for {args.init_source} suite mode")

    selected_type_names = resolve_suite_type_names(args, runtime_config)
    manifest_path = resolve_suite_manifest_path(args)
    write_manifest = (not args.dry_run) and args.save_mode != "none"
    manifest = {
        "suite": args.grasp_suite,
        "runtime_config": args.runtime_config,
        "init_source": args.init_source,
        "start": args.start,
        "end": args.end,
        "exp_name": args.exp_name,
        "types": [],
    }

    suite_progress = ProgressReporter(
        len(selected_type_names),
        desc=f"suite {args.grasp_suite or 'types'}",
        unit="type",
        enabled=progress_enabled(args),
    )
    try:
        for type_name in selected_type_names:
            type_config = runtime_config["grasp_types"][type_name]
            type_args = copy.copy(args)
            type_args.manip_cfg_file = manip_cfg_file_from_type_config(type_name, type_config)
            log_warn(f"Run suite type {type_name}: {type_args.manip_cfg_file}")
            suite_progress.set_postfix(type=type_name, status="running")
            type_entry = {
                "type_name": type_name,
                "manip_cfg_file": type_args.manip_cfg_file,
                "status": "running",
            }
            manifest["types"].append(type_entry)
            if write_manifest:
                write_suite_manifest(manifest_path, manifest)
            try:
                type_entry.update(run_single_type(type_args))
                type_entry["status"] = "completed"
                suite_progress.update(1, type=type_name, status="completed")
            except Exception as exc:
                type_entry["status"] = "failed"
                type_entry["error"] = repr(exc)
                suite_progress.set_postfix(type=type_name, status="failed")
                if write_manifest:
                    write_suite_manifest(manifest_path, manifest)
                raise
            if write_manifest:
                write_suite_manifest(manifest_path, manifest)
    finally:
        suite_progress.close()

    if write_manifest:
        log_warn(f"Suite manifest saved to {manifest_path}")


def main() -> None:
    """Run batch grasp synthesis with heuristic or human-prior seeds.

    Args:
        None.

    Returns:
        None.
    """
    setup_logger("warn")
    args = parse_args()
    args.init_source = canonical_init_source(args.init_source)
    if args.grasp_suite is not None:
        run_suite(args)
    else:
        run_single_type(args)


if __name__ == "__main__":
    main()

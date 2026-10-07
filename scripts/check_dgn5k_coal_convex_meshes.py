#!/usr/bin/env python3
"""Check DGN_5k convex meshes with COAL's buildConvexHull path.

This script exercises the same ``coal_openmp_wrapper.loadConvexMeshCpp`` entry
point used by BimanBODex contact checking.  Each object is tested in a separate
Python subprocess so a hard COAL/Qhull crash cannot stop the parent scan.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET = "DGN_5k"
DEFAULT_ENV_NAME = "HUGS_DATASET_ROOT"
DEFAULT_SCENE_TYPE = "tabletop_ur10e"
DEFAULT_SCALE_DENOMINATOR = 100.0
DEFAULT_TIMEOUT_SEC = 120.0
WORKER_OBJECT_MODE = "_worker-object"
WORKER_PIECE_MODE = "_worker-piece"
SCALE_RE = re.compile(r"scale(\d+)")
ScaleTuple = Tuple[float, float, float]


@dataclass(frozen=True)
class ObjectJob:
    """One object-level COAL check request."""

    object_id: str
    scales: Tuple[ScaleTuple, ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for scan and hidden worker modes.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_OBJECT_MODE:
        return parse_worker_object_args(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_PIECE_MODE:
        return parse_worker_piece_args(sys.argv[2:])
    return parse_scan_args(sys.argv[1:])


def parse_scan_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse parent scan arguments.

    Args:
        argv: Command-line arguments after the executable name.

    Returns:
        argparse.Namespace: Parsed scan arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check DGN_5k COACD convex_piece meshes by loading them through "
            "coal_openmp_wrapper.loadConvexMeshCpp at scene-used object scales."
        )
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Path to the dataset object asset folder. Defaults to ${HUGS_DATASET_ROOT}/object/DGN_5k.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset folder under ${HUGS_DATASET_ROOT}/object when --asset-root is not set.",
    )
    parser.add_argument(
        "--dataset-root-env",
        default=DEFAULT_ENV_NAME,
        help="Environment variable that points to the HUGS dataset root.",
    )
    parser.add_argument(
        "--scene-type",
        default=DEFAULT_SCENE_TYPE,
        help="Scene type folder used to collect per-object scales.",
    )
    parser.add_argument(
        "--scale-source",
        choices=("filename", "scene_cfg", "manual"),
        default="filename",
        help=(
            "Where object scales come from. filename parses scaleXXX from scene names; "
            "scene_cfg loads every .npy scene config; manual uses --scales."
        ),
    )
    parser.add_argument(
        "--scale-denominator",
        type=float,
        default=DEFAULT_SCALE_DENOMINATOR,
        help="Denominator for filename scales, e.g. scale008 / 100 = 0.08.",
    )
    parser.add_argument(
        "--scales",
        nargs="*",
        default=None,
        help="Manual scalar scales such as 0.02 0.03 1.0, used when --scale-source manual.",
    )
    parser.add_argument(
        "--include-unit-scale",
        action="store_true",
        help="Also test scale 1.0 for every object.",
    )
    parser.add_argument(
        "--object-id",
        action="append",
        default=None,
        help="Object id to check. Can be passed multiple times.",
    )
    parser.add_argument(
        "--object-list",
        type=Path,
        default=None,
        help="Optional text file containing one object id per line.",
    )
    parser.add_argument(
        "--limit-objects",
        type=int,
        default=None,
        help="Optional maximum number of objects to check after sorting and filtering.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of object subprocesses to run concurrently.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help="Timeout for each object subprocess.",
    )
    parser.add_argument(
        "--diagnose-crashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If an object subprocess exits abnormally, retry piece/scale pairs one by one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports. Defaults to <asset-root>/coal_convex_mesh_check_<timestamp>.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N completed objects. Use 0 to disable periodic progress.",
    )
    parser.set_defaults(mode="scan")
    return parser.parse_args(argv)


def parse_worker_object_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse hidden object-worker arguments.

    Args:
        argv: Command-line arguments after the hidden worker mode.

    Returns:
        argparse.Namespace: Parsed worker arguments.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--scales-json", required=True)
    parser.set_defaults(mode=WORKER_OBJECT_MODE)
    return parser.parse_args(argv)


def parse_worker_piece_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse hidden single-piece worker arguments.

    Args:
        argv: Command-line arguments after the hidden piece-worker mode.

    Returns:
        argparse.Namespace: Parsed worker arguments.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--scale-json", required=True)
    parser.set_defaults(mode=WORKER_PIECE_MODE)
    return parser.parse_args(argv)


def resolve_asset_root(args: argparse.Namespace) -> Path:
    """Resolve the DGN object asset folder.

    Args:
        args: Parsed scan arguments.

    Returns:
        Path: Absolute dataset object asset folder.
    """
    if args.asset_root is not None:
        return args.asset_root.expanduser().resolve()
    dataset_root = os.environ.get(args.dataset_root_env)
    if not dataset_root:
        raise EnvironmentError(f"Please set {args.dataset_root_env} or pass --asset-root.")
    return (Path(dataset_root).expanduser().resolve() / "object" / args.dataset).resolve()


def sorted_object_ids(asset_root: Path, object_ids: Optional[Sequence[str]], object_list: Optional[Path]) -> List[str]:
    """Collect object ids from CLI filters or the processed_data folder.

    Args:
        asset_root: DGN object asset folder.
        object_ids: Optional object ids passed directly on the command line.
        object_list: Optional text file with one object id per line.

    Returns:
        List[str]: Sorted object ids to check.
    """
    selected = set(object_ids or [])
    if object_list is not None:
        with object_list.expanduser().open("r", encoding="utf-8") as file_obj:
            selected.update(line.strip() for line in file_obj if line.strip() and not line.startswith("#"))
    if selected:
        return sorted(selected)

    processed_root = asset_root / "processed_data"
    return sorted(path.name for path in processed_root.iterdir() if path.is_dir())


def collect_object_scales(asset_root: Path, object_id: str, args: argparse.Namespace) -> Tuple[ScaleTuple, ...]:
    """Collect unique scene object scales for one object.

    Args:
        asset_root: DGN object asset folder.
        object_id: Object directory name under processed_data.
        args: Parsed scan arguments.

    Returns:
        Tuple[ScaleTuple, ...]: Unique xyz scale tuples for this object.
    """
    scales: List[ScaleTuple] = []
    if args.scale_source == "manual":
        scales.extend(scalar_scale_tuple(float(value)) for value in (args.scales or []))
    elif args.scale_source == "filename":
        scales.extend(scales_from_scene_filenames(asset_root, object_id, args.scene_type, args.scale_denominator))
    elif args.scale_source == "scene_cfg":
        scales.extend(scales_from_scene_configs(asset_root, object_id, args.scene_type))
    else:
        raise ValueError(f"Unsupported scale source: {args.scale_source}")

    if args.include_unit_scale:
        scales.append((1.0, 1.0, 1.0))
    if not scales:
        scales.append((1.0, 1.0, 1.0))
    return tuple(sorted(set(scales)))


def scales_from_scene_filenames(
    asset_root: Path,
    object_id: str,
    scene_type: str,
    scale_denominator: float,
) -> List[ScaleTuple]:
    """Parse scalar scales from scene config filenames.

    Args:
        asset_root: DGN object asset folder.
        object_id: Object directory name under processed_data.
        scene_type: Scene type directory name, for example tabletop_ur10e.
        scale_denominator: Divisor for the integer in filenames such as scale008.

    Returns:
        List[ScaleTuple]: Unique xyz scale tuples found in filenames.
    """
    scene_dir = asset_root / "scene_cfg" / object_id / scene_type
    scales = []
    for scene_path in scene_dir.glob("*.npy"):
        match = SCALE_RE.search(scene_path.stem)
        if match is None:
            continue
        scales.append(scalar_scale_tuple(int(match.group(1)) / scale_denominator))
    return sorted(set(scales))


def scales_from_scene_configs(asset_root: Path, object_id: str, scene_type: str) -> List[ScaleTuple]:
    """Load scene configs and read exact object scales.

    Args:
        asset_root: DGN object asset folder.
        object_id: Object directory name under processed_data.
        scene_type: Scene type directory name, for example tabletop_ur10e.

    Returns:
        List[ScaleTuple]: Unique xyz scale tuples stored in scene configs.
    """
    import numpy as np

    scene_dir = asset_root / "scene_cfg" / object_id / scene_type
    scales = []
    for scene_path in scene_dir.glob("*.npy"):
        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        obj_name = scene_cfg["task"]["obj_name"]
        scales.append(normalize_scale(scene_cfg["scene"][obj_name]["scale"]))
    return sorted(set(scales))


def scalar_scale_tuple(value: float) -> ScaleTuple:
    """Convert one scalar scale to an xyz tuple.

    Args:
        value: Scalar object scale.

    Returns:
        ScaleTuple: Equivalent xyz scale tuple.
    """
    scale = float(value)
    return (scale, scale, scale)


def normalize_scale(scale_value) -> ScaleTuple:
    """Normalize scalar or xyz scale data to a three-float tuple.

    Args:
        scale_value: Scalar or length-three scale value.

    Returns:
        ScaleTuple: Scale as an xyz tuple.
    """
    import numpy as np

    scale = np.asarray(scale_value, dtype=np.float64)
    if scale.ndim == 0:
        return scalar_scale_tuple(float(scale))
    if scale.shape == (3,):
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    raise ValueError(f"Unsupported scale shape: {scale.shape}")


def convex_piece_paths(asset_root: Path, object_id: str) -> List[Path]:
    """Collect convex decomposition OBJ files for one object.

    Args:
        asset_root: DGN object asset folder.
        object_id: Object directory name under processed_data.

    Returns:
        List[Path]: Sorted convex_piece OBJ paths.
    """
    mesh_dir = asset_root / "processed_data" / object_id / "urdf" / "meshes"
    return sorted(mesh_dir.glob("convex_piece_*.obj"))


def local_scales_from_urdf(asset_root: Path, object_id: str) -> Dict[str, ScaleTuple]:
    """Read local mesh scales from coacd.urdf when available.

    Args:
        asset_root: DGN object asset folder.
        object_id: Object directory name under processed_data.

    Returns:
        Dict[str, ScaleTuple]: Mapping from mesh basename to local xyz scale.
    """
    urdf_path = asset_root / "processed_data" / object_id / "urdf" / "coacd.urdf"
    if not urdf_path.exists():
        return {}

    result: Dict[str, ScaleTuple] = {}
    root = ET.parse(urdf_path).getroot()
    for mesh_node in root.findall(".//mesh"):
        filename = mesh_node.attrib.get("filename")
        if not filename:
            continue
        scale_text = mesh_node.attrib.get("scale", "1.0 1.0 1.0")
        values = tuple(float(value) for value in scale_text.split())
        if len(values) == 1:
            scale = scalar_scale_tuple(values[0])
        elif len(values) == 3:
            scale = (values[0], values[1], values[2])
        else:
            raise ValueError(f"Unsupported mesh scale in {urdf_path}: {scale_text}")
        result[Path(filename).name] = scale
    return result


def multiply_scale(lhs: ScaleTuple, rhs: ScaleTuple) -> ScaleTuple:
    """Multiply two xyz scale tuples elementwise.

    Args:
        lhs: First xyz scale tuple.
        rhs: Second xyz scale tuple.

    Returns:
        ScaleTuple: Elementwise product.
    """
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])


def run_scan(args: argparse.Namespace) -> int:
    """Run the parent object scan and write reports.

    Args:
        args: Parsed scan arguments.

    Returns:
        int: Process exit code, zero only when no failures are found.
    """
    asset_root = resolve_asset_root(args)
    if not (asset_root / "processed_data").is_dir():
        raise FileNotFoundError(f"Missing processed_data folder: {asset_root / 'processed_data'}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (asset_root / f"coal_convex_mesh_check_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "coal_convex_mesh_check.jsonl"
    failed_path = output_dir / "coal_convex_mesh_check_failures.jsonl"
    summary_path = output_dir / "coal_convex_mesh_check_summary.json"

    object_ids = sorted_object_ids(asset_root, args.object_id, args.object_list)
    if args.limit_objects is not None:
        object_ids = object_ids[: args.limit_objects]
    jobs = [ObjectJob(object_id=obj_id, scales=collect_object_scales(asset_root, obj_id, args)) for obj_id in object_ids]

    start_time = time.time()
    failures: List[dict] = []
    checked_objects = 0
    checked_pieces = 0
    checked_loads = 0
    print(f"asset_root={asset_root}")
    print(f"objects={len(jobs)} workers={args.workers} output_dir={output_dir}")

    with detail_path.open("w", encoding="utf-8") as detail_file, failed_path.open("w", encoding="utf-8") as failed_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
            future_to_job = {
                executor.submit(run_object_subprocess, asset_root, job, args.timeout_sec): job for job in jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                record = future.result()
                if record.get("status") in {"timeout", "worker_failed", "bad_worker_output"} and args.diagnose_crashes:
                    record["diagnosed_failures"] = diagnose_crashed_object(asset_root, job, args.timeout_sec)

                detail_file.write(json.dumps(record, sort_keys=True) + "\n")
                detail_file.flush()
                checked_objects += 1
                checked_pieces += int(record.get("num_pieces", 0))
                checked_loads += int(record.get("num_loads", 0))

                record_failures = flatten_failure_records(record)
                for failure in record_failures:
                    failures.append(failure)
                    failed_file.write(json.dumps(failure, sort_keys=True) + "\n")
                failed_file.flush()

                if args.progress_interval and checked_objects % args.progress_interval == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"checked {checked_objects}/{len(jobs)} objects, "
                        f"failures={len(failures)}, elapsed={elapsed:.1f}s"
                    )

    summary = {
        "asset_root": str(asset_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - start_time, 3),
        "num_objects": len(jobs),
        "num_objects_checked": checked_objects,
        "num_piece_files_seen": checked_pieces,
        "num_coal_load_calls": checked_loads,
        "num_failures": len(failures),
        "detail_path": str(detail_path),
        "failed_path": str(failed_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


def run_object_subprocess(asset_root: Path, job: ObjectJob, timeout_sec: float) -> dict:
    """Run one object check in an isolated Python subprocess.

    Args:
        asset_root: DGN object asset folder.
        job: Object id and scales to test.
        timeout_sec: Maximum subprocess runtime in seconds.

    Returns:
        dict: Object-level result record.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        WORKER_OBJECT_MODE,
        "--asset-root",
        str(asset_root),
        "--object-id",
        job.object_id,
        "--scales-json",
        json.dumps(job.scales),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "object_id": job.object_id,
            "status": "timeout",
            "returncode": None,
            "num_scales": len(job.scales),
            "num_pieces": len(convex_piece_paths(asset_root, job.object_id)),
            "num_loads": 0,
            "stderr_tail": tail_text(exc.stderr),
        }
    if completed.returncode != 0:
        return {
            "object_id": job.object_id,
            "status": "worker_failed",
            "returncode": completed.returncode,
            "num_scales": len(job.scales),
            "num_pieces": len(convex_piece_paths(asset_root, job.object_id)),
            "num_loads": 0,
            "stdout_tail": tail_text(completed.stdout),
            "stderr_tail": tail_text(completed.stderr),
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "object_id": job.object_id,
            "status": "bad_worker_output",
            "returncode": completed.returncode,
            "num_scales": len(job.scales),
            "num_pieces": len(convex_piece_paths(asset_root, job.object_id)),
            "num_loads": 0,
            "error": repr(exc),
            "stdout_tail": tail_text(completed.stdout),
            "stderr_tail": tail_text(completed.stderr),
        }


def diagnose_crashed_object(asset_root: Path, job: ObjectJob, timeout_sec: float) -> List[dict]:
    """Retry a failed object one piece/scale pair at a time.

    Args:
        asset_root: DGN object asset folder.
        job: Object id and scales to test.
        timeout_sec: Maximum subprocess runtime per piece/scale pair.

    Returns:
        List[dict]: Failure records for individual piece/scale attempts.
    """
    failures = []
    local_scales = local_scales_from_urdf(asset_root, job.object_id)
    for mesh_path in convex_piece_paths(asset_root, job.object_id):
        local_scale = local_scales.get(mesh_path.name, (1.0, 1.0, 1.0))
        for scene_scale in job.scales:
            final_scale = multiply_scale(local_scale, scene_scale)
            failure = run_piece_subprocess(mesh_path, final_scale, timeout_sec)
            if failure is not None:
                failure["object_id"] = job.object_id
                failure["scene_scale"] = list(scene_scale)
                failure["local_scale"] = list(local_scale)
                failures.append(failure)
    return failures


def run_piece_subprocess(mesh_path: Path, scale: ScaleTuple, timeout_sec: float) -> Optional[dict]:
    """Run one mesh/scale load in an isolated subprocess.

    Args:
        mesh_path: Convex piece OBJ path.
        scale: Final xyz scale passed to COAL.
        timeout_sec: Maximum subprocess runtime in seconds.

    Returns:
        Optional[dict]: Failure record, or None when the load succeeds.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        WORKER_PIECE_MODE,
        "--mesh-path",
        str(mesh_path),
        "--scale-json",
        json.dumps(scale),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "piece_timeout",
            "mesh_path": str(mesh_path),
            "scale": list(scale),
            "stderr_tail": tail_text(exc.stderr),
        }
    if completed.returncode == 0:
        return None
    return {
        "status": "piece_failed",
        "returncode": completed.returncode,
        "mesh_path": str(mesh_path),
        "scale": list(scale),
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }


def flatten_failure_records(record: dict) -> List[dict]:
    """Extract failure records from an object-level result.

    Args:
        record: Object-level check result.

    Returns:
        List[dict]: Normalized failure records for failed output JSONL.
    """
    object_id = record.get("object_id")
    failures = []
    for failure in record.get("failures", []):
        merged = {"object_id": object_id, **failure}
        failures.append(merged)
    for failure in record.get("diagnosed_failures", []):
        failures.append(failure)
    if record.get("status") != "ok" and not failures:
        failures.append(record)
    return failures


def tail_text(value: Optional[str], max_chars: int = 4000) -> str:
    """Return the tail of command output for compact reports.

    Args:
        value: Optional command output text.
        max_chars: Maximum number of characters to keep.

    Returns:
        str: Output tail.
    """
    if value is None:
        return ""
    return str(value)[-max_chars:]


def run_worker_object(args: argparse.Namespace) -> int:
    """Load every convex piece for one object and write one JSON record.

    Args:
        args: Parsed hidden object-worker arguments.

    Returns:
        int: Worker process exit code.
    """
    import numpy as np
    from coal_openmp_wrapper import loadConvexMeshCpp

    scales = tuple(tuple(float(value) for value in scale) for scale in json.loads(args.scales_json))
    mesh_paths = convex_piece_paths(args.asset_root, args.object_id)
    local_scales = local_scales_from_urdf(args.asset_root, args.object_id)
    failures = []
    num_loads = 0

    for mesh_path in mesh_paths:
        local_scale = local_scales.get(mesh_path.name, (1.0, 1.0, 1.0))
        for scene_scale in scales:
            final_scale = multiply_scale(local_scale, scene_scale)
            num_loads += 1
            try:
                loadConvexMeshCpp(str(mesh_path), np.asarray(final_scale, dtype=np.float64))
            except Exception as exc:  # noqa: BLE001 - the report must preserve all COAL failure types.
                failures.append(
                    {
                        "status": "coal_exception",
                        "mesh_path": str(mesh_path),
                        "scene_scale": list(scene_scale),
                        "local_scale": list(local_scale),
                        "scale": list(final_scale),
                        "error": repr(exc),
                    }
                )

    if not mesh_paths:
        failures.append(
            {
                "status": "missing_convex_pieces",
                "mesh_dir": str(args.asset_root / "processed_data" / args.object_id / "urdf" / "meshes"),
                "error": "No convex_piece_*.obj files found.",
            }
        )

    record = {
        "object_id": args.object_id,
        "status": "ok" if not failures else "failed",
        "num_pieces": len(mesh_paths),
        "num_scales": len(scales),
        "num_loads": num_loads,
        "failures": failures,
    }
    print(json.dumps(record, sort_keys=True))
    return 0


def run_worker_piece(args: argparse.Namespace) -> int:
    """Load one convex piece at one final scale.

    Args:
        args: Parsed hidden piece-worker arguments.

    Returns:
        int: Worker process exit code, zero only when COAL loading succeeds.
    """
    import numpy as np
    from coal_openmp_wrapper import loadConvexMeshCpp

    scale = tuple(float(value) for value in json.loads(args.scale_json))
    loadConvexMeshCpp(str(args.mesh_path), np.asarray(scale, dtype=np.float64))
    return 0


def main() -> int:
    """Run scan or hidden worker mode.

    Args:
        None.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    if args.mode == WORKER_OBJECT_MODE:
        return run_worker_object(args)
    if args.mode == WORKER_PIECE_MODE:
        return run_worker_piece(args)
    return run_scan(args)


if __name__ == "__main__":
    raise SystemExit(main())

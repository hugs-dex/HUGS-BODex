#!/usr/bin/env python3
"""Save posed and scaled tabletop object heights for DGN scene configs.

The script reads tabletop scene config files from
`${AnyScaleGraspDataset}/object/{DGN_2k,DGN_5k}/scene_cfg`, computes the
z-height of each posed and scaled mesh from OBJ vertices, and writes the
results back into the corresponding object asset folder.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_DATASETS = ("DGN_2k", "DGN_5k")
DEFAULT_ENV_NAME = "AnyScaleGraspDataset"
DEFAULT_OUTPUT_STEM = "tabletop_scene_object_heights"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute tabletop scene object heights from mesh vertices after "
            "applying each scene's object scale and pose."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folders under ${AnyScaleGraspDataset}/object to process.",
    )
    parser.add_argument(
        "--dataset-root-env",
        default=DEFAULT_ENV_NAME,
        help="Environment variable that points to the AnyScaleGrasp dataset root.",
    )
    parser.add_argument(
        "--scene-type",
        default="tabletop_ur10e",
        help="Scene type folder to scan under each object's scene_cfg directory.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output basename written under each processed object asset folder.",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=None,
        help="Optional inclusive minimum height for writing the filtered scene list.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Optional inclusive maximum height for writing the filtered scene list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of scene configs per dataset, useful for debugging.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10000,
        help="Print progress every N processed scenes. Use 0 to disable.",
    )
    return parser.parse_args()


def dataset_root_from_env(dataset_name: str, env_name: str) -> Path:
    """Build one DGN object asset folder from the configured environment variable.

    Args:
        dataset_name: Dataset folder name, such as DGN_2k or DGN_5k.
        env_name: Environment variable name that stores the AnyScaleGrasp dataset root.

    Returns:
        Path: Absolute path to the selected object asset folder.
    """
    dataset_root = os.environ.get(env_name)
    if not dataset_root:
        raise EnvironmentError(f"Please set {env_name} to the AnyScaleGrasp dataset root.")
    return Path(dataset_root).expanduser().resolve() / "object" / dataset_name


def resolve_scene_paths(asset_root: Path, scene_type: str, limit: Optional[int]) -> List[Path]:
    """Collect tabletop scene config paths under one object asset folder.

    Args:
        asset_root: Path to the DGN object asset folder.
        scene_type: Scene type directory name to scan, for example tabletop_ur10e.
        limit: Optional maximum number of scene paths to return.

    Returns:
        List[Path]: Sorted scene config paths.
    """
    scene_root = asset_root / "scene_cfg"
    scene_paths = sorted(scene_root.glob(f"*/{scene_type}/*.npy"))
    if limit is not None:
        scene_paths = scene_paths[:limit]
    return scene_paths


def load_scene_config(scene_path: Path) -> Dict:
    """Load one numpy scene config file.

    Args:
        scene_path: Path to a .npy scene config file.

    Returns:
        Dict: Scene config dictionary stored in the .npy file.
    """
    return np.load(scene_path, allow_pickle=True).item()


def resolve_asset_path(scene_path: Path, stored_path: str) -> Path:
    """Resolve a scene-relative asset path to an absolute filesystem path.

    Args:
        scene_path: Path to the scene config that contains the relative asset path.
        stored_path: Asset path string stored inside the scene config.

    Returns:
        Path: Absolute resolved asset path.
    """
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return (scene_path.parent / path).resolve()


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    """Load vertex positions from an OBJ mesh file.

    Args:
        obj_path: Path to an OBJ mesh file.

    Returns:
        np.ndarray: Vertex array with shape (N, 3).
    """
    vertices = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {obj_path}")
    return np.asarray(vertices, dtype=np.float64)


def quaternion_wxyz_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix.

    Args:
        quaternion: Quaternion ordered as (w, x, y, z).

    Returns:
        np.ndarray: Rotation matrix with shape (3, 3).
    """
    quat = np.asarray(list(quaternion), dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def scale_vector(scale_value) -> np.ndarray:
    """Normalize a scalar or xyz scale value to a three-element vector.

    Args:
        scale_value: Scalar scale or array-like xyz scale from the scene config.

    Returns:
        np.ndarray: Scale vector with shape (3,).
    """
    scale = np.asarray(scale_value, dtype=np.float64)
    if scale.ndim == 0:
        return np.full(3, float(scale), dtype=np.float64)
    if scale.shape == (3,):
        return scale
    raise ValueError(f"Unsupported scale shape: {scale.shape}")


def posed_scaled_z_extent(vertices: np.ndarray, scale, pose) -> Tuple[float, float, float]:
    """Compute the z-extent of mesh vertices after scene scale and pose.

    Args:
        vertices: Mesh vertex array with shape (N, 3).
        scale: Scalar or xyz scale from the rigid object config.
        pose: Object pose ordered as (x, y, z, qw, qx, qy, qz).

    Returns:
        Tuple[float, float, float]: Object height, minimum z, and maximum z.
    """
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (7,):
        raise ValueError(f"Unsupported pose shape: {pose_array.shape}")

    scaled_vertices = vertices * scale_vector(scale)
    rotation = quaternion_wxyz_to_matrix(pose_array[3:])
    transformed_z = scaled_vertices @ rotation[2, :] + pose_array[2]
    z_min = float(np.min(transformed_z))
    z_max = float(np.max(transformed_z))
    return z_max - z_min, z_min, z_max


def scene_height_record(scene_path: Path, asset_root: Path, vertex_cache: Dict[Path, np.ndarray]) -> Dict:
    """Compute and package the object-height record for one tabletop scene.

    Args:
        scene_path: Path to one tabletop scene config file.
        asset_root: Root folder for the current object asset dataset.
        vertex_cache: Mutable cache that maps mesh paths to loaded vertex arrays.

    Returns:
        Dict: JSON-serializable height record for the scene.
    """
    scene_cfg = load_scene_config(scene_path)
    object_name = scene_cfg["task"]["obj_name"]
    object_cfg = scene_cfg["scene"][object_name]
    mesh_path = resolve_asset_path(scene_path, object_cfg["file_path"])

    if mesh_path not in vertex_cache:
        vertex_cache[mesh_path] = load_obj_vertices(mesh_path)

    height, z_min, z_max = posed_scaled_z_extent(
        vertex_cache[mesh_path],
        object_cfg["scale"],
        object_cfg["pose"],
    )
    try:
        mesh_path_record = mesh_path.relative_to(asset_root).as_posix()
    except ValueError:
        # Some synced scene_cfg files keep absolute mesh paths outside the local
        # asset root. Preserve the absolute path string so height indexing still
        # works instead of failing during metadata serialization.
        mesh_path_record = str(mesh_path)
    return {
        "scene_path": scene_path.relative_to(asset_root).as_posix(),
        "scene_id": str(scene_cfg["scene_id"]),
        "object_name": str(object_name),
        "mesh_path": mesh_path_record,
        "scale": np.asarray(object_cfg["scale"], dtype=float).tolist(),
        "pose": np.asarray(object_cfg["pose"], dtype=float).tolist(),
        "height": float(height),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def height_in_filter(height: float, min_height: Optional[float], max_height: Optional[float]) -> bool:
    """Check whether a height satisfies optional inclusive filter bounds.

    Args:
        height: Object height value to test.
        min_height: Optional inclusive lower bound.
        max_height: Optional inclusive upper bound.

    Returns:
        bool: True if the height is inside the requested bounds.
    """
    if min_height is not None and height < min_height:
        return False
    if max_height is not None and height > max_height:
        return False
    return True


def process_dataset(dataset_name: str, args: argparse.Namespace) -> Dict:
    """Process all selected tabletop scenes for one DGN dataset.

    Args:
        dataset_name: Dataset folder name, such as DGN_2k or DGN_5k.
        args: Parsed command-line arguments.

    Returns:
        Dict: Summary metadata for the processed dataset.
    """
    asset_root = dataset_root_from_env(dataset_name, args.dataset_root_env)
    scene_paths = resolve_scene_paths(asset_root, args.scene_type, args.limit)
    output_jsonl = asset_root / f"{args.output_stem}.jsonl"
    filtered_txt = asset_root / f"{args.output_stem}_filtered.txt"

    vertex_cache: Dict[Path, np.ndarray] = {}
    filtered_count = 0
    min_seen = None
    max_seen = None
    should_write_filter = args.min_height is not None or args.max_height is not None

    with output_jsonl.open("w", encoding="utf-8") as jsonl_file:
        filter_file = filtered_txt.open("w", encoding="utf-8") if should_write_filter else None
        try:
            for index, scene_path in enumerate(scene_paths, start=1):
                record = scene_height_record(scene_path, asset_root, vertex_cache)
                height = record["height"]
                min_seen = height if min_seen is None else min(min_seen, height)
                max_seen = height if max_seen is None else max(max_seen, height)

                jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
                if filter_file is not None and height_in_filter(height, args.min_height, args.max_height):
                    filter_file.write(record["scene_path"] + "\n")
                    filtered_count += 1

                if args.progress_interval and index % args.progress_interval == 0:
                    print(f"{dataset_name}: processed {index}/{len(scene_paths)} scenes")
        finally:
            if filter_file is not None:
                filter_file.close()

    summary = {
        "dataset": dataset_name,
        "asset_root": str(asset_root),
        "scene_type": args.scene_type,
        "output_jsonl": output_jsonl.name,
        "filtered_txt": filtered_txt.name if should_write_filter else None,
        "height_source": "OBJ mesh vertices after scene scale and pose",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_scenes": len(scene_paths),
        "num_meshes_loaded": len(vertex_cache),
        "min_height": min_seen,
        "max_height": max_seen,
        "filter_min_height": args.min_height,
        "filter_max_height": args.max_height,
        "num_filtered_scenes": filtered_count if should_write_filter else None,
    }
    write_summary(asset_root, args.output_stem, summary)
    return summary


def write_summary(asset_root: Path, output_stem: str, summary: Dict) -> Path:
    """Write a compact JSON summary next to the JSONL height records.

    Args:
        asset_root: Root folder for the current object asset dataset.
        output_stem: Output basename shared by result files.
        summary: JSON-serializable summary dictionary.

    Returns:
        Path: Path to the written summary file.
    """
    summary_path = asset_root / f"{output_stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return summary_path


def main() -> None:
    """Run the tabletop object height export script.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    for dataset_name in args.datasets:
        summary = process_dataset(dataset_name, args)
        print(
            f"{dataset_name}: wrote {summary['num_scenes']} height records, "
            f"loaded {summary['num_meshes_loaded']} meshes"
        )


if __name__ == "__main__":
    main()

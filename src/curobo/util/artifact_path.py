"""Resolve relocated input artifacts without modifying saved metadata."""

import json
import os
from pathlib import Path

import numpy as np

def resolve_artifact_path(path):
    """Apply explicit old-root to new-root mappings from HUGS_PATH_MAP.

    Args:
        path: Path stored in an input artifact.

    Returns:
        Expanded path, with the longest matching configured prefix replaced.
    """
    value = os.path.expanduser(os.fspath(path))
    mappings = json.loads(os.environ.get("HUGS_PATH_MAP", "{}"))
    if not isinstance(mappings, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in mappings.items()
    ):
        raise ValueError("HUGS_PATH_MAP must be a JSON object mapping old roots to new roots")
    source = Path(value)
    for old, new in sorted(mappings.items(), key=lambda pair: len(Path(pair[0]).parts), reverse=True):
        try:
            relative = source.relative_to(Path(os.path.expanduser(old)))
        except ValueError:
            continue
        return str(Path(os.path.expanduser(new)) / relative)
    root = os.environ.get("HUGS_DATASET_ROOT")
    if root and not source.is_absolute():
        if ".." in source.parts:
            raise ValueError(f"Dataset reference escapes HUGS_DATASET_ROOT: {path}")
        return str(Path(root).expanduser() / source)
    return value


def portable_artifact_metadata(metadata):
    """Copy result metadata using dataset-relative paths when a root is set."""
    root = os.environ.get("HUGS_DATASET_ROOT")
    if not root:
        return metadata
    root = os.path.abspath(os.path.expanduser(root))

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(item) for item in value)
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
            result = np.empty(value.shape, dtype=object)
            for index, item in enumerate(value.flat):
                result.flat[index] = convert(item)
            return result.astype(str) if value.dtype.kind == "U" else result
        if isinstance(value, (str, Path)) and os.path.isabs(value):
            if os.path.commonpath([root, os.path.abspath(value)]) == root:
                return Path(os.path.relpath(value, root)).as_posix()
        return value

    return {**convert(metadata), "path_root": "HUGS_DATASET_ROOT"}

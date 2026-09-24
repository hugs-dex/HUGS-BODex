"""Resolve relocated input artifacts without modifying saved metadata."""

import json
import os
from pathlib import Path


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
    return value

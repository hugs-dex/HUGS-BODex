"""Process-local caches for contact object convex geometry."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable, Optional, Tuple

import numpy as np
import torch

from coal_openmp_wrapper import loadConvexMeshCpp
from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.geom.sdf.gjk_coal import getBoundingPrimitive


_DEFAULT_MAX_ENTRIES = 4096
_URDF_CACHE_MAX_ENTRIES = int(os.environ.get("CUROBO_CONTACT_URDF_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
_CONVEX_MESH_CACHE_MAX_ENTRIES = int(
    os.environ.get("CUROBO_CONTACT_CONVEX_MESH_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES)
)
_OBB_CACHE_MAX_ENTRIES = int(os.environ.get("CUROBO_CONTACT_OBB_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
_URDF_MANIFEST_CACHE: "OrderedDict[Hashable, Tuple[ContactConvexPart, ...]]" = OrderedDict()
_CONVEX_MESH_CACHE: "OrderedDict[Hashable, Any]" = OrderedDict()
_OBB_PARAM_CACHE: "OrderedDict[Hashable, torch.Tensor]" = OrderedDict()
_CONTACT_OBB_CACHE_MODE_SCALED = "scaled"
_CONTACT_OBB_CACHE_MODE_UNIFORM_RAW = "uniform_raw"


@dataclass(frozen=True)
class ContactConvexPart:
    """One convex-decomposition mesh entry for a contact object."""

    file_path: str
    local_scale: Tuple[float, float, float]


def clear_contact_mesh_cache() -> None:
    """Clear all process-local contact geometry caches.

    Args:
        None.

    Returns:
        None.
    """
    _URDF_MANIFEST_CACHE.clear()
    _CONVEX_MESH_CACHE.clear()
    _OBB_PARAM_CACHE.clear()


def contact_mesh_cache_info() -> dict:
    """Return lightweight diagnostics for contact geometry caches.

    Args:
        None.

    Returns:
        Dictionary containing active entry counts and configured capacities.
    """
    return {
        "urdf_manifest_size": len(_URDF_MANIFEST_CACHE),
        "urdf_manifest_max_entries": _URDF_CACHE_MAX_ENTRIES,
        "convex_mesh_size": len(_CONVEX_MESH_CACHE),
        "convex_mesh_max_entries": _CONVEX_MESH_CACHE_MAX_ENTRIES,
        "obb_param_size": len(_OBB_PARAM_CACHE),
        "obb_param_max_entries": _OBB_CACHE_MAX_ENTRIES,
    }


def get_cached_contact_convex_parts(urdf_path: str) -> Tuple[ContactConvexPart, ...]:
    """Read or parse the convex-decomposition manifest for one object URDF.

    Args:
        urdf_path: Absolute path to the contact object's convex-decomposition URDF.

    Returns:
        Tuple of convex part mesh paths and local mesh scales. The manifest is
        independent of scene object scale, so it can be shared across scales.
    """
    cache_key = ("contact_urdf_manifest", _file_signature(urdf_path))
    if _URDF_CACHE_MAX_ENTRIES <= 0:
        return _load_contact_convex_parts(urdf_path)
    cached = _URDF_MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        _URDF_MANIFEST_CACHE.move_to_end(cache_key)
        return cached
    manifest = _load_contact_convex_parts(urdf_path)
    _URDF_MANIFEST_CACHE[cache_key] = manifest
    _URDF_MANIFEST_CACHE.move_to_end(cache_key)
    _trim_lru_cache(_URDF_MANIFEST_CACHE, _URDF_CACHE_MAX_ENTRIES)
    return manifest


def get_cached_contact_convex_mesh(file_path: str, scale: Any):
    """Read or load a scaled COAL convex collision geometry.

    Args:
        file_path: Absolute path to a convex part mesh file.
        scale: Final xyz scale applied before building the COAL convex geometry.

    Returns:
        COAL collision geometry shared pointer returned by ``loadConvexMeshCpp``.
    """
    scale_tuple = _scale_tuple(scale)
    cache_key = ("contact_convex_mesh", _file_signature(file_path), scale_tuple)
    if _CONVEX_MESH_CACHE_MAX_ENTRIES <= 0:
        return loadConvexMeshCpp(file_path, np.asarray(scale_tuple, dtype=np.float64))
    cached = _CONVEX_MESH_CACHE.get(cache_key)
    if cached is not None:
        _CONVEX_MESH_CACHE.move_to_end(cache_key)
        return cached
    convex_mesh = loadConvexMeshCpp(file_path, np.asarray(scale_tuple, dtype=np.float64))
    _CONVEX_MESH_CACHE[cache_key] = convex_mesh
    _CONVEX_MESH_CACHE.move_to_end(cache_key)
    _trim_lru_cache(_CONVEX_MESH_CACHE, _CONVEX_MESH_CACHE_MAX_ENTRIES)
    return convex_mesh


def get_cached_contact_obb_param(
    file_path: str,
    scale: Any,
    cache_mode: Optional[str] = _CONTACT_OBB_CACHE_MODE_UNIFORM_RAW,
) -> torch.Tensor:
    """Read or compute a contact convex part's object-local OBB parameters.

    Args:
        file_path: Absolute path to a convex part mesh file.
        scale: Final xyz scale used for this scene object.
        cache_mode: ``"scaled"`` caches the old exact result for each final
            scale. ``"uniform_raw"`` computes the raw OBB once and scales it
            when the final scale is positive uniform; non-uniform scales fall
            back to the exact scaled OBB.

    Returns:
        CPU tensor with 19 values: flattened 4x4 OBB transform followed by half
        extents. A clone is returned so callers cannot mutate cached tensors.
    """
    mode = _normalize_contact_obb_cache_mode(cache_mode)
    scale_tuple = _scale_tuple(scale)
    file_sig = _file_signature(file_path)
    if mode == _CONTACT_OBB_CACHE_MODE_UNIFORM_RAW and _is_positive_uniform_scale(scale_tuple):
        raw_param = _get_or_compute_obb_param(("contact_raw_obb", file_sig), file_path, (1.0, 1.0, 1.0))
        return _scale_uniform_obb_param(raw_param, scale_tuple[0])

    scaled_param = _get_or_compute_obb_param(("contact_scaled_obb", file_sig, scale_tuple), file_path, scale_tuple)
    return scaled_param.clone()


def _load_contact_convex_parts(urdf_path: str) -> Tuple[ContactConvexPart, ...]:
    """Parse one contact object's convex-decomposition URDF.

    Args:
        urdf_path: Absolute path to the contact object's convex-decomposition URDF.

    Returns:
        Tuple of convex part mesh paths and local mesh scales.
    """
    parser = UrdfKinematicsParser(urdf_path, coacd_obj_version=True)
    parts = []
    for link in parser._robot.link_map.values():
        if len(link.visuals) != 1:
            raise ValueError(f"Expected one visual per convex part in {urdf_path}, got {len(link.visuals)}")
        mesh = link.visuals[0].geometry.mesh
        abs_file_path = parser._robot._filename_handler(fname=mesh.filename)
        parts.append(ContactConvexPart(file_path=abs_file_path, local_scale=_scale_tuple(mesh.scale)))
    return tuple(parts)


def _get_or_compute_obb_param(cache_key: Hashable, file_path: str, scale: Tuple[float, float, float]) -> torch.Tensor:
    """Read or compute an OBB parameter tensor for one mesh/scale key.

    Args:
        cache_key: Hashable key for the requested OBB.
        file_path: Absolute path to the convex part mesh file.
        scale: Final xyz scale applied before OBB computation.

    Returns:
        Cached CPU tensor containing OBB parameters.
    """
    if _OBB_CACHE_MAX_ENTRIES <= 0:
        return _compute_obb_param(file_path, scale)
    cached = _OBB_PARAM_CACHE.get(cache_key)
    if cached is not None:
        _OBB_PARAM_CACHE.move_to_end(cache_key)
        return cached
    obb_param = _compute_obb_param(file_path, scale)
    _OBB_PARAM_CACHE[cache_key] = obb_param
    _OBB_PARAM_CACHE.move_to_end(cache_key)
    _trim_lru_cache(_OBB_PARAM_CACHE, _OBB_CACHE_MAX_ENTRIES)
    return obb_param


def _compute_obb_param(file_path: str, scale: Tuple[float, float, float]) -> torch.Tensor:
    """Compute OBB parameters using the legacy trimesh path.

    Args:
        file_path: Absolute path to the convex part mesh file.
        scale: Final xyz scale applied before OBB computation.

    Returns:
        Detached CPU tensor containing OBB parameters.
    """
    obb_param = getBoundingPrimitive(file_path, np.asarray(scale, dtype=np.float64), "obb")
    return obb_param.detach().cpu().clone()


def _scale_uniform_obb_param(raw_obb_param: torch.Tensor, scalar_scale: float) -> torch.Tensor:
    """Scale a raw-mesh OBB exactly for a positive uniform scale.

    Args:
        raw_obb_param: Raw OBB parameters for scale ``[1, 1, 1]``.
        scalar_scale: Positive uniform scale factor.

    Returns:
        New OBB parameter tensor for the uniformly scaled mesh.
    """
    scaled_param = raw_obb_param.clone()
    transform = scaled_param[:16].view(4, 4).clone()
    transform[:3, 3] = transform[:3, 3] * float(scalar_scale)
    scaled_param[:16] = transform.reshape(-1)
    scaled_param[16:] = scaled_param[16:] * abs(float(scalar_scale))
    return scaled_param


def _trim_lru_cache(cache: OrderedDict, max_entries: int) -> None:
    """Trim an ordered dict LRU cache in-place.

    Args:
        cache: OrderedDict used as an LRU cache.
        max_entries: Maximum entries to retain.

    Returns:
        None.
    """
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _normalize_contact_obb_cache_mode(value: Optional[str]) -> str:
    """Normalize the configured contact OBB cache mode.

    Args:
        value: User-provided cache mode.

    Returns:
        Canonical cache mode name.
    """
    if value is None:
        return _CONTACT_OBB_CACHE_MODE_UNIFORM_RAW
    mode = str(value).strip().lower()
    if mode in {"", _CONTACT_OBB_CACHE_MODE_UNIFORM_RAW, "raw_uniform", "uniform", "auto"}:
        return _CONTACT_OBB_CACHE_MODE_UNIFORM_RAW
    if mode in {_CONTACT_OBB_CACHE_MODE_SCALED, "exact", "legacy"}:
        return _CONTACT_OBB_CACHE_MODE_SCALED
    raise ValueError(
        f"Unsupported contact_obb_cache_mode={value!r}. "
        f"Expected '{_CONTACT_OBB_CACHE_MODE_SCALED}' or '{_CONTACT_OBB_CACHE_MODE_UNIFORM_RAW}'."
    )


def _is_positive_uniform_scale(scale: Tuple[float, float, float]) -> bool:
    """Check whether an xyz scale is positive and uniform.

    Args:
        scale: Normalized xyz scale tuple.

    Returns:
        True when all scale components match and the scalar is positive.
    """
    scale_array = np.asarray(scale, dtype=np.float64)
    return bool(scale_array[0] > 0.0 and np.allclose(scale_array, scale_array[0], rtol=1e-9, atol=1e-12))


def _scale_tuple(value: Any) -> Tuple[float, float, float]:
    """Normalize an optional scalar or xyz scale to a 3-float tuple.

    Args:
        value: Optional scalar or sequence scale.

    Returns:
        Tuple with exactly three float scale values.
    """
    if value is None:
        return (1.0, 1.0, 1.0)
    scale = np.asarray(value, dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scalar = float(scale[0])
        return (scalar, scalar, scalar)
    if scale.size != 3:
        raise ValueError(f"Scale must be scalar or xyz, got shape {scale.shape}")
    return tuple(float(item) for item in scale)


def _file_signature(file_path: str) -> Tuple[Hashable, ...]:
    """Build a lightweight stale-safe file signature.

    Args:
        file_path: Absolute or relative file path.

    Returns:
        Tuple containing real path, file size, and nanosecond mtime.
    """
    real_path = os.path.realpath(file_path)
    stat = os.stat(real_path)
    return (real_path, int(stat.st_size), int(stat.st_mtime_ns))

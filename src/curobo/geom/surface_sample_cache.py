"""Object-local surface sampling cache for repeated synthesis scene updates."""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple

import numpy as np
import trimesh
import trimesh.scene


_DEFAULT_MAX_ENTRIES = 4096
_CACHE_MAX_ENTRIES = int(os.environ.get("CUROBO_SURFACE_SAMPLE_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
_OBJECT_HULL_CACHE_MAX_ENTRIES = int(os.environ.get("CUROBO_OBJECT_HULL_CACHE_MAX_ENTRIES", _CACHE_MAX_ENTRIES))
_SURFACE_SAMPLE_CACHE: "OrderedDict[Hashable, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_OBJECT_HULL_CACHE: "OrderedDict[Hashable, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_HULL_CACHE_MODE_EXACT = "exact"
_HULL_CACHE_MODE_OBJECT_HULL = "object_hull"


def clear_surface_sample_cache() -> None:
    """Clear all cached object-local surface samples.

    Args:
        None.

    Returns:
        None.
    """
    _SURFACE_SAMPLE_CACHE.clear()
    _OBJECT_HULL_CACHE.clear()


def surface_sample_cache_info() -> dict:
    """Return lightweight cache diagnostics for benchmark scripts.

    Args:
        None.

    Returns:
        Dictionary containing the active entry count and configured capacity.
    """
    return {
        "size": len(_SURFACE_SAMPLE_CACHE),
        "max_entries": _CACHE_MAX_ENTRIES,
        "object_hull_size": len(_OBJECT_HULL_CACHE),
        "object_hull_max_entries": _OBJECT_HULL_CACHE_MAX_ENTRIES,
    }


def get_cached_surface_samples(
    mesh_obstacle: Any,
    sample_num: int,
    inflate: float,
    convex_hull: bool,
    return_radius: bool,
    hull_cache_mode: str = _HULL_CACHE_MODE_EXACT,
):
    """Sample a mesh surface using an object-local LRU cache.

    Args:
        mesh_obstacle: ``Mesh``-like obstacle with ``get_trimesh_mesh()``, ``pose``,
            ``file_path``, ``scale``, ``vertices``, and ``faces`` attributes.
        sample_num: Number of points requested from the surface sampler.
        inflate: Distance used to push vertices along object-local vertex normals
            before sampling.
        convex_hull: Whether sampling should happen on the inflated convex hull.
        return_radius: Whether to append the collision-free sphere radius column.
        hull_cache_mode: ``"exact"`` preserves the legacy order
            ``scale -> inflate -> convex_hull``. ``"object_hull"`` reuses one
            base convex hull per mesh and then applies scene scale and inflate.

    Returns:
        Tuple ``(points, normals)`` in world coordinates. When ``return_radius`` is
        true, ``points`` has a fourth radius column matching the legacy behavior.
    """
    sample_num = int(sample_num)
    inflate = float(inflate)
    convex_hull = bool(convex_hull)
    hull_cache_mode = _normalize_hull_cache_mode(hull_cache_mode)
    cache_key = _make_cache_key(mesh_obstacle, sample_num, inflate, convex_hull, hull_cache_mode)
    local_points, local_normals = _get_or_sample_local_surface(
        mesh_obstacle,
        cache_key,
        sample_num,
        inflate,
        convex_hull,
        hull_cache_mode,
    )
    return _transform_local_samples(mesh_obstacle.pose, local_points, local_normals, inflate, return_radius)


def _get_or_sample_local_surface(
    mesh_obstacle: Any,
    cache_key: Hashable,
    sample_num: int,
    inflate: float,
    convex_hull: bool,
    hull_cache_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read or create object-local surface samples for one mesh/sample setting.

    Args:
        mesh_obstacle: ``Mesh``-like obstacle used to build the base trimesh.
        cache_key: Hashable key describing mesh identity and sampling parameters.
        sample_num: Number of points requested from the sampler.
        inflate: Object-local normal inflation distance.
        convex_hull: Whether to sample on the inflated convex hull.
        hull_cache_mode: Surface sampling mode selected by config.

    Returns:
        Tuple of cached object-local ``(points, normals)`` arrays.
    """
    if _CACHE_MAX_ENTRIES <= 0:
        return _sample_local_surface(mesh_obstacle, sample_num, inflate, convex_hull, hull_cache_mode)

    cached = _SURFACE_SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        _SURFACE_SAMPLE_CACHE.move_to_end(cache_key)
        return cached

    local_points, local_normals = _sample_local_surface(mesh_obstacle, sample_num, inflate, convex_hull, hull_cache_mode)
    _SURFACE_SAMPLE_CACHE[cache_key] = (local_points, local_normals)
    _SURFACE_SAMPLE_CACHE.move_to_end(cache_key)
    while len(_SURFACE_SAMPLE_CACHE) > _CACHE_MAX_ENTRIES:
        _SURFACE_SAMPLE_CACHE.popitem(last=False)
    return local_points, local_normals


def _sample_local_surface(
    mesh_obstacle: Any,
    sample_num: int,
    inflate: float,
    convex_hull: bool,
    hull_cache_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an object-local mesh surface without applying the scene pose.

    Args:
        mesh_obstacle: ``Mesh``-like obstacle used to build the base trimesh.
        sample_num: Number of points requested from the sampler.
        inflate: Object-local normal inflation distance.
        convex_hull: Whether to sample on the inflated convex hull.
        hull_cache_mode: Surface sampling mode selected by config.

    Returns:
        Tuple ``(points, normals)`` in object-local coordinates.
    """
    sample_mesh = _build_sample_mesh(mesh_obstacle, inflate, convex_hull, hull_cache_mode)
    points, face_index = trimesh.sample.sample_surface_even(sample_mesh, sample_num)
    points, face_index = points.view(np.ndarray), face_index.view(np.ndarray)
    repeat_num = math.ceil(sample_num / points.shape[0])
    points = np.tile(points, (repeat_num, 1))[:sample_num]
    face_index = np.tile(face_index, repeat_num)[:sample_num]
    normals = sample_mesh.face_normals[face_index]
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def _build_sample_mesh(
    mesh_obstacle: Any,
    inflate: float,
    convex_hull: bool,
    hull_cache_mode: str,
) -> trimesh.Trimesh:
    """Build the local mesh used by ``sample_surface_even``.

    Args:
        mesh_obstacle: ``Mesh``-like obstacle used to build the base trimesh.
        inflate: Object-local normal inflation distance.
        convex_hull: Whether the caller requested convex-hull sampling.
        hull_cache_mode: Surface sampling mode selected by config.

    Returns:
        Trimesh object in object-local coordinates ready for surface sampling.
    """
    if (
        hull_cache_mode == _HULL_CACHE_MODE_OBJECT_HULL
        and convex_hull
        and getattr(mesh_obstacle, "file_path", None) is not None
    ):
        return _build_object_hull_sample_mesh(mesh_obstacle, inflate)

    mesh = mesh_obstacle.get_trimesh_mesh().copy()
    if inflate != 0.0:
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + inflate * np.asarray(mesh.vertex_normals)
    return mesh.convex_hull if convex_hull else mesh


def _build_object_hull_sample_mesh(mesh_obstacle: Any, inflate: float) -> trimesh.Trimesh:
    """Build a sampled mesh from one cached raw-object convex hull.

    Args:
        mesh_obstacle: File-backed ``Mesh`` obstacle. Its scene scale is applied
            after loading the base object hull.
        inflate: Object-local normal inflation distance applied after scale.

    Returns:
        Trimesh convex-hull approximation in object-local coordinates.
    """
    base_vertices, base_faces = _get_or_load_object_hull(mesh_obstacle)
    scale = _scale_vector(getattr(mesh_obstacle, "scale", None))
    scaled_vertices = base_vertices * scale.reshape(1, 3)
    scaled_hull = trimesh.Trimesh(vertices=scaled_vertices, faces=base_faces, process=False)
    if inflate == 0.0:
        return scaled_hull
    inflated_vertices = np.asarray(scaled_hull.vertices, dtype=np.float64) + inflate * np.asarray(
        scaled_hull.vertex_normals,
        dtype=np.float64,
    )
    return trimesh.Trimesh(vertices=inflated_vertices, faces=base_faces, process=False)


def _get_or_load_object_hull(mesh_obstacle: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Read or compute the raw-object convex hull for a file-backed mesh.

    Args:
        mesh_obstacle: File-backed ``Mesh`` obstacle. Scene scale is intentionally
            excluded from this cache so every scale of the same object reuses one hull.

    Returns:
        Tuple ``(vertices, faces)`` for the raw unscaled convex hull.
    """
    cache_key = _object_hull_cache_key(mesh_obstacle)
    if _OBJECT_HULL_CACHE_MAX_ENTRIES <= 0:
        return _load_object_hull(mesh_obstacle)

    cached = _OBJECT_HULL_CACHE.get(cache_key)
    if cached is not None:
        _OBJECT_HULL_CACHE.move_to_end(cache_key)
        return cached

    hull_data = _load_object_hull(mesh_obstacle)
    _OBJECT_HULL_CACHE[cache_key] = hull_data
    _OBJECT_HULL_CACHE.move_to_end(cache_key)
    while len(_OBJECT_HULL_CACHE) > _OBJECT_HULL_CACHE_MAX_ENTRIES:
        _OBJECT_HULL_CACHE.popitem(last=False)
    return hull_data


def _load_object_hull(mesh_obstacle: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a raw unscaled convex hull for a file-backed mesh.

    Args:
        mesh_obstacle: File-backed ``Mesh`` obstacle.

    Returns:
        Tuple ``(vertices, faces)`` copied from the raw mesh convex hull.
    """
    file_path = getattr(mesh_obstacle, "file_path", None)
    if file_path is None:
        raise ValueError("object_hull mode requires a file-backed mesh obstacle")
    mesh = trimesh.load(file_path, process=True, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    hull = mesh.convex_hull
    return np.asarray(hull.vertices, dtype=np.float64).copy(), np.asarray(hull.faces, dtype=np.int64).copy()


def _transform_local_samples(
    pose,
    local_points: np.ndarray,
    local_normals: np.ndarray,
    inflate: float,
    return_radius: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform cached object-local samples into one scene pose.

    Args:
        pose: Object pose in ``[x, y, z, qw, qx, qy, qz]`` format.
        local_points: Cached object-local points with shape ``[N, 3]``.
        local_normals: Cached object-local normals with shape ``[N, 3]``.
        inflate: Inflation distance used to derive the optional radius column.
        return_radius: Whether to append the legacy radius column.

    Returns:
        Tuple ``(points, normals)`` in world coordinates.
    """
    if pose is None:
        raise ValueError("Mesh pose is required for cached surface sample transformation")
    transform_matrix = trimesh.scene.transforms.kwargs_to_matrix(translation=pose[:3], quaternion=pose[3:])
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]
    world_points = local_points @ rotation.T + translation
    world_normals = local_normals @ rotation.T
    if return_radius:
        radius = 0.9 * float(inflate) * np.ones((world_points.shape[0], 1), dtype=world_points.dtype)
        world_points = np.concatenate((world_points, radius), axis=-1)
    return world_points, world_normals


def _make_cache_key(
    mesh_obstacle: Any,
    sample_num: int,
    inflate: float,
    convex_hull: bool,
    hull_cache_mode: str,
) -> Tuple[Hashable, ...]:
    """Build a cache key for object-local sampling.

    Args:
        mesh_obstacle: ``Mesh``-like obstacle containing mesh identity fields.
        sample_num: Number of points requested from the sampler.
        inflate: Object-local normal inflation distance.
        convex_hull: Whether to sample on the inflated convex hull.
        hull_cache_mode: Surface sampling mode selected by config.

    Returns:
        Hashable key that intentionally excludes pose so samples can be reused
        across many scene poses of the same object geometry.
    """
    return (
        _mesh_identity(mesh_obstacle),
        int(sample_num),
        float(inflate),
        bool(convex_hull),
        hull_cache_mode,
    )


def _object_hull_cache_key(mesh_obstacle: Any) -> Tuple[Hashable, ...]:
    """Build a pose- and scale-independent key for base object hulls.

    Args:
        mesh_obstacle: File-backed ``Mesh`` obstacle.

    Returns:
        Hashable key containing raw mesh file identity only.
    """
    file_path = getattr(mesh_obstacle, "file_path", None)
    if file_path is None:
        raise ValueError("object_hull mode requires a file-backed mesh obstacle")
    return ("object_hull_file", _file_signature(file_path))


def _mesh_identity(mesh_obstacle: Any) -> Tuple[Hashable, ...]:
    """Return a pose-independent identity for a mesh obstacle.

    Args:
        mesh_obstacle: ``Mesh``-like obstacle containing file or array geometry.

    Returns:
        Hashable mesh identity including scale and file metadata when available.
    """
    scale = _float_tuple(getattr(mesh_obstacle, "scale", None))
    file_path = getattr(mesh_obstacle, "file_path", None)
    if file_path is not None:
        return ("file", _file_signature(file_path), scale)

    vertices = getattr(mesh_obstacle, "vertices", None)
    faces = getattr(mesh_obstacle, "faces", None)
    if vertices is not None and faces is not None:
        return ("arrays", _array_digest(vertices), _array_digest(faces), scale)

    # Generated obstacle types such as cuboids or spheres do not expose vertices
    # directly. Hash their generated trimesh geometry so the base-class method
    # remains correct outside the BimanBODex Mesh contact-object path.
    mesh = mesh_obstacle.get_trimesh_mesh()
    return (
        "generated",
        type(mesh_obstacle).__name__,
        _array_digest(mesh.vertices),
        _array_digest(mesh.faces),
        scale,
    )


def _file_signature(file_path: str) -> Tuple[Hashable, ...]:
    """Build a lightweight signature for a mesh file.

    Args:
        file_path: Absolute or relative path to the mesh file.

    Returns:
        Tuple containing real path, size, and mtime so stale file edits miss cache.
    """
    real_path = os.path.realpath(file_path)
    stat = os.stat(real_path)
    return (real_path, int(stat.st_size), int(stat.st_mtime_ns))


def _array_digest(value: Any) -> Tuple[Hashable, ...]:
    """Build a stable digest for in-memory mesh arrays.

    Args:
        value: Array-like value, or ``None``.

    Returns:
        Tuple describing shape, dtype, and content digest.
    """
    if value is None:
        return ("none",)
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha1(array.view(np.uint8)).hexdigest()
    return (tuple(array.shape), str(array.dtype), digest)


def _float_tuple(value: Any) -> Tuple[float, ...]:
    """Normalize an optional numeric sequence for cache keys.

    Args:
        value: Optional scalar or sequence.

    Returns:
        Tuple of float values, or an empty tuple when ``value`` is ``None``.
    """
    if value is None:
        return ()
    return tuple(float(item) for item in np.ravel(value))


def _scale_vector(value: Any) -> np.ndarray:
    """Normalize an optional mesh scale to an xyz vector.

    Args:
        value: Optional scalar or sequence scale.

    Returns:
        Float64 numpy array with shape ``(3,)``.
    """
    if value is None:
        return np.ones(3, dtype=np.float64)
    scale = np.asarray(value, dtype=np.float64).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale[0], 3).astype(np.float64)
    if scale.size != 3:
        raise ValueError(f"Mesh scale must be scalar or xyz, got shape {scale.shape}")
    return scale.astype(np.float64)


def _normalize_hull_cache_mode(value: Optional[str]) -> str:
    """Normalize the configured hull cache mode.

    Args:
        value: User-provided hull cache mode.

    Returns:
        Canonical mode name.
    """
    if value is None:
        return _HULL_CACHE_MODE_EXACT
    mode = str(value).strip().lower()
    if mode in {"", _HULL_CACHE_MODE_EXACT, "legacy"}:
        return _HULL_CACHE_MODE_EXACT
    if mode in {_HULL_CACHE_MODE_OBJECT_HULL, "object", "mesh_hull"}:
        return _HULL_CACHE_MODE_OBJECT_HULL
    raise ValueError(
        f"Unsupported hull_cache_mode={value!r}. "
        f"Expected '{_HULL_CACHE_MODE_EXACT}' or '{_HULL_CACHE_MODE_OBJECT_HULL}'."
    )

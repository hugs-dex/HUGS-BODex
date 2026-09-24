#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""World represented as Meshes can be used with this module for collision checking."""

# Standard Library
from collections import OrderedDict
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple, Union

# Third Party
import numpy as np
import torch
import warp as wp
import trimesh 

# CuRobo
from coal_openmp_wrapper import loadConvexMeshCpp
from curobo.geom.contact_mesh_cache import (
    get_cached_contact_convex_mesh,
    get_cached_contact_convex_parts,
    get_cached_contact_obb_param,
)
from curobo.geom.sdf.gjk_coal import getBoundingPrimitive, ContactPDNConvexMesh, RobotSelfPenetration
from curobo.geom.sdf.warp_primitives import SdfMeshWarpPy, SweptSdfMeshWarpPy, ContactPDNMeshWarpPy
from curobo.geom.sdf.world import (
    ContactBuffer,
    CollisionQueryBuffer,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.types import Mesh, WorldConfig
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.warp_init import init_warp
from curobo.util.sample_lib import random_sample_points_on_sphere

@dataclass(frozen=True)
class WarpMeshData:
    """Data helper to use with warp for representing a mesh"""

    #: Name of the mesh.
    name: str

    #: Mesh ID, created by warp once mesh is loaded to device.
    m_id: int

    #: Vertices of the mesh.
    vertices: wp.array

    #: Faces of the mesh.
    faces: wp.array

    #: Warp mesh instance.
    mesh: wp.Mesh

    #: Approximate bytes used by vertices and faces stored in this Warp mesh.
    cache_bytes: int = 0


class WorldMeshCollision(WorldPrimitiveCollision):
    """World Mesh Collision using Nvidia's warp library."""

    def __init__(self, config: WorldCollisionConfig):
        """Initialize World Mesh Collision with given configuration."""
        init_warp()

        self.tensor_args = config.tensor_args

        self._env_n_mesh = None
        self._mesh_tensor_list = None
        self._env_mesh_names = None
        self._env_mesh_cache_keys = None
        self._contact_mesh_warp_idx = None
        self._wp_device = wp.device_from_torch(self.tensor_args.device)
        self._warp_mesh_cache_max_entries = max(
            0, int(getattr(config, "warp_mesh_cache_max_entries", 0))
        )
        self._warp_mesh_cache_max_bytes = max(
            0, int(getattr(config, "warp_mesh_cache_max_bytes", 0))
        )
        self._wp_mesh_cache = OrderedDict()  # stores warp meshes across environments
        self._wp_mesh_cache_current_bytes = 0
        
        super().__init__(config)

    def _init_cache(self):
        """Initialize cache for storing meshes."""
        if self.cache is None:
            self.cache = {} 
        if "mesh" not in self.cache:
            self.cache['mesh'] = 0 
        if (
            self.cache is not None
            and "mesh" in self.cache
            and (not self.cache["mesh"] in [None, 0])
        ):
            self._create_mesh_cache(self.cache["mesh"])

        return super()._init_cache()

    def load_contact_obj(self, world_model: Union[WorldConfig, List[WorldConfig]], contact_obj_names: Union[str, List[str]]):
        # world model should be initialized first!
        if not isinstance(world_model, List):
            world_model = [world_model]

        if not isinstance(contact_obj_names, List):
            contact_obj_names = [contact_obj_names]

        sample_params = self.contact_obj_sample_params
        hull_cache_mode = "exact" if sample_params is None else sample_params.get("hull_cache_mode", "exact")
        contact_obb_cache_mode = (
            "uniform_raw" if sample_params is None else sample_params.get("contact_obb_cache_mode", "uniform_raw")
        )
        load_contact_surface_points, contact_surface_sample_num = self._contact_surface_point_request(sample_params)
        load_mesh = self.contact_obj_mesh_loading
        
        point_list = []
        normal_list = []
        scale_list = []
        mesh_idx_lst = []
        mesh_pose_lst = []
        mesh_inv_pose_lst = []
        self._eval_mesh_idx = []
        self._contact_mesh_surface_points = [] if load_contact_surface_points else None
        self._contact_mesh_surface_normals = [] if load_contact_surface_points else None
        for env_id, world in enumerate(world_model):
            idx, m = world.find_mesh_by_name(contact_obj_names[env_id])
            self._eval_mesh_idx.append(idx)
            mesh_idx_lst.append(self._mesh_tensor_list[0][env_id, idx])
            mesh_inv_pose_lst.append(self._mesh_tensor_list[1][env_id, idx])
            mesh_pose_lst.append(m.pose)
            scale_list.append(m.scale)
            
            if load_contact_surface_points:
                surface_points, surface_normals = m.get_samples_on_surface(
                    contact_surface_sample_num,
                    0.0,
                    False,
                    False,
                )
                self._contact_mesh_surface_points.append(surface_points)
                self._contact_mesh_surface_normals.append(surface_normals)
            
            if sample_params is not None:
                p, normal = m.get_samples_on_surface(
                    sample_params['num'],
                    sample_params['inflate'],
                    sample_params['convex_hull'],
                    sample_params['collision_free'],
                    hull_cache_mode=hull_cache_mode,
                )
                point_list.append(p)
                normal_list.append(normal)
        
        self._contact_mesh_warp_idx = torch.stack(mesh_idx_lst)
        self._contact_mesh_inv_poses = torch.stack(mesh_inv_pose_lst)
        self._contact_mesh_poses = self.tensor_args.to_device(np.stack(mesh_pose_lst))
        self._contact_mesh_scales = np.stack(scale_list)
        if load_contact_surface_points:
            self._contact_mesh_surface_points = self.tensor_args.to_device(np.stack(self._contact_mesh_surface_points))
            self._contact_mesh_surface_normals = self.tensor_args.to_device(np.stack(self._contact_mesh_surface_normals))
        
        # For grasp pose initialization 
        if sample_params is not None:
            points = self.tensor_args.to_device(np.stack(point_list, axis=0)).unsqueeze(1)
            normals = self.tensor_args.to_device(np.stack(normal_list, axis=0))
            # filter out samples collided with environments
            if sample_params['collision_free']:
                _collision_query_buffer = CollisionQueryBuffer()
                _collision_query_buffer.update_buffer_shape(
                    points.shape, self.tensor_args, self.collision_types
                )
                env_query_idx = self.tensor_args.to_device(torch.arange(0, points.shape[0])).int()
                coll_result = self.get_sphere_distance(points, 
                    _collision_query_buffer, 
                    weight=self.tensor_args.to_device(torch.ones(points.shape[0])),
                    activation_distance=self.tensor_args.to_device([0.0]),
                    contact_distance=self.tensor_args.to_device([0.0]*points.shape[-2]),
                    env_query_idx=env_query_idx,
                    compute_esdf=True)
                valid_flag = (coll_result + points[...,-1]).squeeze(1) < 1e-5
                ind = torch.nonzero(valid_flag, as_tuple=True)
                valid_num = torch.bincount(ind[0])
                self.surface_sample_positions = points[ind[0], 0, ind[1], :3]
                self.surface_sample_normals = normals[ind[0], ind[1], :]
                self.surface_sample_ind_upper = torch.cumsum(valid_num, dim=0)
                self.surface_sample_ind_lower = torch.cat([self.surface_sample_ind_upper[0:1] * 0, self.surface_sample_ind_upper[:-1]])
            else:
                self.surface_sample_positions = points[:, 0, :, :3]
                self.surface_sample_normals = normals
                valid_num = self.tensor_args.to_device(torch.ones(normals.shape[0])*normals.shape[1])
                self.surface_sample_ind_upper = torch.cumsum(valid_num, dim=0)
                self.surface_sample_ind_lower = torch.cat([self.surface_sample_ind_upper[0:1] * 0, self.surface_sample_ind_upper[:-1]])

        # For mesh-mesh evaluation in grasp_cost
        if load_mesh:
            self._contact_scene_convex_meshes = []
            self._contact_scene_mesh_num = []
            self._contact_scene_obb_param = []
            for i, world in enumerate(world_model):
                om = world.mesh[self._eval_mesh_idx[i]]
                for convex_part in get_cached_contact_convex_parts(om.urdf_path):
                    obj_scale = np.asarray(convex_part.local_scale, dtype=np.float64) * self._contact_mesh_scales[i]
                    hppmesh = get_cached_contact_convex_mesh(convex_part.file_path, obj_scale)
                    obb_param = get_cached_contact_obb_param(
                        convex_part.file_path,
                        obj_scale,
                        cache_mode=contact_obb_cache_mode,
                    )
                    self._contact_scene_convex_meshes.append(hppmesh)
                    self._contact_scene_mesh_num.append(i)
                    self._contact_scene_obb_param.append(obb_param)
            self._contact_scene_obb_param = self.tensor_args.to_device(torch.stack(self._contact_scene_obb_param))
            self._contact_scene_mesh_num = self.tensor_args.to_device(np.stack(self._contact_scene_mesh_num)).long()
            self._contact_scene_poses = self._contact_mesh_poses[self._contact_scene_mesh_num]

        return 

    @staticmethod
    def _contact_surface_point_request(sample_params: Optional[Dict]) -> Tuple[bool, int]:
        """Resolve the optional broad contact-surface point tensor request.

        Args:
            sample_params: Optional ``seeder_cfg.obj_sample`` dictionary passed
                through ``WorldCollisionConfig.contact_obj_sample_params``.

        Returns:
            Tuple ``(enabled, sample_num)``. ``enabled`` indicates whether
            ``load_contact_obj()`` should materialize ``_contact_mesh_surface_points``;
            ``sample_num`` is the configured number of raw object surface points.
        """
        if sample_params is None:
            return True, 4096

        mode = str(sample_params.get("contact_surface_points_mode", "always")).strip().lower()
        if mode in {"", "always", "on", "legacy"}:
            enabled = True
        elif mode in {"off", "none", "false", "0"}:
            enabled = False
        elif mode == "auto":
            sampling_mode = str(sample_params.get("sampling_mode", "random")).strip().lower()
            enabled = sampling_mode == "ultradexgrasp"
        else:
            raise ValueError(
                "Unsupported contact_surface_points_mode="
                f"{sample_params.get('contact_surface_points_mode')!r}. "
                "Expected 'always', 'auto', or 'off'."
            )

        sample_num = int(sample_params.get("contact_surface_sample_num", 4096))
        if sample_num < 0:
            raise ValueError(f"contact_surface_sample_num must be >= 0, got {sample_num}")
        return enabled and sample_num > 0, sample_num
    
    def load_contact_robot(self, robot_meshes: List[Mesh]):
        self._contact_robot_convex_meshes = []
        self._contact_robot_sphere_param = []
        self._contact_robot_offset_pose = []
        for rm in robot_meshes:
            robot_scale = np.array([1., 1., 1.]) * rm.scale if rm.scale is not None else np.array([1., 1., 1.])
            robot_hppmesh = loadConvexMeshCpp(rm.file_path, robot_scale)
            sphere_param = getBoundingPrimitive(rm.file_path, robot_scale, 'sphere')
            self._contact_robot_convex_meshes.append(robot_hppmesh)
            self._contact_robot_sphere_param.append(sphere_param)
            self._contact_robot_offset_pose.append(rm.pose)
        self._contact_robot_offset_pose = self.tensor_args.to_device(np.stack(self._contact_robot_offset_pose))
        self._contact_robot_sphere_param = self.tensor_args.to_device(np.stack(self._contact_robot_sphere_param))
        return
    
    def load_contact_robot_pc(self, robot_meshes: List[Mesh]):
        self._contact_robot_offset_pose = []
        self._contact_robot_pc = []
        for rm in robot_meshes:
            robot_scale = np.array([1., 1., 1.]) * rm.scale if rm.scale is not None else np.array([1., 1., 1.])
            tm_mesh = trimesh.convex.convex_hull(trimesh.load(rm.file_path, force='mesh'))
            tm_mesh.vertices *= robot_scale
            surface_points, _ = trimesh.sample.sample_surface_even(tm_mesh, 1024)
            self._contact_robot_offset_pose.append(rm.pose)
            self._contact_robot_pc.append(surface_points)
        self._contact_robot_offset_pose = self.tensor_args.to_device(np.stack(self._contact_robot_offset_pose))
        self._contact_robot_pc = self.tensor_args.to_device(np.stack(self._contact_robot_pc))
        return 
    
    def load_collision_model(
        self,
        world_model: WorldConfig,
        env_idx: int = 0,
        load_obb_obs: bool = True,
        fix_cache_reference: bool = False,
    ):
        """Load world obstacles into collision checker.

        Args:
            world_model: Obstacles to load.
            env_idx: Environment index to load obstacles into.
            load_obb_obs: Load OBB (cuboid) obstacles.
            fix_cache_reference: If True, throws error if number of obstacles is greater than
                cache. If False, creates a larger cache. Note that when using collision checker
                inside a recorded cuda graph, recreating the cache will break the graph as the
                reference pointer to the cache will change.
        """
        max_nmesh = len(world_model.mesh)
        if max_nmesh > 0:
            if self._mesh_tensor_list is None or self._mesh_tensor_list[0].shape[1] < max_nmesh:
                log_warn("Creating new Mesh cache: " + str(max_nmesh))
                self._create_mesh_cache(max_nmesh)

            # load all meshes as a batch:
            name_list, key_list, w_mid, w_inv_pose = self._load_batch_mesh_to_warp(world_model.mesh)
            self._mesh_tensor_list[0][env_idx, :max_nmesh] = w_mid
            self._mesh_tensor_list[1][env_idx, :max_nmesh, :7] = w_inv_pose
            self._mesh_tensor_list[2][env_idx, :max_nmesh] = 1
            self._mesh_tensor_list[2][env_idx, max_nmesh:] = 0

            self._env_mesh_names[env_idx][:max_nmesh] = name_list
            self._env_mesh_names[env_idx][max_nmesh:] = [
                None for _ in self._env_mesh_names[env_idx][max_nmesh:]
            ]
            self._env_mesh_cache_keys[env_idx][:max_nmesh] = key_list
            self._env_mesh_cache_keys[env_idx][max_nmesh:] = [
                None for _ in self._env_mesh_cache_keys[env_idx][max_nmesh:]
            ]
            self._env_n_mesh[env_idx] = max_nmesh

            self.collision_types["mesh"] = True
        if load_obb_obs:
            super().load_collision_model(
                world_model, env_idx, fix_cache_reference=fix_cache_reference
            )
        # else:
        #     self.world_model = world_model

    def load_batch_collision_model(self, world_config_list: List[WorldConfig]):
        """Load multiple world obstacles into collision checker across environments.

        Args:
            world_config_list: List of obstacles to load.
        """
        max_nmesh = max([len(x.mesh) for x in world_config_list])
        reset_buffers = False
        if len(world_config_list) != self.n_envs:
            self.n_envs = len(world_config_list)
            reset_buffers = True
        if self._mesh_tensor_list is not None and self._mesh_tensor_list[0].shape[1] < max_nmesh:
            reset_buffers = True
        # Mesh cache stores one row per collision environment, so rebuild it whenever
        # the environment count changes or the current mesh capacity is too small.
        if self._mesh_tensor_list is None or reset_buffers:
            log_info("Creating Mesh cache" + str(max_nmesh))
            self._create_mesh_cache(max_nmesh)
                      
        for env_idx, world_model in enumerate(world_config_list):
            self.load_collision_model(world_model, env_idx=env_idx, load_obb_obs=False)
        super().load_batch_collision_model(world_config_list)

    def _load_mesh_to_warp(self, mesh: Mesh) -> WarpMeshData:
        """Load cuRobo mesh into warp.

        Args:
            mesh: mesh instance to load.

        Returns:
            loaded mesh data.
        """
        verts, faces = mesh.get_mesh_data()
        verts = np.ascontiguousarray(np.asarray(verts))
        faces = np.ascontiguousarray(np.asarray(np.ravel(faces)))
        v = wp.array(verts, dtype=wp.vec3, device=self._wp_device)
        f = wp.array(faces, dtype=int, device=self._wp_device)
        new_mesh = wp.Mesh(points=v, indices=f)
        return WarpMeshData(
            mesh.name,
            new_mesh.id,
            v,
            f,
            new_mesh,
            cache_bytes=int(verts.nbytes + faces.nbytes),
        )

    def _mesh_geometry_cache_key(self, mesh: Mesh) -> Tuple[Hashable, ...]:
        """Build a pose- and scene-name-independent key for a Warp mesh.

        Args:
            mesh: Mesh obstacle whose geometry will be loaded into Warp.

        Returns:
            Hashable key that identifies scaled object-local geometry. The scene
            pose and obstacle name are intentionally excluded so many poses of
            the same mesh can share one Warp mesh id.
        """
        scale = self._mesh_scale_tuple(mesh.scale)
        if mesh.file_path is not None:
            return ("file", self._file_signature(mesh.file_path), scale)
        if mesh.vertices is not None and mesh.faces is not None:
            return ("arrays", self._array_digest(mesh.vertices), self._array_digest(mesh.faces), scale)
        verts, faces = mesh.get_mesh_data()
        return ("generated", type(mesh).__name__, self._array_digest(verts), self._array_digest(faces), scale)

    @staticmethod
    def _mesh_scale_tuple(value) -> Tuple[float, float, float]:
        """Normalize a mesh scale into an xyz tuple for cache keys.

        Args:
            value: Optional scalar or xyz-like mesh scale.

        Returns:
            Three-float tuple describing the final mesh scale applied before
            Warp loading.
        """
        if value is None:
            return (1.0, 1.0, 1.0)
        scale = np.asarray(value, dtype=np.float64).reshape(-1)
        if scale.size == 1:
            return (float(scale[0]), float(scale[0]), float(scale[0]))
        if scale.size != 3:
            raise ValueError(f"Mesh scale must be scalar or xyz, got shape {scale.shape}")
        return (float(scale[0]), float(scale[1]), float(scale[2]))

    @staticmethod
    def _file_signature(file_path: str) -> Tuple[Hashable, ...]:
        """Build a lightweight file signature for a mesh cache key.

        Args:
            file_path: Absolute or relative mesh file path.

        Returns:
            Tuple containing real path, byte size, and nanosecond mtime.
        """
        real_path = os.path.realpath(file_path)
        stat = os.stat(real_path)
        return (real_path, int(stat.st_size), int(stat.st_mtime_ns))

    @staticmethod
    def _array_digest(value) -> Tuple[Hashable, ...]:
        """Build a stable digest for in-memory mesh geometry arrays.

        Args:
            value: Array-like vertices or faces.

        Returns:
            Tuple containing shape, dtype, and a SHA1 digest of the raw bytes.
        """
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha1(array.view(np.uint8)).hexdigest()
        return (tuple(array.shape), str(array.dtype), digest)

    def _load_mesh_into_cache(
        self,
        mesh: Mesh,
        cache_key: Optional[Tuple[Hashable, ...]] = None,
        protected_keys: Optional[Set[Tuple[Hashable, ...]]] = None,
    ) -> WarpMeshData:
        """Load cuRobo mesh into cache.

        Args:
            mesh: mesh instance to load.
            cache_key: Optional precomputed pose-independent geometry key.
            protected_keys: Geometry keys that must not be evicted during this load.

        Returns:
            loaded mesh data.
        """
        cache_key = self._mesh_geometry_cache_key(mesh) if cache_key is None else cache_key
        if cache_key not in self._wp_mesh_cache:
            # load mesh into cache:
            self._wp_mesh_cache[cache_key] = self._load_mesh_to_warp(mesh)
            self._wp_mesh_cache_current_bytes += int(self._wp_mesh_cache[cache_key].cache_bytes)
            self._trim_wp_mesh_cache(protected_keys)
        else:
            log_info("Object already in warp cache, using existing instance for: " + mesh.name)
            self._wp_mesh_cache.move_to_end(cache_key)
        return self._wp_mesh_cache[cache_key]

    def _load_batch_mesh_to_warp(self, mesh_list: List[Mesh]) -> List:
        """Load multiple meshes into warp.

        Args:
            mesh_list: List of meshes to load.

        Returns:
            List of mesh names, geometry cache keys, mesh ids, and inverse poses.
        """
        # First load all verts and faces:
        name_list = []
        key_list = []
        pose_list = []
        id_list = torch.zeros((len(mesh_list)), device=self.tensor_args.device, dtype=torch.int64)
        cache_keys = [self._mesh_geometry_cache_key(mesh) for mesh in mesh_list]
        protected_keys = self._get_active_wp_mesh_cache_keys().union(cache_keys)
        for i, m_idx in enumerate(mesh_list):
            m_data = self._load_mesh_into_cache(
                m_idx,
                cache_key=cache_keys[i],
                protected_keys=protected_keys,
            )
            pose_list.append(m_idx.pose)

            id_list[i] = m_data.m_id
            name_list.append(m_idx.name)
            key_list.append(cache_keys[i])
        pose_buffer = Pose.from_batch_list(pose_list, self.tensor_args)
        inv_pose_buffer = pose_buffer.inverse()
        return name_list, key_list, id_list, inv_pose_buffer.get_pose_vector()

    def add_mesh(self, new_mesh: Mesh, env_idx: int = 0):
        """Add a mesh to the world.

        Args:
            new_mesh: Mesh to add.
            env_idx: Environment index to add mesh to.
        """
        if self._env_n_mesh[env_idx] >= self._mesh_tensor_list[0].shape[1]:
            log_error(
                "Cannot add new mesh as we are at mesh cache limit, increase cache limit in WorldMeshCollision"
            )
            return

        cache_key = self._mesh_geometry_cache_key(new_mesh)
        wp_mesh_data = self._load_mesh_into_cache(
            new_mesh,
            cache_key=cache_key,
            protected_keys=self._get_active_wp_mesh_cache_keys().union({cache_key}),
        )

        # get mesh pose:
        w_obj_pose = Pose.from_list(new_mesh.pose, self.tensor_args)
        # add loaded mesh into scene:

        curr_idx = self._env_n_mesh[env_idx]
        self._mesh_tensor_list[0][env_idx, curr_idx] = wp_mesh_data.m_id
        self._mesh_tensor_list[1][env_idx, curr_idx, :7] = w_obj_pose.inverse().get_pose_vector()
        self._mesh_tensor_list[2][env_idx, curr_idx] = 1
        self._env_mesh_names[env_idx][curr_idx] = new_mesh.name
        self._env_mesh_cache_keys[env_idx][curr_idx] = cache_key
        self._env_n_mesh[env_idx] = curr_idx + 1

    def get_mesh_idx(
        self,
        name: str,
        env_idx: int = 0,
    ) -> int:
        """Get index of mesh with given name.

        Args:
            name: Name of the mesh.
            env_idx: Environment index to search in.

        Returns:
            Mesh index.
        """
        if name not in self._env_mesh_names[env_idx]:
            log_error("Obstacle with name: " + name + " not found in current world", exc_info=True)
        return self._env_mesh_names[env_idx].index(name)

    def create_collision_cache(
        self,
        mesh_cache: Optional[int] = None,
        obb_cache: Optional[int] = None,
        n_envs: Optional[int] = None,
    ):
        """Create cache of obstacles.

        Args:
            mesh_cache: Number of mesh obstacles to cache.
            obb_cache: Number of OBB (cuboid) obstacles to cache.
            n_envs: Number of environments to cache.
        """
        if n_envs is not None:
            self.n_envs = n_envs
        if mesh_cache is not None:
            self._create_mesh_cache(mesh_cache)
        if obb_cache is not None:
            self._create_obb_cache(obb_cache)

    def _create_mesh_cache(self, mesh_cache: int):
        """Create cache for mesh obstacles.

        Args:
            mesh_cache: Number of mesh obstacles to cache.
        """
        # create cache to store meshes, mesh poses and inverse poses
        
        self.cache['mesh'] = mesh_cache
        self._env_n_mesh = torch.zeros(
            (self.n_envs), device=self.tensor_args.device, dtype=torch.int32
        )

        obs_enable = torch.zeros(
            (self.n_envs, mesh_cache), dtype=torch.uint8, device=self.tensor_args.device
        )
        obs_inverse_pose = torch.zeros(
            (self.n_envs, mesh_cache, 8),
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        obs_ids = torch.zeros(
            (self.n_envs, mesh_cache), device=self.tensor_args.device, dtype=torch.int64
        )

        # warp requires uint64 for mesh indices, supports conversion from int64 to uint64
        self._mesh_tensor_list = [
            obs_ids,
            obs_inverse_pose,
            obs_enable,
        ]  # 0=mesh idx, 1=pose, 2=mesh enable
        self.collision_types["mesh"] = True  # TODO: enable this after loading first mesh
        self._env_mesh_names = [[None for _ in range(mesh_cache)] for _ in range(self.n_envs)]
        self._env_mesh_cache_keys = [[None for _ in range(mesh_cache)] for _ in range(self.n_envs)]
        if self._warp_mesh_cache_enabled():
            self._trim_wp_mesh_cache(set())
        else:
            self._clear_wp_mesh_cache()

    def update_mesh_pose(
        self,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        name: Optional[str] = None,
        env_obj_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Update pose of mesh.

        Args:
            w_obj_pose: Pose of the mesh in world frame.
            obj_w_pose: Inverse pose of the mesh.
            name: Name of mesh to update.
            env_obj_idx: Index of mesh in environment. If name is given, this is ignored.
            env_idx: Environment index to update mesh in.
        """
        w_inv_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)

        if name is not None:
            obs_idx = self.get_mesh_idx(name, env_idx)
            self._mesh_tensor_list[1][env_idx, obs_idx, :7] = w_inv_pose.get_pose_vector()
        elif env_obj_idx is not None:
            self._mesh_tensor_list[1][env_idx, env_obj_idx, :7] = w_inv_pose.get_pose_vector()
        else:
            log_error("name or env_obj_idx needs to be given to update mesh pose")

    def update_mesh_from_warp(
        self,
        warp_mesh_idx: int,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        obj_idx: int = 0,
        env_idx: int = 0,
        name: Optional[str] = None,
    ):
        """Update mesh information in world cache.

        Args:
            warp_mesh_idx: New warp mesh index.
            w_obj_pose: Pose of the mesh in world frame.
            obj_w_pose: Invserse pose of the mesh. Not required if w_obj_pose is given.
            obj_idx: Index of mesh in environment. If name is given, this is ignored.
            env_idx: Environment index to update mesh in.
            name: Name of mesh to update.
        """
        if name is not None:
            obj_idx = self.get_mesh_idx(name, env_idx)

        if obj_idx >= self._mesh_tensor_list[0][env_idx].shape[0]:
            log_error("Out of cache memory")
        w_inv_pose = self._get_obstacle_poses(w_obj_pose, obj_w_pose)

        self._mesh_tensor_list[0][env_idx, obj_idx] = warp_mesh_idx
        self._mesh_tensor_list[1][env_idx, obj_idx] = w_inv_pose
        self._mesh_tensor_list[2][env_idx, obj_idx] = 1
        self._env_mesh_names[env_idx][obj_idx] = name
        self._env_mesh_cache_keys[env_idx][obj_idx] = None
        if self._env_n_mesh[env_idx] <= obj_idx:
            self._env_n_mesh[env_idx] = obj_idx + 1

    def update_obstacle_pose(
        self,
        name: str,
        w_obj_pose: Pose,
        env_idx: int = 0,
        update_cpu_reference: bool = False,
    ):
        """Update pose of obstacle.

        Args:
            name: Name of the obstacle.
            w_obj_pose: Pose of obstacle in world frame.
            env_idx: Environment index to update obstacle in.
            update_cpu_reference: If True, updates the CPU reference with the new pose. This is
                useful for debugging and visualization. Only supported for env_idx=0.
        """
        if self._env_mesh_names is not None and name in self._env_mesh_names[env_idx]:
            self.update_mesh_pose(name=name, w_obj_pose=w_obj_pose, env_idx=env_idx)
            if update_cpu_reference:
                self.update_obstacle_pose_in_world_model(name, w_obj_pose, env_idx)
        else:
            super().update_obstacle_pose(
                name=name,
                w_obj_pose=w_obj_pose,
                env_idx=env_idx,
                update_cpu_reference=update_cpu_reference,
            )

    def enable_obstacle(
        self,
        name: str,
        enable: bool = True,
        env_idx: int = 0,
    ):
        """Enable/Disable object in collision checking functions.

        Args:
            name: Name of the obstacle to enable.
            enable: True to enable, False to disable.
            env_idx: Index of the environment to enable the obstacle in.
        """
        if self._env_mesh_names is not None and name in self._env_mesh_names[env_idx]:
            self.enable_mesh(enable, name, None, env_idx)
        elif self._env_obbs_names is not None and name in self._env_obbs_names[env_idx]:
            self.enable_obb(enable, name, None, env_idx)
        else:
            log_error("Obstacle not found in world model: " + name)

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        """Get names of all obstacles in the environment.

        Args:
            env_idx: Environment index to get obstacles from.

        Returns:
            List of obstacle names.
        """
        base_obstacles = super().get_obstacle_names(env_idx)
        return self._env_mesh_names[env_idx] + base_obstacles

    def enable_mesh(
        self,
        enable: bool = True,
        name: Optional[str] = None,
        env_mesh_idx: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """Enable/Disable mesh in collision checking functions.

        Args:
            enable: True to enable, False to disable.
            name: Name of the mesh to enable.
            env_mesh_idx: Index of the mesh in environment. If name is given, this is ignored.
            env_idx: Environment index to enable the mesh in.
        """
        if env_mesh_idx is not None:
            self._mesh_tensor_list[2][env_mesh_idx] = int(enable)  # enable == 1
        else:
            # find index of given name:
            obs_idx = self.get_mesh_idx(name, env_idx)
            self._mesh_tensor_list[2][env_idx, obs_idx] = int(enable)

    def _get_sdf(
        self,
        query_spheres: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        contact_distance = None, 
        return_loss: bool = False,
        compute_esdf: bool = False,
    ) -> torch.Tensor:
        """Compute signed distance for mesh obstacles using warp kernel.

        Args:
            query_spheres: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Weight to scale the collision cost.
            activation_distance: Distance outside the object to start computing the cost.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: If the returned tensor will be scaled or changed before calling backward,
                set this to True. If the returned tensor will be used directly through addition,
                set this to False.
            compute_esdf: Compute Euclidean signed distance instead of collision cost. When True,
                the returned tensor will be the signed distance with positive values inside an
                obstacle and negative values outside obstacles.

        Returns:
            Collision cost between query spheres and world obstacles.
        """
        d = SdfMeshWarpPy.apply(
            query_spheres,
            collision_query_buffer.mesh_collision_buffer.distance_buffer,
            collision_query_buffer.mesh_collision_buffer.grad_distance_buffer,
            collision_query_buffer.mesh_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            self._mesh_tensor_list[0],
            self._mesh_tensor_list[1],
            self._mesh_tensor_list[2],
            self._env_n_mesh,
            self.max_distance,
            self._contact_mesh_warp_idx,
            env_query_idx,
            contact_distance,
            return_loss,
            compute_esdf,
        )
        return d

    def _get_swept_sdf(
        self,
        query_spheres,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        contact_distance=None,
        enable_speed_metric=False,
        env_query_idx=None,
        return_loss: bool = False,
    ):
        """Compute signed distance for mesh obstacles using warp kernel.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Collision cost weight.
            activation_distance: Distance outside the object to start computing the cost. A smooth
                scaling is applied to the cost starting from this distance. See
                :ref:`research_page` for more details.
            speed_dt: Length of time (seconds) to use when calculating the speed of the sphere
                using finite difference.
            sweep_steps: Number of steps to sweep the sphere along the trajectory. More steps will
                allow for catching small obstacles, taking more time to compute.
            enable_speed_metric: True will scale the collision cost by the speed of the sphere.
                This has the effect of slowing down the robot when near obstacles. This also has
                shown to improve convergence from poor initialization.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: If the returned tensor will be scaled or changed before calling backward,
                set this to True. If the returned tensor will be used directly through addition,
                set this to False.

        Returns:
            Collision cost between trajectory of spheres and world obstacles.
        """
        d = SweptSdfMeshWarpPy.apply(
            query_spheres,
            collision_query_buffer.mesh_collision_buffer.distance_buffer,
            collision_query_buffer.mesh_collision_buffer.grad_distance_buffer,
            collision_query_buffer.mesh_collision_buffer.sparsity_index_buffer,
            weight,
            activation_distance,
            speed_dt,
            self._mesh_tensor_list[0],
            self._mesh_tensor_list[1],
            self._mesh_tensor_list[2],
            self._env_n_mesh,
            self.max_distance,
            self._contact_mesh_warp_idx,
            contact_distance,
            sweep_steps,
            enable_speed_metric,
            env_query_idx,
            return_loss,
        )
        return d
    
    def get_mesh_contact_pdn(
        self,
        contact_robot_pose: torch.Tensor,
        perturb: torch.Tensor,
        env_query_idx: torch.Tensor,
        dist_upper_bound: torch.Tensor,
    ):
        
        position, distance, normal, debug_posi, debug_normal = ContactPDNConvexMesh.apply(
            contact_robot_pose,
            self._contact_robot_offset_pose,
            self._contact_scene_poses,
            self._contact_robot_convex_meshes,
            self._contact_scene_convex_meshes,
            self._contact_robot_sphere_param,
            self._contact_scene_obb_param,
            perturb,
            self._contact_scene_mesh_num,
            env_query_idx,
            dist_upper_bound
        )
        return position, distance, normal, debug_posi, debug_normal

    def get_mesh_robot_self_penetration(
        self, 
        robot_link_poses: torch.Tensor, 
        self_collision_link_mask: torch.Tensor,
        mesh_order: List,
    ):
        inv_mesh_order = [mesh_order.index(i) for i in sorted(mesh_order)]
        return RobotSelfPenetration(
            robot_link_poses, 
            self._contact_robot_offset_pose[inv_mesh_order], 
            [self._contact_robot_convex_meshes[midx] for midx in inv_mesh_order],
            self_collision_link_mask,
        )
    
    def get_sphere_contact_pdn(
        self,
        contact_robot_sphere: torch.Tensor,
        contact_query_buffer: ContactBuffer,
        perturb: torch.Tensor,
        env_query_idx: torch.Tensor,
    ):
        position, distance, normal, debug_posi, debug_normal = ContactPDNMeshWarpPy.apply(
            contact_robot_sphere,
            perturb,
            contact_query_buffer.pos_buf,
            contact_query_buffer.dist_buf,
            contact_query_buffer.normal_buf,
            contact_query_buffer.debug_posi,
            contact_query_buffer.debug_normal,
            contact_query_buffer.grad_pos_buf,
            contact_query_buffer.grad_dist_buf,
            contact_query_buffer.grad_normal_buf,
            self._contact_mesh_warp_idx,
            self._contact_mesh_inv_poses,
            env_query_idx,
        )
        return position, distance, normal, debug_posi, debug_normal

    def get_sphere_distance(
        self,
        query_sphere: torch.Tensor,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        contact_distance: torch.Tensor = None, 
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):
        """Compute the signed distance between query spheres and world obstacles.

        This distance can be used as a collision cost for optimization.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Buffer to store collision query results.
            weight: Weight of the collision cost.
            activation_distance: Distance outside the object to start computing the cost.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: If the returned tensor will be scaled or changed before calling backward,
                set this to True. If the returned tensor will be used directly through addition,
                set this to False.
            sum_collisions: Sum the collision cost across all obstacles. This variable is currently
                not passed to the underlying CUDA kernel as setting this to False caused poor
                performance.
            compute_esdf: Compute Euclidean signed distance instead of collision cost. When True,
                the returned tensor will be the signed distance with positive values inside an
                obstacle and negative values outside obstacles.

        Returns:
            Signed distance between query spheres and world obstacles.
        """
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                activation_distance=activation_distance,
                env_query_idx=env_query_idx,
                return_loss=return_loss,
                sum_collisions=sum_collisions,
                compute_esdf=compute_esdf,
            )

        d = self._get_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            contact_distance=contact_distance,
            return_loss=return_loss,
            compute_esdf=compute_esdf,
        )

        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d
        d_prim = super().get_sphere_distance(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
            sum_collisions=sum_collisions,
            compute_esdf=compute_esdf,
        )
        if compute_esdf:
            d_val = torch.maximum(d.view(d_prim.shape), d_prim)
        else:
            d_val = d.view(d_prim.shape) + d_prim
        return d_val

    def get_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        contact_distance: torch.Tensor = None, 
        env_query_idx=None,
        return_loss=False,
        **kwargs,
    ):
        """Compute binary collision between query spheres and world obstacles.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Weight to scale the collision cost.
            activation_distance: Distance outside the object to start computing the cost.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: True is not supported for binary classification. Set to False.

        Returns:
            Tensor with binary collision results.
        """
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                activation_distance=activation_distance,
                env_query_idx=env_query_idx,
                return_loss=return_loss,
            )

        d = self._get_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            contact_distance=contact_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )

        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d

        d_prim = super().get_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight,
            activation_distance=activation_distance,
            env_query_idx=env_query_idx,
            return_loss=return_loss,
        )
        d_val = d.view(d_prim.shape) + d_prim

        return d_val

    def get_swept_sphere_distance(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        sweep_steps: int,
        contact_distance: torch.Tensor = None, 
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        sum_collisions: bool = True,
    ):
        """Compute the signed distance between trajectory of spheres and world obstacles.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Collision cost weight.
            activation_distance: Distance outside the object to start computing the cost. A smooth
                scaling is applied to the cost starting from this distance. See
                :ref:`research_page` for more details.
            speed_dt: Length of time (seconds) to use when calculating the speed of the sphere
                using finite difference.
            sweep_steps: Number of steps to sweep the sphere along the trajectory. More steps will
                allow for catching small obstacles, taking more time to compute.
            enable_speed_metric: True will scale the collision cost by the speed of the sphere.
                This has the effect of slowing down the robot when near obstacles. This also has
                shown to improve convergence from poor initialization.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: If the returned tensor will be scaled or changed before calling backward,
                set this to True. If the returned tensor will be used directly through addition,
                set this to False.
            sum_collisions: Sum the collision cost across all obstacles. This variable is currently
                not passed to the underlying CUDA kernel as setting this to False caused poor
                performance.

        Returns:
            Collision cost between trajectory of spheres and world obstacles.
        """
        # log_warn("Swept: Mesh + Primitive Collision Checking is experimental")
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_swept_sphere_distance(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                env_query_idx=env_query_idx,
                sweep_steps=sweep_steps,
                activation_distance=activation_distance,
                speed_dt=speed_dt,
                enable_speed_metric=enable_speed_metric,
                return_loss=return_loss,
                sum_collisions=sum_collisions,
            )

        d = self._get_swept_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            activation_distance=activation_distance,
            contact_distance=contact_distance,
            speed_dt=speed_dt,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d

        d_prim = super().get_swept_sphere_distance(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
            sum_collisions=sum_collisions,
        )
        d_val = d.view(d_prim.shape) + d_prim

        return d_val

    def get_swept_sphere_collision(
        self,
        query_sphere,
        collision_query_buffer: CollisionQueryBuffer,
        weight: torch.Tensor,
        sweep_steps,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        enable_speed_metric=False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        """Get binary collision between trajectory of spheres and world obstacles.

        Args:
            query_sphere: Input tensor with query spheres [batch, horizon, number of spheres, 4].
                With [x, y, z, radius] as the last column for each sphere.
            collision_query_buffer: Collision query buffer to store the results.
            weight: Collision cost weight.
            activation_distance: Distance outside the object to start computing the cost. A smooth
                scaling is applied to the cost starting from this distance. See
                :ref:`research_page` for more details.
            speed_dt: Length of time (seconds) to use when calculating the speed of the sphere
                using finite difference. This is not used.
            sweep_steps: Number of steps to sweep the sphere along the trajectory. More steps will
                allow for catching small obstacles, taking more time to compute.
            enable_speed_metric: True will scale the collision cost by the speed of the sphere.
                This has the effect of slowing down the robot when near obstacles. This also has
                shown to improve convergence from poor initialization. This is not used.
            env_query_idx: Environment index for each batch of query spheres.
            return_loss: This is not supported for binary classification. Set to False.

        Returns:
            Collision value between trajectory of spheres and world obstacles.
        """
        if "mesh" not in self.collision_types or not self.collision_types["mesh"]:
            return super().get_swept_sphere_collision(
                query_sphere,
                collision_query_buffer,
                weight=weight,
                env_query_idx=env_query_idx,
                sweep_steps=sweep_steps,
                activation_distance=activation_distance,
                speed_dt=speed_dt,
                enable_speed_metric=enable_speed_metric,
                return_loss=return_loss,
            )
        d = self._get_swept_sdf(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )
        if "primitive" not in self.collision_types or not self.collision_types["primitive"]:
            return d

        d_prim = super().get_swept_sphere_collision(
            query_sphere,
            collision_query_buffer,
            weight=weight,
            env_query_idx=env_query_idx,
            sweep_steps=sweep_steps,
            activation_distance=activation_distance,
            speed_dt=speed_dt,
            enable_speed_metric=enable_speed_metric,
            return_loss=return_loss,
        )
        d_val = d.view(d_prim.shape) + d_prim

        return d_val

    def clear_cache(self):
        """Delete all cuboid and mesh obstacles from the world."""
        if self._mesh_tensor_list is not None:
            self._mesh_tensor_list[2][:] = 0
        if self._env_n_mesh is not None:
            self._env_n_mesh[:] = 0
        if self._env_mesh_names is not None:
            self._env_mesh_names = [
                [None for _ in range(self.cache["mesh"])] for _ in range(self.n_envs)
            ]
        if self._env_mesh_cache_keys is not None:
            self._env_mesh_cache_keys = [
                [None for _ in range(self.cache["mesh"])] for _ in range(self.n_envs)
            ]
        if self._warp_mesh_cache_enabled():
            self._trim_wp_mesh_cache(set())
        else:
            self._clear_wp_mesh_cache()

        super().clear_cache()

    def warp_mesh_cache_info(self) -> Dict[str, int]:
        """Return current persistent Warp mesh cache statistics.

        Args:
            None.

        Returns:
            Dictionary containing current size, byte usage, and configured limits.
        """
        return {
            "size": len(self._wp_mesh_cache),
            "current_bytes": int(self._wp_mesh_cache_current_bytes),
            "max_entries": int(self._warp_mesh_cache_max_entries),
            "max_bytes": int(self._warp_mesh_cache_max_bytes),
            "enabled": int(self._warp_mesh_cache_enabled()),
        }

    def _warp_mesh_cache_enabled(self) -> bool:
        """Report whether persistent Warp mesh caching is enabled.

        Args:
            None.

        Returns:
            ``True`` when at least one positive cache bound is configured.
        """
        return self._warp_mesh_cache_max_entries > 0 or self._warp_mesh_cache_max_bytes > 0

    def _clear_wp_mesh_cache(self) -> None:
        """Drop all persistent Warp mesh cache entries.

        Args:
            None.

        Returns:
            None.
        """
        self._wp_mesh_cache = OrderedDict()
        self._wp_mesh_cache_current_bytes = 0

    def _get_active_wp_mesh_cache_keys(self) -> Set[Tuple[Hashable, ...]]:
        """Collect geometry keys referenced by the current env mesh tensors.

        Args:
            None.

        Returns:
            Set of active mesh geometry keys that must not be evicted.
        """
        active_keys: Set[Tuple[Hashable, ...]] = set()
        if self._env_mesh_cache_keys is None:
            return active_keys
        for env_keys in self._env_mesh_cache_keys:
            for key in env_keys:
                if key is not None:
                    active_keys.add(key)
        return active_keys

    def _trim_wp_mesh_cache(self, protected_keys: Optional[Set[Tuple[Hashable, ...]]] = None) -> None:
        """Evict least-recently-used inactive Warp meshes until limits are satisfied.

        Args:
            protected_keys: Geometry keys that are currently in use and must not be evicted.

        Returns:
            None.
        """
        if not self._wp_mesh_cache:
            self._wp_mesh_cache_current_bytes = 0
            return
        if not self._warp_mesh_cache_enabled():
            return

        protected = set() if protected_keys is None else set(protected_keys)
        while True:
            over_entries = (
                self._warp_mesh_cache_max_entries > 0
                and len(self._wp_mesh_cache) > self._warp_mesh_cache_max_entries
            )
            over_bytes = (
                self._warp_mesh_cache_max_bytes > 0
                and self._wp_mesh_cache_current_bytes > self._warp_mesh_cache_max_bytes
            )
            if not over_entries and not over_bytes:
                break

            evicted = False
            for cache_key in list(self._wp_mesh_cache.keys()):
                if cache_key in protected:
                    continue
                cache_entry = self._wp_mesh_cache.pop(cache_key)
                self._wp_mesh_cache_current_bytes -= int(cache_entry.cache_bytes)
                evicted = True
                break
            if not evicted:
                break
        self._wp_mesh_cache_current_bytes = max(0, int(self._wp_mesh_cache_current_bytes))

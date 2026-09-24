from typing import Dict, List
import os
import torch
import numpy as np
import math

from curobo.util.sample_lib import HaltonGenerator
from curobo.util.tensor_util import normalize_vector
from curobo.util.logger import log_warn
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.basic_transform import euler_angles_to_matrix, matrix_from_rot_repre, matrix_to_quaternion
from curobo.geom.sdf.world import WorldCollision
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


class HeurGraspSeedGenerator:
    def __init__(
        self,
        seeder_cfg: Dict,
        full_robot_model: CudaRobotModel,
        ik_solver: IKSolver,
        world_coll_checker: WorldCollision,
        obj_lst: List,
        tensor_args: TensorDeviceType,
    ):
        self.tensor_args = tensor_args
        all_dof = full_robot_model.dof
        use_root_pose = full_robot_model.use_root_pose
        self.full_robot_model = full_robot_model
        self.ik_solver = ik_solver
        self.tr_link_names = full_robot_model.transfered_link_name
        if ik_solver is not None:
            self.replace_ind = ik_solver.kinematics.kinematics_config.get_replace_index(
                full_robot_model.kinematics_config
            )

        self.b_dummy_arm = (full_robot_model.dummy_trans_joints is not None) and (
            full_robot_model.dummy_rot_joints is not None
        )

        # Sampling mode configuration
        self.sampling_mode = seeder_cfg.get("sampling_mode", "random")

        if use_root_pose:
            assert len(self.tr_link_names) == 1
            self.q_dof = all_dof - 7
        else:
            if seeder_cfg["ik_init_q"] is None:
                self.ik_init = None
            else:
                self.ik_init = tensor_args.to_device(seeder_cfg["ik_init_q"]).view(1, 1, -1)
            if self.b_dummy_arm:
                self.ik_retract_config = tensor_args.to_device(torch.zeros((6))).view(1, 1, -1)
            else:
                self.ik_retract_config = None
            self.q_dof = all_dof
        self.ik_max_retries = int(seeder_cfg.get("ik_max_retries", 0) or 0)
        if self.ik_max_retries < 0:
            raise ValueError(f"ik_max_retries must be non-negative, got {self.ik_max_retries}")

        # jitter r and t
        if self.tr_link_names is not None:
            self.skip_transfer = seeder_cfg["skip_transfer"]
            self.jitter_rt_random_gen = self._set_jitter_tr(seeder_cfg["jitter_dist"], seeder_cfg["jitter_angle"])

        self.seeder_cfg = seeder_cfg
        self.world_coll_checker = world_coll_checker
        self.reset(obj_lst)
        return

    def _init_r_from_axis(self, r0):
        base_axis_palm = normalize_vector(r0)
        base_t1 = self.tensor_args.to_device([0, 1, 0])
        base_t2 = self.tensor_args.to_device([0, 0, 1])  # avoid base_axis_palm parallel to base_t1
        proj_xy = (base_t1 * base_axis_palm).sum(dim=-1, keepdim=True).abs()
        base_axis_thumb = torch.where(proj_xy > 0.99, base_t2, base_t1)
        r6d = torch.cat([base_axis_palm, base_axis_thumb], dim=-1)
        return r6d

    def _init_ultradexgrasp_pose(self, extra_info: WorldCollision):
        """Initialize pose for UltraDexGrasp sampling mode."""
        if getattr(extra_info, "_contact_mesh_surface_points", None) is None:
            raise ValueError(
                "sampling_mode='ultradexgrasp' requires contact mesh surface points. "
                "Set seeder_cfg.obj_sample.contact_surface_points_mode to 'always' or 'auto' "
                "and use contact_surface_sample_num > 0."
            )
        num_samples = self.seeder_cfg["obj_sample"]["num"]
        base_t_list = []
        ind_upper_list = []

        for obj_idx in range(extra_info._contact_mesh_surface_points.shape[0]):
            center_pos = extra_info._contact_mesh_surface_points[obj_idx].mean(dim=-2)
            num_point = extra_info._contact_mesh_surface_points[obj_idx].shape[0]
            # Sample along the object X axis; the Y bounds below place the two hands on opposite sides.
            x_min = center_pos[0] - 0.07
            x_max = center_pos[0] + 0.03
            obj_height = (
                extra_info._contact_mesh_surface_points[obj_idx][..., 2].max()
                - extra_info._contact_mesh_surface_points[obj_idx][..., 2].min()
            )
            if obj_height < 0.1:
                z_min = extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.15
                z_max = extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() + 0.1
            elif obj_height < 0.35:
                z_min = extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.15
                z_max = torch.max(
                    extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() - 0.1,
                    extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.2,
                )
            else:
                z_min = torch.max(
                    extra_info._contact_mesh_surface_points[obj_idx][..., 2].min() + 0.16, center_pos[2] - 0.04
                )
                z_max = torch.min(
                    extra_info._contact_mesh_surface_points[obj_idx][..., 2].max() - 0.14, center_pos[2] + 0.06
                )
            y_low = extra_info._contact_mesh_surface_points[obj_idx][..., 1].min() - 0.10
            y_high = extra_info._contact_mesh_surface_points[obj_idx][..., 1].max() + 0.10

            # Sample num_samples points for this object
            x = torch.rand(num_samples, device=self.tensor_args.device) * (x_max - x_min) + x_min
            z = torch.rand(num_samples, device=self.tensor_args.device) * (z_max - z_min) + z_min
            right_hand = torch.stack([x, y_low.repeat(num_samples), z], dim=-1)
            left_hand = torch.stack([x, y_high.repeat(num_samples), z], dim=-1)
            base_t_list.append(torch.stack([right_hand, left_hand], dim=1))
            ind_upper_list.append(num_samples)

            # # Debug prints
            # mesh_name = extra_info.contact_obj_names[obj_idx]
            # print(f"Processing object {obj_idx}: {mesh_name}, samples: {num_samples}")
            # print(f"  y_low: {y_low.item():.4f}, y_high: {y_high.item():.4f}")

        base_t = torch.cat(base_t_list, dim=0)

        # Palm definition using Euler angles (similar to UltraDexGrasp)
        random_z_rotation = -torch.pi * 7 / 16 * torch.rand(base_t.shape[0], device=self.tensor_args.device)

        euler_right = (
            torch.tensor([torch.pi / 2, torch.pi / 2, 0], device=self.tensor_args.device)
            .unsqueeze(0)
            .expand(base_t.shape[0], 3)
            .clone()
        )
        euler_right[:, 2] = -random_z_rotation
        rot_mat_right = euler_angles_to_matrix(euler_right, "XYX")  # column-first

        euler_left = (
            torch.tensor([-torch.pi / 2, torch.pi / 2, 0], device=self.tensor_args.device)
            .unsqueeze(0)
            .expand(base_t.shape[0], 3)
            .clone()
        )
        euler_left[:, 2] = random_z_rotation
        rot_mat_left = euler_angles_to_matrix(euler_left, "XYX")  # column-first

        # Convert rotation matrices to 6D representation
        r_repre_right = torch.cat([rot_mat_right[..., 0], rot_mat_right[..., 1]], dim=-1).unsqueeze(1)
        r_repre_left = torch.cat([rot_mat_left[..., 0], rot_mat_left[..., 1]], dim=-1).unsqueeze(1)
        r_repre = torch.cat([r_repre_right, r_repre_left], dim=1)

        # Create index generator for sampling
        ind_upper = self.tensor_args.to_device(torch.tensor(ind_upper_list, dtype=torch.int32))
        ind_upper = torch.cumsum(ind_upper, dim=0)
        ind_lower = torch.cat([ind_upper[0:1] * 0, ind_upper[:-1]])
        ind_random_gen = HaltonGenerator(
            len(ind_upper_list),
            self.tensor_args,
            up_bounds=ind_upper,
            low_bounds=ind_lower,
            seed=1312,
        )

        return base_t, r_repre, ind_random_gen

    def _set_base_trq(self, t, r, q, extra_info: WorldCollision = None):
        # log_warn(f'Initialize hand pose. t: {t}, r: {r}, q: {q}')
        self.surface_xy_centers = None
        if self.tr_link_names is None:
            base_t = None
            base_r = None
        else:
            tr_num = len(self.tr_link_names)
            if t is not None and r is not None:
                base_t = self.tensor_args.to_device(t).view(1, 1, -1)
                r_repre = self.tensor_args.to_device(r).view(1, 1, -1)
                ind_random_gen = None
            elif t is None and r is None:
                base_t = extra_info.surface_sample_positions.view(-1, 1, 3)
                r_repre = self._init_r_from_axis(-extra_info.surface_sample_normals).view(-1, 1, 6)
                ind_random_gen = HaltonGenerator(
                    len(extra_info.surface_sample_ind_upper),
                    self.tensor_args,
                    up_bounds=extra_info.surface_sample_ind_upper,
                    low_bounds=extra_info.surface_sample_ind_lower,
                    seed=1312,
                )
                if tr_num == 1:
                    pass  # No change needed for single hand
                elif tr_num == 2:
                    if self.sampling_mode == "ultradexgrasp":
                        base_t, r_repre, ind_random_gen = self._init_ultradexgrasp_pose(extra_info)
                    else:
                        # Repeat for both hands
                        base_t = base_t.repeat(1, 2, 1)
                        r_repre = r_repre.repeat(1, 2, 1)
                        if self.sampling_mode == "random_xy_symmetric":
                            self.surface_xy_centers = self._compute_surface_xy_centers(
                                extra_info.surface_sample_positions,
                                extra_info.surface_sample_ind_lower,
                                extra_info.surface_sample_ind_upper,
                            )
                else:
                    raise NotImplementedError(f"tr_num={tr_num} not supported")
            else:
                raise NotImplementedError

            base_r = matrix_from_rot_repre(r_repre)

        base_q = q if q is not None else [0] * self.q_dof
        assert len(base_q) == self.q_dof, self.q_dof

        base_q = self.tensor_args.to_device(base_q).view(1, 1, -1)
        if self.tr_link_names is not None and base_t.shape[0] > 1:
            base_q = base_q.expand(base_t.shape[0], 1, -1)

        return base_t, base_r, base_q, ind_random_gen

    def _set_jitter_tr(self, jitter_dist, jitter_angle):
        jitter_bound_low = (jitter_dist[0] + [i / 180 * np.pi for i in jitter_angle[0]]) * len(self.tr_link_names)
        jitter_bound_up = (jitter_dist[1] + [i / 180 * np.pi for i in jitter_angle[1]]) * len(self.tr_link_names)
        random_gen = HaltonGenerator(
            len(jitter_bound_low), self.tensor_args, up_bounds=jitter_bound_up, low_bounds=jitter_bound_low, seed=1312
        )
        return random_gen

    def _load_base_trq(self, obj_lst, load_path_dict):
        raise NotImplementedError

        robot_pose = []
        for obj_code in obj_lst:
            obj_code = obj_code.split("_scale_")[0]  # This is only to fit the need of jialiang's data
            path = os.path.join(load_path_dict["base"], obj_code, load_path_dict["suffix"])
            log_warn(f"load hand pose initialization from {path}")
            data = dict(np.load(path, allow_pickle=True))
            tmp_robot_pose = self.tensor_args.to_device(data["robot_pose"])
            robot_pose.append(tmp_robot_pose)
        robot_pose = torch.stack(robot_pose, dim=0)
        if self.tr_link_names is None:
            base_t = None
            base_r = None
        else:
            rot_repre_num = (robot_pose.shape[-1] - self.q_dof) // len(self.tr_link_names) - 3
            if len(self.tr_link_names) > 1:
                raise NotImplementedError
            base_t = robot_pose[..., :3].unsqueeze(-2)
            base_r = matrix_from_rot_repre(robot_pose[..., 3 : 3 + rot_repre_num]).unsqueeze(-3)
        base_q = robot_pose[..., -self.q_dof :][..., load_path_dict["reorder_q"]]
        return base_t, base_r, base_q

    def reset(self, obj_lst=None):
        self.jitter_rt_random_gen.reset()

        # init base r, t, q
        if self.seeder_cfg["load_path"] is not None:
            assert obj_lst is not None
            # [b, n, tr_num, 3], [b, n, tr_num, 3, 3], [b, n, q_dof]
            self.base_t, self.base_r, self.base_q = self._load_base_trq(obj_lst, self.seeder_cfg["load_path"])
            self.ind_random_gen = None
        else:
            # [b/1, 1, tr_num, 3], [b/1, 1, tr_num, 3, 3], [b/1, 1, q_dof]
            self.base_t, self.base_r, self.base_q, self.ind_random_gen = self._set_base_trq(
                self.seeder_cfg["t"], self.seeder_cfg["r"], self.seeder_cfg["q"], extra_info=self.world_coll_checker
            )
        return

    def _jitter_on_base_tr(self, base_trans, base_rot):
        batch, num_samples, tr_num = base_trans.shape[:-1]
        rand_num = self.jitter_rt_random_gen.get_samples(batch * num_samples, bounded=True).view(
            batch, num_samples, tr_num, 6
        )
        rand_dist = rand_num[..., :3]
        rand_jitter_angle = rand_num[..., 3:]

        # calculate jittered translation and rotation
        jitter_rotation = euler_angles_to_matrix(torch.flip(rand_jitter_angle, [-1]), "ZYX")
        final_rotation = base_rot @ jitter_rotation
        final_translation = base_trans - (final_rotation @ rand_dist.unsqueeze(-1)).squeeze(-1)
        return final_translation, final_rotation

    def _compute_surface_xy_centers(self, surface_positions, ind_lower, ind_upper):
        """Compute each environment object's xy center from surface candidates.

        Args:
            surface_positions: Flattened surface candidate positions with shape ``[candidate_num, 3]``.
            ind_lower: Inclusive candidate slice starts for each environment.
            ind_upper: Exclusive candidate slice ends for each environment.

        Returns:
            Tensor with shape ``[env_num, 2]`` containing the mean xy position for each environment.
        """
        centers = []
        for low, upper in zip(ind_lower.long(), ind_upper.long()):
            if upper <= low:
                raise ValueError("Cannot compute random_xy_symmetric center from an empty surface candidate slice")
            centers.append(surface_positions[low:upper, :2].mean(dim=0))
        return torch.stack(centers, dim=0)

    def _sample_random_xy_symmetric(self, sample_idx):
        """Choose left-hand candidates symmetric to right-hand candidates in the object xy plane.

        Args:
            sample_idx: Sample indices with shape ``[batch, num_samples, tr_num]``.

        Returns:
            Updated sample indices where the second transfer link is the nearest same-object xy-symmetric candidate.
        """
        if self.surface_xy_centers is None:
            raise ValueError("sampling_mode='random_xy_symmetric' requires surface sample candidate ranges")

        right_sample_idx = sample_idx[:, :, 0]
        right_trans = self.base_t[right_sample_idx, 0]
        batch = right_sample_idx.shape[0]
        ind_lower = self.ind_random_gen.low_bounds.long()
        ind_upper = (self.ind_random_gen.low_bounds + self.ind_random_gen.range_b).long()

        if batch > self.surface_xy_centers.shape[0]:
            raise ValueError(
                f"random_xy_symmetric received batch={batch}, "
                f"but only {self.surface_xy_centers.shape[0]} surface candidate ranges are available"
            )

        for env_idx in range(batch):
            low = ind_lower[env_idx]
            upper = ind_upper[env_idx]
            if upper <= low:
                raise ValueError("Cannot sample random_xy_symmetric from an empty surface candidate slice")

            # Mirror right-hand xy about the current object's xy center while using right z only as a search target.
            center_xy = self.surface_xy_centers[env_idx]
            left_target = right_trans[env_idx].clone()
            left_target[:, :2] = 2.0 * center_xy.view(1, 2) - left_target[:, :2]
            left_candidates = self.base_t[low:upper, 1, :]
            dist = torch.cdist(left_target.unsqueeze(0), left_candidates.unsqueeze(0))[0]
            local_left_idx = torch.argmin(dist, dim=-1)
            sample_idx[env_idx, :, 1] = local_left_idx + low
        return sample_idx

    def _sample_bimanual(self, sample_idx, num_samples, tr_num):
        """Apply bimanual sampling strategy based on sampling_mode."""
        if self.sampling_mode == "random":
            # Completely random sampling for both hands
            pass
        elif self.sampling_mode == "random_symmetric":
            # Find symmetric configuration
            right_sample_idx = sample_idx[:, :, 0]
            right_trans = self.base_t[right_sample_idx, 0]
            left_trans = -right_trans
            base_t_left = self.base_t[:, 1, :]
            dist = torch.cdist(left_trans, base_t_left)
            left_sample_idx = torch.argmin(dist, dim=-1)
            sample_idx[:, :, 1] = left_sample_idx
        elif self.sampling_mode == "random_xy_symmetric":
            sample_idx = self._sample_random_xy_symmetric(sample_idx)
        elif self.sampling_mode == "ultradexgrasp":
            # Already handled in _init_ultradexgrasp_pose
            pass
        else:
            raise NotImplementedError(f"Sampling mode '{self.sampling_mode}' not implemented")
        return sample_idx

    def _sample_to_shape(self, batch, num_samples):
        tr_num = len(self.tr_link_names)
        if self.ind_random_gen is not None:
            sample_idx = (
                self.ind_random_gen.get_samples(num_samples * tr_num, bounded=True)
                .long()
                .view(num_samples, tr_num, -1)
                .permute(2, 0, 1)
            )
            if tr_num == 2:
                sample_idx = self._sample_bimanual(sample_idx, num_samples, tr_num)

            # Index each hand separately
            base_trans = torch.stack([self.base_t[sample_idx[:, :, i], i] for i in range(tr_num)], dim=2)
            base_rot = torch.stack([self.base_r[sample_idx[:, :, i], i] for i in range(tr_num)], dim=2)
            base_q = self.base_q[sample_idx[:, :, 0], 0]
        else:
            if self.base_q.shape[0] < batch or self.base_q.shape[1] < num_samples:
                repeat_b = math.ceil(batch / self.base_q.shape[0])
                repeat_n = math.ceil(num_samples / self.base_q.shape[1])
                self.base_r = self.base_r.repeat(repeat_b, repeat_n, 1, 1, 1) if self.base_r is not None else None
                self.base_t = self.base_t.repeat(repeat_b, repeat_n, 1, 1) if self.base_t is not None else None
                self.base_q = self.base_q.repeat(repeat_b, repeat_n, 1)
            base_rot = self.base_r[:batch, :num_samples] if self.base_r is not None else None
            base_trans = self.base_t[:batch, :num_samples] if self.base_t is not None else None
            base_q = self.base_q[:batch, :num_samples]

        return base_trans, base_rot, base_q

    def get_samples(self, batch, num_samples):
        base_trans, base_rot, base_q = self._sample_to_shape(batch, num_samples)
        return self.get_samples_from_base_pose(base_trans, base_rot, base_q=base_q, apply_transfer=True)

    def get_samples_from_base_pose(self, base_trans, base_rot, base_q=None, apply_transfer=False, apply_jitter=True):
        """Convert explicit transfer-link poses into full robot seed configurations.

        Args:
            base_trans: Transfer-link translations with shape ``[batch, num_samples, link_num, 3]``.
            base_rot: Transfer-link rotations with shape ``[batch, num_samples, link_num, 3, 3]``.
            base_q: Optional full-robot seed q used before replacing IK-controlled joints.
            apply_transfer: Whether to apply the robot hand-pose transfer before IK.
            apply_jitter: Whether to apply the heuristic surface jitter before transfer.

        Returns:
            Full robot seed tensor with shape ``[batch, num_samples, dof]``.
        """
        batch, num_samples = base_trans.shape[:2]
        if base_q is None:
            if self.base_q.shape[0] < batch or self.base_q.shape[1] < num_samples:
                repeat_b = math.ceil(batch / self.base_q.shape[0])
                repeat_n = math.ceil(num_samples / self.base_q.shape[1])
                self.base_q = self.base_q.repeat(repeat_b, repeat_n, 1)
            base_q = self.base_q[:batch, :num_samples]

        if self.tr_link_names is not None:
            # Skip jittering for ultradexgrasp mode (uses pre-computed rotations)
            if apply_transfer and apply_jitter and self.sampling_mode != "ultradexgrasp":
                final_trans, final_rot = self._jitter_on_base_tr(base_trans, base_rot)
            else:
                # Use base poses directly without jittering
                final_trans, final_rot = base_trans, base_rot
            if apply_transfer and not self.skip_transfer:
                final_trans, final_rot = self.full_robot_model.get_transfered_pose(
                    final_trans.contiguous(), final_rot.contiguous(), self.tr_link_names
                )
            final_quat = matrix_to_quaternion(final_rot)

        # solve IK of the tr_link_names to get the arm qpos
        if self.ik_solver is not None:
            target_link_poses = {}
            for i, link_name in enumerate(self.tr_link_names):
                target_link_poses[link_name] = Pose(
                    final_trans[..., i, :].reshape(-1, 3), final_quat[..., i, :].reshape(-1, 4)
                )
                if i == 0:
                    goal = target_link_poses[link_name]

            # Materialize expanded IK buffers because the solver keeps these tensors in the
            # goal buffer and later updates them in-place via copy_().
            ik_init = self.ik_init.expand(batch, num_samples, -1).clone() if self.ik_init is not None else None
            ik_retract_config = (
                self.ik_retract_config.expand(batch, num_samples, -1).clone()
                if self.ik_retract_config is not None
                else None
            )

            result = self.ik_solver.solve_batch(
                goal, link_poses=target_link_poses, seed_config=ik_init, retract_config=ik_retract_config
            )
            self._retry_failed_ik(result, target_link_poses, ik_init, ik_retract_config)

            if torch.any(~result.success):
                failed_count = int(torch.count_nonzero(~result.success).item())
                total_count = int(result.success.numel())
                log_warn(
                    f"IK failed for {failed_count}/{total_count} target samples "
                    f"after {self.ik_max_retries} retries"
                )
            arm_q = result.solution.view(batch, num_samples, -1)
            hand_pose = base_q.clone()
            hand_pose[..., self.replace_ind] = arm_q
        else:
            hand_pose = torch.cat([final_trans.squeeze(-2), final_quat.squeeze(-2), base_q], dim=-1)

        return hand_pose

    def _retry_failed_ik(self, result, target_link_poses, seed_config=None, retract_config=None):
        """Retry IK only for targets that failed in the initial batch solve.

        Args:
            result: IK result from the initial solve; successful retry solutions are written into it.
            target_link_poses: Mapping from transfer link name to flattened target ``Pose``.
            seed_config: Optional expanded IK seed tensor aligned with the initial batch.
            retract_config: Optional expanded retract tensor aligned with the initial batch.

        Returns:
            None.
        """
        if self.ik_max_retries <= 0 or result.success is None or result.solution is None:
            return

        solution_view = result.solution.reshape(-1, result.solution.shape[-1])
        success_view = result.success.reshape(-1)

        for _ in range(self.ik_max_retries):
            failed_indices = torch.nonzero(~success_view, as_tuple=False).view(-1)
            if failed_indices.numel() == 0:
                return
            # Keep the retry solve at the original batch size. The IK solver keeps
            # internal solve-state buffers, so changing to a compact failed-only
            # batch can corrupt the next call's expected batch size.
            retry_result = self.ik_solver.solve_batch(
                target_link_poses[self.tr_link_names[0]],
                link_poses=target_link_poses,
                seed_config=seed_config,
                retract_config=retract_config,
            )
            retry_success = retry_result.success.reshape(-1)
            recovered_mask = retry_success.index_select(0, failed_indices)
            if torch.any(recovered_mask):
                solved_indices = failed_indices[recovered_mask]
                retry_solution_view = retry_result.solution.reshape(-1, retry_result.solution.shape[-1])
                solution_view[solved_indices] = retry_solution_view.index_select(0, solved_indices)
                success_view[solved_indices] = True

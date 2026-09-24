import os
import sys
import glob
import logging
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import re
import imageio
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import multiprocessing
from pathlib import Path
import trimesh as tm
import matplotlib.cm as cm

from curobo.util.visualizer import Visualizer
from curobo.util.artifact_path import resolve_artifact_path
from curobo.util.visualization import look_at, create_colored_axes
from curobo.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
    get_robot_path,
    get_assets_path,
)

from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def save_grasp_images(params):
    file_path, cfg = params[0], params[1]
    device = cfg.device

    # Find robot URDF
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), cfg.manip_cfg_file))
    robot_file = manip_config_data["robot_file"]
    robot_config_data = load_yaml(join_path(get_robot_path(), robot_file))
    urdf_file = robot_config_data["robot_cfg"]["kinematics"]["urdf_path"]
    robot_urdf_path = join_path(get_assets_path(), urdf_file)
    mesh_dir_path = os.path.dirname(robot_urdf_path)

    use_root_pose = robot_config_data["robot_cfg"]["kinematics"]["use_root_pose"]

    # Create visualizer
    visualizer = Visualizer(
        robot_urdf_path=robot_urdf_path,
        mesh_dir_path=mesh_dir_path,
        device=device,
    )

    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist.")

    grasp_data = np.load(file_path, allow_pickle=True).item()
    scene_path = resolve_artifact_path(str(grasp_data["scene_path"][0]))
    scene_data = np.load(scene_path, allow_pickle=True).item()

    joint_names = grasp_data["joint_names"]
    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

    # object mesh
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)

    axis_mesh = create_colored_axes(origin_size=0.005, axis_length=1.0, radius=0.002)

    syn_pose = torch.from_numpy(grasp_data["robot_pose"]).to(device=device).squeeze(0)
    n_sample, n_pose, n_dof = syn_pose.shape[-3:]

    b_opt_process = cfg.task.b_opt_process
    if b_opt_process:
        pose_lst = ((n_pose - 1) * np.array(cfg.task.opt_progress)).astype(np.int64)
    else:
        if n_pose != 3:
            raise ValueError(f"Rendering three stages requires 3 saved poses; got {n_pose} in {file_path}. Use task.b_opt_process=true task.opt_progress=[1.0] for final-only output, or synthesize with task.save_debug=true task.save_data=final_and_mid.")
        pose_lst = [0, 1, 2]

    requested_sample_lst = [int(sample_idx) for sample_idx in cfg.task.sample_lst]
    sample_lst = [sample_idx for sample_idx in requested_sample_lst if 0 <= sample_idx < n_sample]
    if len(sample_lst) != len(requested_sample_lst):
        logging.warning(
            "Requested render samples %s, but %s only contains %d grasp sample(s); rendering existing samples %s.",
            requested_sample_lst,
            file_path,
            n_sample,
            sample_lst,
        )
    if not sample_lst:
        logging.warning("Skip rendering %s because none of the requested grasp samples exist.", file_path)
        return

    robot_pose = syn_pose[sample_lst, ...][:, pose_lst, :].clone().view(-1, n_dof)
    if not use_root_pose:
        base_pose = torch.tensor([[0, 0, 0, 1, 0, 0, 0]], device=device).repeat(robot_pose.shape[0], 1)
        robot_pose = torch.cat([base_pose, robot_pose], dim=-1)

    visualizer.set_robot_parameters(robot_pose, joint_names=joint_names)

    for i_sample, sample_idx in enumerate(sample_lst):
        for i_pose, pose_idx in enumerate(pose_lst):
            # Render the meshes
            # t = (i_pose + 1) / len(pose_lst)
            # color = cm.Reds(t)  # drop alpha
            color = [0.941, 0.502, 0.502, 0.7]

            idx = i_sample * len(pose_lst) + i_pose
            robot_mesh = visualizer.get_robot_trimesh_data(idx, color=color)
            all_meshes = axis_mesh + [obj_mesh, robot_mesh]

            ############## Meshes to images ##############

            # Create pyrender scene
            scene = pyrender.Scene(bg_color=[226 / 255, 240 / 255, 217 / 255, 1.0])

            # trimesh to pyrender scene
            for m in all_meshes:
                if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors.shape[1] == 4:
                    color = m.visual.vertex_colors[0] / 255.0
                else:
                    color = [0.8, 0.8, 0.8, 1.0]
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.9
                )
                mesh_pyr = pyrender.Mesh.from_trimesh(m, material=material)
                scene.add(mesh_pyr)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

            os.environ["PYOPENGL_PLATFORM"] = "egl"  # use EGL on headless server
            r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)

            save_dir = file_path.replace("graspdata", "grasp_imgs").replace(".npy", "")
            save_dir = os.path.join(save_dir, f"grasp_{sample_idx}")
            os.makedirs(save_dir, exist_ok=True)

            cam_pose_lst = 0.5 * np.array(
                [
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, -1],
                ]
            )
            for i_cam, cam_pose in enumerate(cam_pose_lst):
                cam_pose = look_at(cam_pose, [0, 0, 0])
                cam_node = scene.add(camera, pose=cam_pose)
                light_node = scene.add(light, pose=cam_pose)
                color, _ = r.render(scene)
                scene.remove_node(cam_node)
                scene.remove_node(light_node)
                path = os.path.join(save_dir, f"pose_{pose_idx}_view_{i_cam}.jpg")
                imageio.imwrite(path, color)

            r.delete()
            logging.info(f"Saved images: {save_dir}, pose {pose_idx}, {len(cam_pose_lst)} views")


def task_render(cfg: DictConfig):
    """
    Rendering and saving images of the synthesized bimanual grasps (before filtering).
    Rendering intermediate optimization results.
    No pregrasp and squeeze poses. No arms.
    """

    output_path, _ = os.path.splitext(str(cfg.manip_cfg_file))  # robot_name/type
    output_path = os.path.join(cfg.output_path, output_path, cfg.name, "graspdata")

    all_grasp_file_lst = glob.glob(os.path.join(output_path, "**/*.npy"), recursive=True)

    # Select grasp data files with the specified object codes and opt steps in .yaml
    selected_grasp_file_lst = []
    if "object_lst" in cfg.task and cfg.task.object_lst is not None:
        for grasp_file in all_grasp_file_lst:
            graspfile_path = Path(grasp_file)
            object_name = graspfile_path.parents[1].name

            if object_name in cfg.task.object_lst:
                selected_grasp_file_lst.append(grasp_file)
    else:
        selected_grasp_file_lst = all_grasp_file_lst

    logging.info(f"Select {len(selected_grasp_file_lst)}/{len(all_grasp_file_lst)} grasp file.")

    ########################  Task  ########################

    cfg_lst = [cfg] * len(selected_grasp_file_lst)
    iterable_params = zip(selected_grasp_file_lst, cfg_lst)

    if cfg.task.debug:
        for params in iterable_params:
            save_grasp_images(params)
    else:
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            result_iter = pool.imap_unordered(save_grasp_images, iterable_params)
            results = list(result_iter)

    # ########################  Check saved files  ########################

    # save_dir = os.path.join(exp_path, "visualizations/opt_process")
    # img_lst = glob.glob(os.path.join(save_dir, "**/*.jpg"), recursive=True)
    # logging.info(f"Save {len(img_lst)} images in {save_dir}.")
    # logging.info("Finish grasp image saving.")

    return

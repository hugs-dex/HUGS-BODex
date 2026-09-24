"""
Visualize hand pose transfer frames for a robot hand configuration.

This script supports two transfer types:
- surface_sample: frame sampled from the object surface to the hand base frame
- human: human index MCP frame to the hand base frame

Both modes share the same viser web UI, robot config loading, frame visualization,
and transfer YAML loading.
"""

import argparse
import os
import time

import numpy as np
import torch
import viser
import yaml
from scipy.spatial.transform import Rotation as R

from visualizer import Visualizer


DEFAULT_ROBOT_FILES = {
    "shadow": "src/curobo/content/configs/robot/right_shadow_hand_sim.yml",
    "dual_dummy_arm_shadow": "src/curobo/content/configs/robot/dual_dummy_arm_shadow.yml",
    "leap": "src/curobo/content/configs/robot/leap_hand.yml",
    "dual_dummy_arm_leap": "src/curobo/content/configs/robot/dual_dummy_arm_leap.yml",
    "leap_sp": "src/curobo/content/configs/robot/leap_sp.yml",
    "dual_dummy_arm_leap_sp": "src/curobo/content/configs/robot/dual_dummy_arm_leap_sp.yml",
}

DEFAULT_HUMAN_TRANSFER_FILES = {
    "shadow": "src/curobo/content/configs/robot/hand_pose_human_transfer/right_shadow_hand.yml",
    "dual_dummy_arm_shadow": "src/curobo/content/configs/robot/hand_pose_human_transfer/dual_dummy_arm_shadow.yaml",
    "leap": "src/curobo/content/configs/robot/hand_pose_human_transfer/leap_hand.yml",
    "dual_dummy_arm_leap": "src/curobo/content/configs/robot/hand_pose_human_transfer/dual_dummy_arm_leap.yml",
    "leap_sp": "src/curobo/content/configs/robot/hand_pose_human_transfer/leap_sp.yml",
    "dual_dummy_arm_leap_sp": "src/curobo/content/configs/robot/hand_pose_human_transfer/dual_dummy_arm_leap_sp.yml",
}

ROBOT_CONFIG_ROOT = "src/curobo/content/configs/robot"

TRANSFER_FRAME_NAMES = {
    "surface_sample": "surface_sample_palm",
    "human": "human_index_mcp",
}


def make_transform(rotation, translation):
    """
    Build a homogeneous transform matrix from rotation and translation.

    Args:
        rotation: A 3x3 rotation matrix.
        translation: A 3D translation vector.

    Returns:
        A 4x4 homogeneous transform matrix.
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_robot_file_from_hand_name(hand_name):
    """
    Resolve the robot config file associated with a hand name.

    Args:
        hand_name: Name of the hand preset.

    Returns:
        Robot config YAML path for the hand.
    """
    if hand_name not in DEFAULT_ROBOT_FILES:
        valid_names = ", ".join(sorted(DEFAULT_ROBOT_FILES))
        raise ValueError(f"Unknown hand name '{hand_name}'. Valid names: {valid_names}")
    return DEFAULT_ROBOT_FILES[hand_name]


def load_robot_paths(robot_file):
    """
    Load URDF and mesh directory paths from a robot config YAML file.

    Args:
        robot_file: Path to the robot config YAML file.

    Returns:
        Robot config dictionary, robot URDF path, and mesh directory path.
    """
    with open(robot_file, "r") as file:
        robot_config = yaml.safe_load(file)

    kinematics_cfg = robot_config["robot_cfg"]["kinematics"]
    assets_root = os.path.join("src/curobo/content/assets")
    robot_urdf_path = os.path.join(assets_root, kinematics_cfg["urdf_path"])
    mesh_dir_path = os.path.join(assets_root, kinematics_cfg["asset_root_path"])
    return robot_config, robot_urdf_path, mesh_dir_path


def get_default_human_transfer_path(hand_name):
    """
    Resolve the default human transfer YAML path for a hand preset.

    Args:
        hand_name: Name of the hand preset.

    Returns:
        Human transfer YAML path for the hand preset.
    """
    if hand_name not in DEFAULT_HUMAN_TRANSFER_FILES:
        valid_names = ", ".join(sorted(DEFAULT_HUMAN_TRANSFER_FILES))
        raise ValueError(f"No default human transfer path for '{hand_name}'. Valid names: {valid_names}")
    return DEFAULT_HUMAN_TRANSFER_FILES[hand_name]


def get_default_surface_sample_transfer_path(kinematics_cfg):
    """
    Resolve the default surface-sample transfer YAML path from a robot config.

    Args:
        kinematics_cfg: Robot kinematics configuration dictionary.

    Returns:
        Surface-sample transfer YAML path for the robot config.
    """
    return os.path.join(ROBOT_CONFIG_ROOT, kinematics_cfg["hand_pose_transfer_path"])


def ensure_transfer_path_in_robot_configs(path):
    """
    Ensure a transfer YAML path is under the robot config directory.

    Args:
        path: Transfer YAML path to check.

    Returns:
        Normalized transfer YAML path.
    """
    normalized_path = os.path.normpath(path)
    config_root = os.path.abspath(ROBOT_CONFIG_ROOT)
    transfer_path = os.path.abspath(normalized_path)
    if os.path.commonpath([config_root, transfer_path]) != config_root:
        raise ValueError(f"Transfer path must be under {ROBOT_CONFIG_ROOT}: {path}")
    return normalized_path


def matrix_to_wxyz(rotation):
    """
    Convert a rotation matrix to the quaternion convention used by viser.

    Args:
        rotation: A 3x3 rotation matrix.

    Returns:
        Quaternion in wxyz order.
    """
    xyzw = R.from_matrix(rotation).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def add_frame_to_scene(scene, name, transform, axis_length, axis_radius, origin_radius):
    """
    Add a coordinate frame to the viser scene from a homogeneous transform.

    Args:
        scene: Viser scene API used to create the frame.
        name: Scene path for the frame.
        transform: 4x4 transform from local frame to world frame.
        axis_length: Length of each rendered frame axis.
        axis_radius: Radius of each rendered frame axis.
        origin_radius: Radius of the rendered frame origin.

    Returns:
        Viser frame handle.
    """
    return scene.add_frame(
        name,
        axes_length=axis_length,
        axes_radius=axis_radius,
        origin_radius=origin_radius,
        wxyz=matrix_to_wxyz(transform[:3, :3]),
        position=transform[:3, 3],
    )


def load_transfer(path):
    """
    Load an existing transfer YAML file.

    Args:
        path: YAML path to load.

    Returns:
        Mapping from hand base link name to transfer parameters.
    """
    path = ensure_transfer_path_in_robot_configs(path)
    with open(path, "r") as file:
        transfer_data = yaml.safe_load(file)

    if transfer_data is None:
        return {}
    if not isinstance(transfer_data, dict):
        raise ValueError(f"Transfer file must contain a mapping from link names to transforms: {path}")
    return transfer_data


def resolve_transfer_path(args, kinematics_cfg):
    """
    Resolve the transfer YAML path for the selected transfer type.

    Args:
        args: Parsed command-line arguments.
        kinematics_cfg: Robot kinematics configuration dictionary.

    Returns:
        Transfer YAML path.
    """
    if args.transfer_path is not None:
        return ensure_transfer_path_in_robot_configs(args.transfer_path)
    if args.transfer_type == "surface_sample":
        return ensure_transfer_path_in_robot_configs(get_default_surface_sample_transfer_path(kinematics_cfg))
    return ensure_transfer_path_in_robot_configs(get_default_human_transfer_path(args.hand_name))


def build_transfer_data(transfer_path):
    """
    Load transfer data from a robot config YAML file.

    Args:
        transfer_path: Transfer YAML path.

    Returns:
        Mapping from hand base link name to transfer parameters.
    """
    return load_transfer(transfer_path)


def parse_args(argv=None):
    """
    Parse command-line options for the viser hand pose transfer viewer.

    Args:
        argv: Optional argument list. When None, argparse reads from sys.argv.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize hand pose transfer frames in a viser web UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface used by the viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port used by the viser server.")
    parser.add_argument(
        "--transfer-type",
        default="surface_sample",
        choices=("surface_sample", "human"),
        help="Transfer type to visualize.",
    )
    parser.add_argument(
        "--hand-name",
        default="shadow",
        choices=sorted(DEFAULT_ROBOT_FILES),
        help="Hand preset name used for default robot and transfer files.",
    )
    parser.add_argument("--robot-file", default=None, help="Robot config YAML path used to derive URDF and mesh paths.")
    parser.add_argument("--transfer-path", default=None, help="YAML path containing transfer parameters.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Visualize selected hand pose transfer frames of the robot in a viser web UI.

    Args:
        argv: Optional argument list. When None, argparse reads from sys.argv.

    Returns:
        None.
    """
    args = parse_args(argv)

    robot_file = args.robot_file or get_robot_file_from_hand_name(args.hand_name)
    robot_config, robot_urdf_path, mesh_dir_path = load_robot_paths(robot_file)
    kinematics_cfg = robot_config["robot_cfg"]["kinematics"]

    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    hand_pose = torch.zeros((1, 3 + 4 + len(visualize.joint_names)))
    hand_pose[:, 3] = 1.0  # quat w

    visualize.set_robot_parameters(hand_pose)
    robot_mesh = visualize.get_robot_trimesh_data(i=0, color=[0, 255, 0, 100])

    world_t_robot = hand_pose[0, 0:3].cpu().numpy()
    world_r_robot = visualize.global_rotation[0].cpu().numpy()
    world_t_robot_tf = make_transform(world_r_robot, world_t_robot)

    transfer_path = resolve_transfer_path(args, kinematics_cfg)
    transfer_data = build_transfer_data(transfer_path)

    frame_records = []
    transfer_frame_name = TRANSFER_FRAME_NAMES[args.transfer_type]

    # Each entry stores the transfer frame pose in the corresponding hand base frame.
    for link_name, transfer_cfg in transfer_data.items():
        hand_base_tf = visualize.current_status[link_name].get_matrix()[0].cpu().numpy()
        world_t_hand_base = world_t_robot_tf @ hand_base_tf

        transfer_r_in_base = np.asarray(transfer_cfg["r"], dtype=np.float64)
        transfer_t_in_base = np.asarray(transfer_cfg["t"], dtype=np.float64)
        transfer_tf_in_base = make_transform(transfer_r_in_base, transfer_t_in_base)
        world_t_transfer = world_t_hand_base @ transfer_tf_in_base

        print(f"{link_name} {transfer_frame_name} rotation matrix:")
        print(transfer_r_in_base)
        print(f"{link_name} {transfer_frame_name} translation vector:")
        print(transfer_t_in_base)

        frame_records.append(
            {
                "link_name": link_name,
                "hand_base_tf": world_t_hand_base,
                "transfer_tf": world_t_transfer,
            }
        )

    server = viser.ViserServer(host=args.host, port=args.port, label=f"Hand Pose Transfer: {args.transfer_type}")
    # Keep the single world frame visible but smaller than the hand-specific frames.
    server.scene.world_axes.visible = True
    server.scene.world_axes.axes_length = 0.035
    server.scene.world_axes.axes_radius = 0.001
    server.scene.world_axes.origin_radius = 0.004
    server.scene.add_mesh_trimesh("/robot", robot_mesh)

    for frame_record in frame_records:
        link_name = frame_record["link_name"]
        add_frame_to_scene(
            server.scene,
            f"/frames/{link_name}/hand_base",
            frame_record["hand_base_tf"],
            axis_length=0.06,
            axis_radius=0.002,
            origin_radius=0.008,
        )
        add_frame_to_scene(
            server.scene,
            f"/frames/{link_name}/{transfer_frame_name}",
            frame_record["transfer_tf"],
            axis_length=0.04,
            axis_radius=0.0015,
            origin_radius=0.006,
        )

    server.gui.add_markdown(
        f"Large axes show hand base frames. Small axes show `{transfer_frame_name}` frames. "
    )
    server.gui.add_markdown(f"Loaded `{transfer_path}` from robot config `{robot_file}`.")

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Viser web UI is running at http://{browser_host}:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

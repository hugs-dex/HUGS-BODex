"""
Visualize robot collision spheres in a viser web UI.

The script resolves a robot from a hand preset or a direct robot config YAML,
loads the URDF mesh with the local Visualizer helper, and overlays the
configured cuRobo collision spheres in world coordinates.
"""

import argparse
import os
import time

import numpy as np
import torch
import trimesh as tm
import viser
import yaml

from visualizer import Visualizer


DEFAULT_ROBOT_FILES = {
    "shadow": "src/curobo/content/configs/robot/right_shadow_hand_sim.yml",
    "dual_dummy_arm_shadow": "src/curobo/content/configs/robot/dual_dummy_arm_shadow.yml",
    "leap_sp": "src/curobo/content/configs/robot/leap_sp.yml",
    "dual_dummy_arm_leap_sp": "src/curobo/content/configs/robot/dual_dummy_arm_leap_sp.yml",
}


def get_robot_file_from_hand_name(hand_name):
    """
    Resolve the robot config file associated with a hand name.

    Args:
        hand_name: Name of the hand preset.

    Returns:
        Robot config YAML path for the hand preset.
    """
    if hand_name not in DEFAULT_ROBOT_FILES:
        valid_names = ", ".join(sorted(DEFAULT_ROBOT_FILES))
        raise ValueError(f"Unknown hand name '{hand_name}'. Valid names: {valid_names}")
    return DEFAULT_ROBOT_FILES[hand_name]


def load_robot_paths(robot_file):
    """
    Load URDF, mesh, and sphere paths from a robot config YAML file.

    Args:
        robot_file: Path to the robot config YAML file.

    Returns:
        Robot config dictionary, robot URDF path, mesh directory path, and sphere YAML path.
    """
    with open(robot_file, "r") as file:
        robot_config = yaml.safe_load(file)

    kinematics_cfg = robot_config["robot_cfg"]["kinematics"]
    assets_root = os.path.join("src/curobo/content/assets")
    robot_config_root = os.path.join("src/curobo/content/configs/robot")
    robot_urdf_path = os.path.join(assets_root, kinematics_cfg["urdf_path"])
    mesh_dir_path = os.path.join(assets_root, kinematics_cfg["asset_root_path"])
    collision_sphere_path = os.path.join(robot_config_root, kinematics_cfg["collision_spheres"])
    return robot_config, robot_urdf_path, mesh_dir_path, collision_sphere_path


def load_collision_spheres(collision_sphere_path):
    """
    Load cuRobo collision sphere definitions.

    Args:
        collision_sphere_path: YAML path containing a collision_spheres mapping.

    Returns:
        Mapping from link name to a list of sphere dictionaries.
    """
    with open(collision_sphere_path, "r") as file:
        data = yaml.safe_load(file)
    return data["collision_spheres"]


def make_default_hand_pose(visualize):
    """
    Create a zero joint pose with an identity root quaternion for the robot.

    Args:
        visualize: Visualizer instance whose joint names define the pose width.

    Returns:
        Hand pose tensor with shape [1, 3 + 4 + number_of_joints].
    """
    hand_pose = torch.zeros((1, 3 + 4 + len(visualize.joint_names)))
    hand_pose[:, 3] = 1.0  # Quaternion w component for identity orientation.
    return hand_pose


def make_sphere_mesh(center, radius, color):
    """
    Create a colored trimesh sphere.

    Args:
        center: 3D sphere center in world coordinates.
        radius: Sphere radius.
        color: RGBA color assigned to the sphere mesh.

    Returns:
        A trimesh mesh for the sphere.
    """
    sphere_mesh = tm.creation.icosphere(subdivisions=3, radius=radius)
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    sphere_mesh.apply_transform(transform)
    sphere_mesh.visual.face_colors = color
    return sphere_mesh


def apply_mesh_color(mesh, color, opacity):
    """
    Apply a uniform RGBA color to both faces and vertices of a mesh.

    Args:
        mesh: Trimesh mesh whose visual colors should be overwritten.
        color: RGB or RGBA color values in uint8 range.
        opacity: Opacity value in [0, 1] used for the alpha channel.

    Returns:
        The same mesh with updated face and vertex colors.
    """
    color_rgba = np.asarray(color, dtype=np.uint8).copy()
    if color_rgba.shape[0] == 3:
        color_rgba = np.concatenate([color_rgba, np.array([255], dtype=np.uint8)])
    color_rgba[3] = int(np.clip(opacity, 0.0, 1.0) * 255)

    # Set both face and vertex colors because different trimesh-to-GLB paths
    # preserve alpha through different visual fields.
    mesh.visual.face_colors = np.tile(color_rgba, (len(mesh.faces), 1))
    mesh.visual.vertex_colors = np.tile(color_rgba, (len(mesh.vertices), 1))
    return mesh


def add_transparent_mesh(scene, name, mesh, color, opacity=None):
    """
    Add a mesh to a viser scene using the simple mesh path when possible.

    Args:
        scene: Viser scene API used to create the mesh node.
        name: Scene path for the mesh.
        mesh: Trimesh mesh to render.
        color: RGB color values in uint8 range.
        opacity: Optional opacity in [0, 1].

    Returns:
        Viser mesh handle.
    """
    color_rgb = tuple(int(value) for value in color[:3])
    if opacity is None:
        return scene.add_mesh_trimesh(name, mesh)

    opacity = float(np.clip(opacity, 0.0, 1.0))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    # Use add_mesh_simple first because it creates a native viser/Three.js
    # material with opacity. add_mesh_trimesh serializes through trimesh/GLB,
    # where alpha values can be dropped by some installed viser versions.
    try:
        return scene.add_mesh_simple(
            name,
            vertices=vertices,
            faces=faces,
            color=color_rgb,
            opacity=opacity,
            material="standard",
            side="double",
        )
    except (AttributeError, TypeError):
        pass

    try:
        return scene.add_mesh_simple(
            name,
            vertices=vertices,
            faces=faces,
            color=color_rgb,
            opacity=opacity,
        )
    except (AttributeError, TypeError):
        pass

    try:
        return scene.add_mesh_trimesh(name, mesh, opacity=opacity)
    except TypeError:
        # Last fallback for older viser versions: rely on embedded alpha.
        return scene.add_mesh_trimesh(name, mesh)


def build_collision_sphere_meshes(visualize, collision_spheres, color):
    """
    Transform collision spheres from link frames to world frames.

    Args:
        visualize: Visualizer with current robot forward kinematics already computed.
        collision_spheres: Mapping from link names to sphere definitions.
        color: RGBA color assigned to each sphere.

    Returns:
        List of tuples containing scene path, sphere mesh, link name, local center, and radius.
    """
    sphere_records = []
    for link_name, spheres in collision_spheres.items():
        if link_name not in visualize.current_status:
            print(f"Skip collision spheres for missing link: {link_name}")
            continue
        for sphere_index, sphere in enumerate(spheres):
            center = torch.tensor(sphere["center"], dtype=torch.float32).reshape(1, 3)
            world_center = visualize.current_status[link_name].transform_points(center)
            if len(world_center.shape) == 3:
                world_center = world_center[0]
            world_center = world_center.reshape(3).detach().cpu().numpy()
            radius = float(sphere["radius"])
            sphere_mesh = make_sphere_mesh(world_center, radius, color)
            sphere_records.append(
                (
                    f"/collision_spheres/{link_name}/{sphere_index}",
                    sphere_mesh,
                    link_name,
                    sphere["center"],
                    radius,
                )
            )
    return sphere_records


def parse_args(argv=None):
    """
    Parse command-line options for the collision sphere viewer.

    Args:
        argv: Optional argument list. When None, argparse reads from sys.argv.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize cuRobo collision spheres in a viser web UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface used by the viser server.")
    parser.add_argument("--port", type=int, default=8081, help="Port used by the viser server.")
    parser.add_argument(
        "--hand-name",
        default="leap_sp",
        choices=sorted(DEFAULT_ROBOT_FILES),
        help="Hand preset name used for the default robot config.",
    )
    parser.add_argument("--robot-file", default=None, help="Robot config YAML path used to derive all assets.")
    parser.add_argument("--robot-opacity", type=float, default=0.35, help="Robot mesh opacity in [0, 1].")
    parser.add_argument("--sphere-opacity", type=float, default=0.7, help="Collision sphere opacity in [0, 1].")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Visualize the selected robot and its cuRobo collision spheres in viser.

    Args:
        argv: Optional argument list. When None, argparse reads from sys.argv.

    Returns:
        None.
    """
    args = parse_args(argv)

    robot_file = args.robot_file or get_robot_file_from_hand_name(args.hand_name)
    _, robot_urdf_path, mesh_dir_path, collision_sphere_path = load_robot_paths(robot_file)

    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)
    hand_pose = make_default_hand_pose(visualize)
    visualize.set_robot_parameters(hand_pose)

    robot_opacity = float(np.clip(args.robot_opacity, 0.0, 1.0))
    sphere_opacity = float(np.clip(args.sphere_opacity, 0.0, 1.0))
    sphere_alpha = int(sphere_opacity * 255)
    robot_mesh = visualize.get_robot_trimesh_data(i=0, color=[0, 255, 0, int(robot_opacity * 255)])
    robot_mesh = apply_mesh_color(robot_mesh, [0, 255, 0], robot_opacity)
    collision_spheres = load_collision_spheres(collision_sphere_path)
    sphere_records = build_collision_sphere_meshes(visualize, collision_spheres, [255, 0, 0, sphere_alpha])

    server = viser.ViserServer(host=args.host, port=args.port, label="Collision Spheres")
    server.scene.world_axes.visible = True
    server.scene.world_axes.axes_length = 0.04
    server.scene.world_axes.axes_radius = 0.001
    server.scene.world_axes.origin_radius = 0.004
    add_transparent_mesh(server.scene, "/robot", robot_mesh, [0, 255, 0], opacity=robot_opacity)

    for scene_path, sphere_mesh, _, _, _ in sphere_records:
        add_transparent_mesh(server.scene, scene_path, sphere_mesh, [255, 0, 0], opacity=sphere_opacity)

    server.gui.add_markdown(
        f"Loaded robot config `{robot_file}` with collision spheres from `{collision_sphere_path}`."
    )
    server.gui.add_markdown(f"Rendered `{len(sphere_records)}` collision spheres.")

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Viser web UI is running at http://{browser_host}:{args.port}")
    print(f"Loaded robot config: {robot_file}")
    print(f"Loaded collision spheres: {collision_sphere_path}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

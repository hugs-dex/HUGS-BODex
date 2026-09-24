"""
Visualize high-frequency self-collision link pairs in a viser web UI.

The script follows the same robot config and collision sphere loading pattern
as ``vis_collision_spheres.py``. Each selected link pair is rendered as a
separate translated copy of the hand, with the two involved links and their
cuRobo collision spheres highlighted in different colors.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh as tm
import viser
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_ROOT = REPO_ROOT / "src" / "curobo" / "content" / "assets"
ROBOT_CONFIG_ROOT = REPO_ROOT / "src" / "curobo" / "content" / "configs" / "robot"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualizer import Visualizer

DEFAULT_ROBOT_FILES = {
    "shadow": "src/curobo/content/configs/robot/right_shadow_hand_sim.yml",
    "dual_dummy_arm_shadow": "src/curobo/content/configs/robot/dual_dummy_arm_shadow.yml",
    "leap": "src/curobo/content/configs/robot/leap_hand.yml",
    "dual_dummy_arm_leap": "src/curobo/content/configs/robot/dual_dummy_arm_leap.yml",
    "leap_sp": "src/curobo/content/configs/robot/leap_sp.yml",
    "dual_dummy_arm_leap_sp": "src/curobo/content/configs/robot/dual_dummy_arm_leap_sp.yml",
}

PAIR_COLOR_A = np.asarray([69, 123, 191, 255], dtype=np.uint8)
PAIR_COLOR_B = np.asarray([210, 113, 57, 255], dtype=np.uint8)
ROBOT_COLOR = np.asarray([92, 130, 95, 255], dtype=np.uint8)
CONTEXT_SPHERE_COLOR = np.asarray([150, 150, 150, 255], dtype=np.uint8)


@dataclass(frozen=True)
class PairSpec:
    """Metadata for one self-collision link pair.

    Args:
        link_a: First link name in the pair.
        link_b: Second link name in the pair.
        label: Human-readable label shown in the UI.
        count: Optional observed count from the latest leap_sp analysis.
        percent: Optional observed percentage from the latest leap_sp analysis.

    Returns:
        Immutable self-collision pair specification.
    """

    link_a: str
    link_b: str
    label: str
    count: int | None = None
    percent: float | None = None


LEAP_SP_TOP_PAIRS = [
    PairSpec(
        "rh_fingertip_base_2",
        "rh_fingertip_base_3",
        "R middle fingertip base vs R ring fingertip base",
        12860,
        34.25,
    ),
    PairSpec(
        "rh_fingertip_base",
        "rh_fingertip_base_2",
        "R index fingertip base vs R middle fingertip base",
        10042,
        26.74,
    ),
    PairSpec(
        "rh_dip_3",
        "rh_fingertip_base_2",
        "R ring DIP vs R middle fingertip base",
        3328,
        8.86,
    ),
    PairSpec(
        "lh_fingertip_base_2",
        "lh_fingertip_base_3",
        "L middle fingertip base vs L ring fingertip base",
        1839,
        4.90,
    ),
    PairSpec(
        "rh_fingertip_3",
        "rh_fingertip_base_2",
        "R ring fingertip vs R middle fingertip base",
        1772,
        4.72,
    ),
    PairSpec(
        "rh_dip_2",
        "rh_fingertip_base_3",
        "R middle DIP vs R ring fingertip base",
        1347,
        3.59,
    ),
    PairSpec(
        "lh_dip_3",
        "lh_fingertip_base_2",
        "L ring DIP vs L middle fingertip base",
        824,
        2.19,
    ),
]


def resolve_robot_file(hand_name: str, robot_file: str | None) -> Path:
    """Resolve a robot config path from a preset or explicit path.

    Args:
        hand_name: Hand preset name used when ``robot_file`` is None.
        robot_file: Optional explicit robot config path.

    Returns:
        Absolute robot config YAML path.
    """

    if robot_file is not None:
        path = Path(robot_file).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    if hand_name not in DEFAULT_ROBOT_FILES:
        valid_names = ", ".join(sorted(DEFAULT_ROBOT_FILES))
        raise ValueError(f"Unknown hand name '{hand_name}'. Valid names: {valid_names}")
    return (REPO_ROOT / DEFAULT_ROBOT_FILES[hand_name]).resolve()


def load_robot_paths(robot_file: Path) -> tuple[dict, Path, Path, Path]:
    """Load URDF, mesh, and collision sphere paths from a robot config.

    Args:
        robot_file: Absolute robot config YAML path.

    Returns:
        Robot config dictionary, URDF path, mesh directory path, and collision sphere YAML path.
    """

    with robot_file.open("r") as file:
        robot_config = yaml.safe_load(file)

    kinematics_cfg = robot_config["robot_cfg"]["kinematics"]
    robot_urdf_path = ASSET_ROOT / kinematics_cfg["urdf_path"]
    mesh_dir_path = ASSET_ROOT / kinematics_cfg["asset_root_path"]
    sphere_path = ROBOT_CONFIG_ROOT / kinematics_cfg["collision_spheres"]
    return robot_config, robot_urdf_path.resolve(), mesh_dir_path.resolve(), sphere_path.resolve()


def load_collision_spheres(collision_sphere_path: Path) -> dict[str, list[dict]]:
    """Load cuRobo collision sphere definitions.

    Args:
        collision_sphere_path: YAML path containing a ``collision_spheres`` mapping.

    Returns:
        Mapping from link name to a list of sphere dictionaries.
    """

    with collision_sphere_path.open("r") as file:
        data = yaml.safe_load(file)
    return data["collision_spheres"]


def parse_pair_text(pair_text: str) -> PairSpec:
    """Parse a command-line link pair specification.

    Args:
        pair_text: Pair string formatted as ``link_a:link_b`` or ``link_a,link_b``.

    Returns:
        Pair specification without count metadata.
    """

    separator = ":" if ":" in pair_text else ","
    parts = [part.strip() for part in pair_text.split(separator) if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid pair '{pair_text}'. Use link_a:link_b or link_a,link_b.")
    return PairSpec(parts[0], parts[1], f"{parts[0]} vs {parts[1]}")


def available_links(visualize: Visualizer, collision_spheres: dict[str, list[dict]]) -> set[str]:
    """Collect link names available for mesh or sphere visualization.

    Args:
        visualize: Visualizer containing URDF mesh and kinematics data.
        collision_spheres: Collision sphere mapping loaded from robot config.

    Returns:
        Set of link names available in the current robot model.
    """

    return set(visualize.current_status.keys()) | set(visualize.robot_mesh.keys()) | set(collision_spheres.keys())


def default_pairs_for_robot(
    available_link_names: set[str],
    max_pairs: int,
) -> list[PairSpec]:
    """Choose the highest-frequency leap_sp pairs that exist in this robot.

    Args:
        available_link_names: Link names present in the current robot.
        max_pairs: Maximum number of default pairs to return.

    Returns:
        Filtered list of default pair specifications.
    """

    pairs = [
        pair
        for pair in LEAP_SP_TOP_PAIRS
        if pair.link_a in available_link_names and pair.link_b in available_link_names
    ]
    if not pairs:
        raise ValueError("No built-in leap_sp self-collision pairs exist in this robot. Pass --pairs explicitly.")
    return pairs[:max_pairs]


def make_hand_poses(pair_count: int, joint_count: int, spacing: float) -> torch.Tensor:
    """Create identity hand poses offset along the x-axis.

    Args:
        pair_count: Number of translated hand copies to render.
        joint_count: Number of robot joints in the visualizer chain.
        spacing: Distance between neighboring hand copies in meters.

    Returns:
        Hand pose tensor with shape ``[pair_count, 3 + 4 + joint_count]``.
    """

    hand_pose = torch.zeros((pair_count, 3 + 4 + joint_count), dtype=torch.float32)
    hand_pose[:, 3] = 1.0
    center = (pair_count - 1) / 2.0
    for pair_index in range(pair_count):
        hand_pose[pair_index, 0] = float(pair_index - center) * spacing
    return hand_pose


def make_sphere_mesh(center: np.ndarray, radius: float, color: np.ndarray) -> tm.Trimesh:
    """Create a colored sphere mesh in world coordinates.

    Args:
        center: Sphere center in world coordinates.
        radius: Sphere radius in meters.
        color: RGBA color in uint8 range.

    Returns:
        Trimesh sphere located at the requested center.
    """

    sphere_mesh = tm.creation.icosphere(subdivisions=3, radius=radius)
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    sphere_mesh.apply_transform(transform)
    sphere_mesh.visual.face_colors = color
    sphere_mesh.visual.vertex_colors = np.tile(color, (len(sphere_mesh.vertices), 1))
    return sphere_mesh


def set_mesh_color(mesh: tm.Trimesh, color: np.ndarray, opacity: float) -> tm.Trimesh:
    """Apply one RGBA color to a mesh.

    Args:
        mesh: Trimesh mesh to recolor.
        color: RGB or RGBA color in uint8 range.
        opacity: Opacity in ``[0, 1]``.

    Returns:
        The same mesh with updated visual colors.
    """

    color_rgba = np.asarray(color, dtype=np.uint8).copy()
    if color_rgba.shape[0] == 3:
        color_rgba = np.concatenate([color_rgba, np.asarray([255], dtype=np.uint8)])
    color_rgba[3] = int(np.clip(opacity, 0.0, 1.0) * 255)
    if len(mesh.faces) > 0:
        mesh.visual.face_colors = np.tile(color_rgba, (len(mesh.faces), 1))
    if len(mesh.vertices) > 0:
        mesh.visual.vertex_colors = np.tile(color_rgba, (len(mesh.vertices), 1))
    return mesh


def add_transparent_mesh(scene, name: str, mesh: tm.Trimesh, color: np.ndarray, opacity: float):
    """Add a possibly transparent mesh to a viser scene.

    Args:
        scene: Viser scene API used to create the mesh node.
        name: Scene path for the mesh.
        mesh: Trimesh mesh to render.
        color: RGB or RGBA color in uint8 range.
        opacity: Opacity in ``[0, 1]``.

    Returns:
        Viser mesh handle.
    """

    opacity = float(np.clip(opacity, 0.0, 1.0))
    color_rgb = tuple(int(value) for value in np.asarray(color).reshape(-1)[:3])
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

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
        return scene.add_mesh_simple(name, vertices=vertices, faces=faces, color=color_rgb, opacity=opacity)
    except (AttributeError, TypeError):
        pass

    set_mesh_color(mesh, color, opacity)
    try:
        return scene.add_mesh_trimesh(name, mesh, opacity=opacity)
    except TypeError:
        return scene.add_mesh_trimesh(name, mesh)


def link_points_to_world(visualize: Visualizer, link_name: str, points: torch.Tensor, batch_index: int) -> np.ndarray:
    """Transform link-local points to the rendered world frame.

    Args:
        visualize: Visualizer whose current status stores link transforms.
        link_name: Link frame used by the local points.
        points: Tensor of shape ``[N, 3]`` in the link frame.
        batch_index: Rendered hand copy index.

    Returns:
        NumPy array with shape ``[N, 3]`` in world coordinates.
    """

    world_points = visualize.current_status[link_name].transform_points(points)
    if len(world_points.shape) == 3:
        world_points = world_points[batch_index]
    world_points = world_points @ visualize.global_rotation[batch_index].T + visualize.global_translation[batch_index]
    return world_points.detach().cpu().numpy()


def get_link_mesh(visualize: Visualizer, link_name: str, batch_index: int, color: np.ndarray) -> tm.Trimesh | None:
    """Build one transformed link visual mesh.

    Args:
        visualize: Visualizer containing the robot mesh dictionary.
        link_name: Link whose visual mesh should be extracted.
        batch_index: Rendered hand copy index.
        color: RGBA color assigned to the link mesh.

    Returns:
        Transformed link mesh, or None when the link has no visual mesh.
    """

    if link_name not in visualize.robot_mesh:
        return None
    vertices = link_points_to_world(visualize, link_name, visualize.robot_mesh[link_name]["vertices"], batch_index)
    faces = visualize.robot_mesh[link_name]["faces"].detach().cpu().numpy()
    return tm.Trimesh(vertices=vertices, faces=faces, face_colors=color, process=False)


def build_link_sphere_meshes(
    visualize: Visualizer,
    collision_spheres: dict[str, list[dict]],
    link_name: str,
    batch_index: int,
    color: np.ndarray,
) -> list[tm.Trimesh]:
    """Build transformed collision sphere meshes for one link.

    Args:
        visualize: Visualizer with current forward kinematics.
        collision_spheres: Mapping from link names to sphere definitions.
        link_name: Link whose spheres should be transformed.
        batch_index: Rendered hand copy index.
        color: RGBA color assigned to the spheres.

    Returns:
        List of transformed sphere meshes.
    """

    meshes = []
    for sphere in collision_spheres.get(link_name, []):
        local_center = torch.tensor(sphere["center"], dtype=torch.float32).reshape(1, 3)
        world_center = link_points_to_world(visualize, link_name, local_center, batch_index).reshape(3)
        meshes.append(make_sphere_mesh(world_center, float(sphere["radius"]), color))
    return meshes


def add_pair_to_scene(
    scene,
    visualize: Visualizer,
    collision_spheres: dict[str, list[dict]],
    pair: PairSpec,
    pair_index: int,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Render one hand copy with a highlighted self-collision pair.

    Args:
        scene: Viser scene API used to create render nodes.
        visualize: Visualizer containing robot meshes and transforms.
        collision_spheres: Mapping from link names to sphere definitions.
        pair: Pair specification to highlight.
        pair_index: Batch index and scene index for this hand copy.
        args: Parsed command-line options controlling opacity and rendering.

    Returns:
        Dictionary with counts of rendered pair meshes and spheres.
    """

    rendered = {"pair_meshes": 0, "pair_spheres": 0, "context_spheres": 0}
    root_path = f"/pair_{pair_index + 1:02d}_{pair.link_a}_vs_{pair.link_b}"

    if not args.hide_robot:
        robot_mesh = visualize.get_robot_trimesh_data(i=pair_index, color=[0, 255, 0, 80])
        add_transparent_mesh(scene, f"{root_path}/robot", robot_mesh, ROBOT_COLOR, args.robot_opacity)

    for link_name, color, suffix in (
        (pair.link_a, PAIR_COLOR_A, "a"),
        (pair.link_b, PAIR_COLOR_B, "b"),
    ):
        link_mesh = get_link_mesh(visualize, link_name, pair_index, color)
        if link_mesh is not None:
            add_transparent_mesh(scene, f"{root_path}/highlight_mesh/{suffix}_{link_name}", link_mesh, color, args.link_opacity)
            rendered["pair_meshes"] += 1
        for sphere_index, sphere_mesh in enumerate(
            build_link_sphere_meshes(visualize, collision_spheres, link_name, pair_index, color)
        ):
            add_transparent_mesh(
                scene,
                f"{root_path}/highlight_spheres/{suffix}_{link_name}/{sphere_index}",
                sphere_mesh,
                color,
                args.sphere_opacity,
            )
            rendered["pair_spheres"] += 1

    if args.show_all_spheres:
        pair_links = {pair.link_a, pair.link_b}
        for link_name in sorted(collision_spheres):
            if link_name in pair_links or link_name not in visualize.current_status:
                continue
            for sphere_index, sphere_mesh in enumerate(
                build_link_sphere_meshes(visualize, collision_spheres, link_name, pair_index, CONTEXT_SPHERE_COLOR)
            ):
                add_transparent_mesh(
                    scene,
                    f"{root_path}/context_spheres/{link_name}/{sphere_index}",
                    sphere_mesh,
                    CONTEXT_SPHERE_COLOR,
                    args.context_sphere_opacity,
                )
                rendered["context_spheres"] += 1
    return rendered


def pair_markdown(pairs: list[PairSpec]) -> str:
    """Format selected pairs as a markdown table.

    Args:
        pairs: Pair specifications selected for rendering.

    Returns:
        Markdown table string for the viser GUI.
    """

    rows = ["| Copy | Link A | Link B | Latest count | Latest percent |", "|---:|---|---|---:|---:|"]
    for index, pair in enumerate(pairs, start=1):
        count = "" if pair.count is None else str(pair.count)
        percent = "" if pair.percent is None else f"{pair.percent:.2f}%"
        rows.append(f"| {index} | `{pair.link_a}` | `{pair.link_b}` | {count} | {percent} |")
    return "\n".join(rows)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line options for the self-collision pair viewer.

    Args:
        argv: Optional argument list. When None, argparse reads from sys.argv.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Visualize common self-collision link pairs in a viser web UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface used by the viser server.")
    parser.add_argument("--port", type=int, default=8082, help="Port used by the viser server.")
    parser.add_argument(
        "--hand-name",
        default="leap_sp",
        choices=sorted(DEFAULT_ROBOT_FILES),
        help="Hand preset name used for the default robot config.",
    )
    parser.add_argument("--robot-file", default=None, help="Robot config YAML path used to derive all assets.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Explicit link pairs formatted as link_a:link_b. Defaults to top leap_sp self-collision pairs.",
    )
    parser.add_argument("--max-default-pairs", type=int, default=4, help="Maximum built-in pairs rendered by default.")
    parser.add_argument("--spacing", type=float, default=None, help="Spacing between hand copies in meters.")
    parser.add_argument("--robot-opacity", type=float, default=0.22, help="Full robot mesh opacity in [0, 1].")
    parser.add_argument("--link-opacity", type=float, default=0.82, help="Highlighted link mesh opacity in [0, 1].")
    parser.add_argument("--sphere-opacity", type=float, default=0.88, help="Highlighted collision sphere opacity in [0, 1].")
    parser.add_argument(
        "--context-sphere-opacity",
        type=float,
        default=0.12,
        help="Opacity for non-pair spheres when --show-all-spheres is enabled.",
    )
    parser.add_argument("--hide-robot", action="store_true", help="Hide the translucent full robot mesh.")
    parser.add_argument("--show-all-spheres", action="store_true", help="Render non-pair collision spheres as context.")
    parser.add_argument("--dry-run", action="store_true", help="Load assets and print selected pairs without starting viser.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the viser self-collision pair viewer.

    Args:
        argv: Optional argument list. When None, argparse reads from sys.argv.

    Returns:
        None.
    """

    args = parse_args(argv)
    robot_file = resolve_robot_file(args.hand_name, args.robot_file)
    _, robot_urdf_path, mesh_dir_path, collision_sphere_path = load_robot_paths(robot_file)
    collision_spheres = load_collision_spheres(collision_sphere_path)

    visualize = Visualizer(robot_urdf_path=str(robot_urdf_path), mesh_dir_path=str(mesh_dir_path))
    initial_pose = make_hand_poses(1, len(visualize.joint_names), 0.0)
    visualize.set_robot_parameters(initial_pose)
    link_names = available_links(visualize, collision_spheres)

    pairs = [parse_pair_text(pair_text) for pair_text in args.pairs] if args.pairs else default_pairs_for_robot(
        link_names,
        args.max_default_pairs,
    )
    missing_links = sorted({link for pair in pairs for link in (pair.link_a, pair.link_b) if link not in link_names})
    if missing_links:
        raise ValueError(f"Selected pairs contain links missing from the robot model: {missing_links}")

    spacing = args.spacing
    if spacing is None:
        spacing = 0.48 if args.hand_name.startswith("dual_dummy_arm") else 0.22
    hand_pose = make_hand_poses(len(pairs), len(visualize.joint_names), spacing)
    visualize.set_robot_parameters(hand_pose)

    if args.dry_run:
        print(f"Loaded robot config: {robot_file}")
        print(f"Loaded collision spheres: {collision_sphere_path}")
        print(f"Selected {len(pairs)} pairs with spacing {spacing:.3f}m:")
        for index, pair in enumerate(pairs, start=1):
            sphere_count = len(collision_spheres.get(pair.link_a, [])) + len(collision_spheres.get(pair.link_b, []))
            mesh_count = int(pair.link_a in visualize.robot_mesh) + int(pair.link_b in visualize.robot_mesh)
            count_text = "" if pair.count is None else f", count={pair.count}, percent={pair.percent:.2f}%"
            print(
                f"  {index}. {pair.link_a} <-> {pair.link_b}"
                f" | link_meshes={mesh_count} pair_spheres={sphere_count}{count_text}"
            )
        return

    server = viser.ViserServer(host=args.host, port=args.port, label="Self-Collision Link Pairs")
    server.scene.world_axes.visible = True
    server.scene.world_axes.axes_length = 0.05
    server.scene.world_axes.axes_radius = 0.001
    server.scene.world_axes.origin_radius = 0.004

    rendered_totals = {"pair_meshes": 0, "pair_spheres": 0, "context_spheres": 0}
    for pair_index, pair in enumerate(pairs):
        rendered = add_pair_to_scene(server.scene, visualize, collision_spheres, pair, pair_index, args)
        for key, value in rendered.items():
            rendered_totals[key] += value

    server.gui.add_markdown(
        f"Robot config: `{robot_file}`\n\nCollision spheres: `{collision_sphere_path}`\n\n{pair_markdown(pairs)}"
    )
    server.gui.add_markdown(
        "Rendered "
        f"`{len(pairs)}` hand copies, `{rendered_totals['pair_meshes']}` highlighted link meshes, "
        f"`{rendered_totals['pair_spheres']}` highlighted collision spheres."
    )

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Viser web UI is running at http://{browser_host}:{args.port}")
    print(f"Loaded robot config: {robot_file}")
    print(f"Loaded collision spheres: {collision_sphere_path}")
    print("Rendered pairs:")
    for index, pair in enumerate(pairs, start=1):
        count_text = "" if pair.count is None else f", count={pair.count}, percent={pair.percent:.2f}%"
        print(f"  {index}. {pair.link_a} <-> {pair.link_b}{count_text}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

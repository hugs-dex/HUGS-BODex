import numpy as np
import trimesh


def get_scene(plot_lst):
    """
    plotly utils
    """
    xs, ys, zs = [], [], []
    for trace in plot_lst:
        if hasattr(trace, "x") and hasattr(trace, "y") and hasattr(trace, "z"):
            xs.extend(trace.x)
            ys.extend(trace.y)
            zs.extend(trace.z)

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()

    # 计算各轴范围
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    # 计算比例（归一化到最长轴为1）
    max_range = max(x_range, y_range, z_range)
    aspectratio = dict(
        x=x_range / max_range,
        y=y_range / max_range,
        z=z_range / max_range,
    )

    scene_fixed = dict(
        xaxis=dict(range=[xmin, xmax], visible=False, showgrid=False),
        yaxis=dict(range=[ymin, ymax], visible=False, showgrid=False),
        zaxis=dict(range=[zmin, zmax], visible=False, showgrid=False),
        aspectmode="manual",
        aspectratio=aspectratio,
    )
    return scene_fixed


def look_at(cam_pos, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    """
    Compute a 4x4 camera pose matrix that points from cam_pos to target.

    Args:
        cam_pos: (3,) array-like, camera position in world coordinates
        target: (3,) array-like, point to look at
        up: (3,) array-like, world up direction

    Returns:
        pose: (4,4) numpy array, camera pose matrix for pyrender
    """
    cam_pos = np.array(cam_pos, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # forward vector (from camera to target)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    # right vector
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # true up vector
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    # construct 4x4 pose matrix
    pose = np.eye(4)
    pose[:3, 0] = right  # X axis
    pose[:3, 1] = true_up  # Y axis
    pose[:3, 2] = -forward  # Z axis points forward in pyrender
    pose[:3, 3] = cam_pos  # translation

    return pose


def create_colored_axes(origin_size=0.01, axis_length=0.1, radius=0.002):
    """
    Create separate trimesh meshes: origin sphere, x-axis (red), y-axis (green), z-axis (blue).
    Correctly rotate cylinders (默认 cylinder 沿 z 轴) then translate along target axis.
    Returns: [origin_sphere, x_axis_mesh, y_axis_mesh, z_axis_mesh]
    """
    from trimesh.transformations import rotation_matrix

    meshes = []

    # 原点小球
    origin = trimesh.creation.icosphere(subdivisions=2, radius=origin_size)
    origin.visual.vertex_colors = np.tile(np.array([255, 255, 255, 255], dtype=np.uint8), (len(origin.vertices), 1))
    meshes.append(origin)

    # 基本圆柱（沿 z 轴，center 在原点，高度 = axis_length）
    cyl_z = trimesh.creation.cylinder(radius=radius, height=axis_length, sections=32)
    # Note: cylinder is centered on origin (extends from -h/2 to +h/2 along local z)

    # --- Z 轴（蓝），无需旋转，只需平移 +h/2 到正方向 ---
    z_axis = cyl_z.copy()
    z_axis.apply_translation([0.0, 0.0, axis_length / 2.0])
    z_axis.visual.vertex_colors = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), (len(z_axis.vertices), 1))
    meshes.append(z_axis)

    # --- X 轴（红）: 把 z 映到 x，绕 y 轴旋转 +90deg (pi/2) ---
    x_axis = cyl_z.copy()
    R_x = rotation_matrix(np.pi / 2.0, [0, 1, 0])  # 旋转矩阵 4x4
    x_axis.apply_transform(R_x)  # 先旋转
    # 旋转后，局部 z 对齐到世界 x，因此平移沿世界 x
    x_axis.apply_translation([axis_length / 2.0, 0.0, 0.0])
    x_axis.visual.vertex_colors = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (len(x_axis.vertices), 1))
    meshes.append(x_axis)

    # --- Y 轴（绿）: 把 z 映到 y，绕 x 轴旋转 -90deg (-pi/2) ---
    y_axis = cyl_z.copy()
    R_y = rotation_matrix(-np.pi / 2.0, [1, 0, 0])
    y_axis.apply_transform(R_y)
    # 旋转后平移沿世界 y
    y_axis.apply_translation([0.0, axis_length / 2.0, 0.0])
    y_axis.visual.vertex_colors = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (len(y_axis.vertices), 1))
    meshes.append(y_axis)

    return meshes

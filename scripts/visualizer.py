import xml.etree.ElementTree as ET
import os
import trimesh as tm
import torch
from mr_utils.pytorch3d.rotation_conversions import quaternion_to_matrix
from mr_utils.utils_calc import sciR, transformPositions, posQuat2Isometry3d
import pytorch_kinematics as pk
import numpy as np


def extract_mesh_paths_from_urdf(urdf_path):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Initialize the dictionary to store the link names and mesh file paths
    link_mesh_dict = {}

    # Iterate over each link in the URDF file
    for link in root.findall("link"):
        link_name = link.get("name")  # Extract the link name

        # Find all the visual elements in the link
        visual_elements = link.findall("visual")

        for visual in visual_elements:
            # offset
            origin = visual.find("origin")
            xyz = origin.get("xyz")
            rpy = origin.get("rpy")
            xyz = list(map(float, xyz.split()))
            rpy = list(map(float, rpy.split()))

            # mesh
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    mesh_file = mesh.get("filename")  # Extract the mesh file path

                    # Add the link name and mesh file path to the dictionary
                    link_mesh_dict[link_name] = {"mesh_file": mesh_file, "xyz": xyz, "rpy": rpy}
    return link_mesh_dict


class Visualizer:
    def __init__(self, robot_urdf_path, mesh_dir_path, device="cpu"):
        self.robot_urdf_path = robot_urdf_path

        self.chain = pk.build_chain_from_urdf(
            data=open(robot_urdf_path).read(),
        ).to(dtype=torch.float, device=device)

        self.joint_names = self.chain.get_joint_parameter_names()

        mesh_dict = extract_mesh_paths_from_urdf(robot_urdf_path)

        self.robot_mesh = {}

        for link_name, mesh_path in mesh_dict.items():
            mesh_file = os.path.join(mesh_dir_path, mesh_dict[link_name]["mesh_file"])
            link_mesh = tm.load_mesh(mesh_file, process=False)
            vertices = np.asarray(link_mesh.vertices)

            # mesh offset transformation
            xyz = mesh_dict[link_name]["xyz"]
            rpy = mesh_dict[link_name]["rpy"]
            quat = sciR.from_euler("xyz", rpy, degrees=False).as_quat()
            mesh_pose_in_link = posQuat2Isometry3d(xyz, quat)
            vertices = transformPositions(vertices, target_frame_pose_inv=mesh_pose_in_link)
            vertices = torch.tensor(vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)

            self.robot_mesh[link_name] = {"vertices": vertices, "faces": faces}

        self.hand_doa_pose = None
        self.global_translation = None
        self.global_rotation = None  # (n_batch, 3x3 matrix)
        self.current_status = None

    def set_robot_parameters(self, hand_pose, joint_names=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+4+`n_doas`) torch.FloatTensor
            translation, quaternion (w, x, y, z), and joint angles
        """
        self.global_translation = hand_pose[:, 0:3]

        # check the norm of quaternion
        eps = 1e-4
        norm = torch.norm(hand_pose[:, 3:7], dim=-1)
        assert torch.all(torch.abs(norm - 1.0) < eps)

        self.global_rotation = quaternion_to_matrix(hand_pose[:, 3:7])

        # re-order the joints
        qpos = hand_pose[:, 7:]
        chain_joint_names = self.chain.get_joint_parameter_names().copy()
        if joint_names is None:
            joint_names = chain_joint_names.copy()
        external_to_chain = [joint_names.index(name) for name in chain_joint_names]
        chain_qpos = qpos[:, external_to_chain]

        self.current_status = self.chain.forward_kinematics(chain_qpos)

    def get_robot_trimesh_data(self, i, color=None):
        """
        Get full mesh

        Returns
        -------
        data: trimesh.Trimesh
        """
        data = tm.Trimesh()
        for link_name in self.robot_mesh:
            v = self.current_status[link_name].transform_points(self.robot_mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.robot_mesh[link_name]["faces"].detach().cpu()
            data += tm.Trimesh(vertices=v, faces=f, face_colors=color)
        return data


if __name__ == "__main__":
    """
    Visualize the robot.
    """

    robot_urdf_path = "src/curobo/content/assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf"
    mesh_dir_path = "src/curobo/content/assets/robot/shadow_hand"

    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    hand_pose = torch.zeros((1, 3 + 4 + 6 + 6 + 22 + 22))
    hand_pose[:, 3] = 1.0  # quat w
    visualize.set_robot_parameters(hand_pose)

    robot_mesh = visualize.get_robot_trimesh_data(i=0)

    scene = tm.Scene(geometry=[robot_mesh])
    scene.show()

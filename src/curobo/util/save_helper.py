from typing import Dict, List
from dataclasses import dataclass
import os

import numpy as np
import torch

from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import (
    get_robot_configs_path,
    get_output_path,
    join_path,
    load_yaml,
)
from curobo.util.logger import log_warn
from curobo.util.artifact_path import portable_artifact_metadata


def dict_piece(d: Dict, piece_id: int, piece_num: int):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            new_d[k] = dict_piece(v, piece_id, piece_num)
        elif v is not None:
            each = len(v) // piece_num
            new_d[k] = v[piece_id * each : (piece_id + 1) * each]
        else:
            new_d[k] = None

    return new_d


def dict_select(d: Dict, keys: List, cpu_numpy: bool = True):
    new_d = {}
    for k in keys:
        if k in d.keys():
            new_d[k] = d[k]
            if cpu_numpy and isinstance(new_d[k], torch.Tensor):
                new_d[k] = new_d[k].cpu().numpy()
            # process sub-dictionary
            if k in ["debug_info"] and isinstance(d[k], Dict):
                for sub_k in d[k].keys():
                    new_d[k][sub_k] = d[k][sub_k]
                    if cpu_numpy and isinstance(new_d[k][sub_k], torch.Tensor):
                        new_d[k][sub_k] = new_d[k][sub_k].cpu().numpy()
        # else:
        #     new_d[k] = None
    return new_d


@dataclass
class SaveHelper:
    robot_file: str
    save_folder: str
    task_name: str
    mode: str  # npy or none
    npy_save_key: List = None
    kin_model: CudaRobotModel = None

    def __post_init__(self):
        if self.mode not in {"npy", "none"}:
            raise ValueError(f"Unsupported save mode: {self.mode}")
        if self.kin_model is None:
            robot_config_data = load_yaml(join_path(get_robot_configs_path(), self.robot_file))
            robot_config_data["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
            robot_cfg = RobotConfig.from_dict(robot_config_data)
            self.kin_model = CudaRobotModel(robot_cfg.kinematics)
            if "robot_color" in robot_config_data["robot_cfg"].keys():
                self.robot_color = robot_config_data["robot_cfg"]["robot_color"]
            else:
                self.robot_color = [0.5, 0.5, 0.2, 1.0]

        self.save_folder = os.path.abspath(os.path.join(get_output_path(), self.save_folder))
        if self.npy_save_key is None:
            self.npy_save_key = [
                "robot_pose",
                "world_cfg",
                "manip_name",
                "scene_path",
                "contact_point",
                "contact_frame",
                "contact_force",
                "grasp_error",
                "dist_error",
                "pene_error",
                "debug_info",
            ]
        return

    def save_piece(self, world_info_dict: Dict):
        file_prefix_lst = world_info_dict["save_prefix"]
        npy_dict = dict_select(world_info_dict, self.npy_save_key)
        for i, file_prefix in enumerate(file_prefix_lst):
            if self.mode == "npy":
                self._save_npy(dict_piece(npy_dict, i, len(file_prefix_lst)), file_prefix)
        return

    def exist_piece(self, file_prefix_lst: List[str]):
        if self.mode != "npy":
            return False
        for i, file_prefix in enumerate(file_prefix_lst):
            if not self._exist_npy(file_prefix):
                return False
        return True

    def _exist_npy(self, file_prefix: str):
        npy_path = os.path.join(self.save_folder, file_prefix + self.task_name + ".npy")
        return os.path.exists(npy_path)

    def _save_npy(self, save_dict: Dict, file_prefix: str):
        save_dict["joint_names"] = self.kin_model.joint_names
        save_path = os.path.join(self.save_folder, file_prefix + self.task_name + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, portable_artifact_metadata(save_dict))
        log_warn(f"Save results to {save_path}")
        return save_path

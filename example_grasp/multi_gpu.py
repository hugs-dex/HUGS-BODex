import os
import numpy as np
from glob import glob
import argparse
import sys
import math
from curobo.util_file import (
    get_manip_configs_path,
    get_output_path,
    get_assets_path,
    join_path,
    load_yaml,
    write_yaml,
)
import subprocess
import multiprocessing
import datetime


def get_matching_world_paths(world_cfg):
    template_path = join_path(get_assets_path(), world_cfg["template_path"])

    if world_cfg["type"] == "scene_cfg":
        all_paths = sorted(glob(template_path))
        object_scale_list = world_cfg.get("object_scale_list")
        if object_scale_list is not None:
            scale_patterns = [f"scale{math.floor(s * 100 + 0.5):03d}_" for s in object_scale_list]
            all_paths = [p for p in all_paths if any(pattern in p for pattern in scale_patterns)]
        return all_paths

    if world_cfg["type"] == "grasp":
        return sorted(glob(template_path, recursive=True))

    raise NotImplementedError(f"Unsupported world type: {world_cfg['type']}")


def worker(
    gpu_id,
    task,
    manip_path,
    save_folder,
    output_path,
    save_mode,
    parallel_world,
    save_data,
    save_id,
    save_debug,
    skip,
):
    with open(output_path, "w") as output_file:
        if task == "grasp":
            cmd = [
                sys.executable,
                "example_grasp/plan_batch_env.py",
                "-c",
                manip_path,
                "-f",
                save_folder,
                "-m",
                save_mode,
                "-w",
                str(parallel_world),
                "-d",
                save_data,
            ]
            if save_id is not None:
                cmd.extend(["-i", *[str(v) for v in save_id]])
            if save_debug:
                cmd.append("--save_debug")
            if not skip:
                cmd.append("-k")
            result = subprocess.call(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                stdout=output_file,
                stderr=output_file,
            )
        else:
            cmd = [
                sys.executable,
                "example_grasp/plan_mogen_batch.py",
                "-c",
                manip_path,
                "-f",
                save_folder,
                "-m",
                save_mode,
                "-t",
                task,
            ]
            if not skip:
                cmd.append("-k")
            result = subprocess.call(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                stdout=output_file,
                stderr=output_file,
            )
    if result != 0:
        raise SystemExit(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Split the configured world range across GPUs and launch one worker per GPU. "
            "Current limitation: this script does not rebalance work based on existing outputs. "
            "It first assigns file ranges to GPUs, then each GPU applies skipping independently."
        )
    )

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default="fc_leap.yml",
        help="config file path",
    )

    parser.add_argument(
        "-t",
        "--task",
        choices=["grasp", "mogen", "grasp_and_mogen"],
        default="grasp",
    )

    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["npy", "none"],
        default="npy",
    )

    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="If None, use join_path(manip_cfg_file[:-4], $TIME) as save_folder",
    )

    parser.add_argument(
        "-d",
        "--save_data",
        default="all",
        help="Which grasp results to save",
    )

    parser.add_argument(
        "-i",
        "--save_id",
        type=int,
        nargs="+",
        default=None,
        help="Which grasp results to save",
    )

    parser.add_argument(
        "-debug",
        "--save_debug",
        action="store_true",
        help="Whether to save contact normal debug data for grasp",
    )

    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="parallel world num (only used when task=grasp)",
    )

    parser.add_argument(
        "-p",
        "--exp_name",
        type=str,
        default=None,
        help="If None, use exp_name in manip_cfg_file.",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help=(
            "If True, skip existing files. (default: True) Note: skipping happens independently "
            "inside each GPU worker after ranges have already been assigned, so existing files "
            "are not redistributed across GPUs."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Override world.start from the manipulation config before splitting across GPUs.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Override world.end from the manipulation config before splitting across GPUs.",
    )

    parser.add_argument("-g", "--gpu", nargs="+", required=True, help="gpu id list")
    args = parser.parse_args()

    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    if args.exp_name is not None:
        manip_config_data["exp_name"] = args.exp_name

    if args.start is not None:
        manip_config_data["world"]["start"] = args.start

    if args.end is not None:
        manip_config_data["world"]["end"] = args.end

    world_start = manip_config_data["world"]["start"]
    world_end = manip_config_data["world"]["end"]
    original_start = 0 if world_start is None else world_start
    matching_world_paths = get_matching_world_paths(manip_config_data["world"])

    all_obj_num = len(matching_world_paths[original_start:world_end])
    obj_num_lst = np.array([all_obj_num // len(args.gpu)] * len(args.gpu))
    obj_num_lst[: (all_obj_num % len(args.gpu))] += 1
    assert obj_num_lst.sum() == all_obj_num

    print(
        "multi_gpu scheduling note: file ranges are assigned to GPUs before skip checks. "
        "Existing outputs are not redistributed across GPUs; each GPU skips its own assigned files."
    )

    p_list = []
    if args.save_folder is not None:
        save_folder = args.save_folder
    elif manip_config_data["exp_name"] is not None:
        save_folder = os.path.join(args.manip_cfg_file[:-4], manip_config_data["exp_name"])
    else:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )

    runinfo_folder = os.path.join(get_output_path(), save_folder, "runinfo")
    os.makedirs(runinfo_folder, exist_ok=True)

    # create tmp manip cfg files
    for i, gpu_id in enumerate(args.gpu):
        if obj_num_lst[i] == 0:
            continue
        new_manip_path = join_path(runinfo_folder, str(i) + "_config.yml")
        manip_config_data["world"]["start"] = int(original_start + (obj_num_lst[:i]).sum())
        manip_config_data["world"]["end"] = int(original_start + (obj_num_lst[: (i + 1)]).sum())
        write_yaml(manip_config_data, new_manip_path)

        output_path = join_path(runinfo_folder, str(i) + "_output.txt")

        p = multiprocessing.Process(
            target=worker,
            args=(
                gpu_id,
                args.task,
                new_manip_path,
                save_folder,
                output_path,
                args.save_mode,
                args.parallel_world,
                args.save_data,
                args.save_id,
                args.save_debug,
                args.skip,
            ),
        )
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)

    for p in p_list:
        p.join()
    failed = [(p.pid, p.exitcode) for p in p_list if p.exitcode != 0]
    if failed:
        raise SystemExit(f"GPU workers failed (pid, exit code): {failed}; logs: {runinfo_folder}")

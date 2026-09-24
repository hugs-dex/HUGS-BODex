"""Check source syntax and the supported robot asset closure without a GPU."""

import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    content = root / "src/curobo/content"
    configs = content / "configs"
    assets = content / "assets"
    checked = {}

    def require(path):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Asset escapes checkout: {path}")
        data = path.read_bytes()
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Unresolved LFS pointer: {path}")
        checked[str(path.relative_to(root))] = {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
        }

    python_count = 0
    for directory in ("src", "example_grasp", "scripts"):
        for path in (root / directory).rglob("*.py"):
            if "build" in path.parts:
                continue
            ast.parse(path.read_text(), filename=str(path))
            python_count += 1
    yaml_count = 0
    for directory in (root / "cfg", configs):
        for path in directory.rglob("*"):
            if path.suffix in (".yaml", ".yml"):
                yaml.safe_load(path.read_text())
                yaml_count += 1
    modes = []
    for suite in ("sim_shadow.yml", "sim_leap_sp.yml"):
        suite_path = configs / "manip" / suite
        require(suite_path)
        data = yaml.safe_load(suite_path.read_text())
        assert [v["type_id"] for v in data["grasp_types"].values()] == [1, 2, 3, 4, 5]
        for name, spec in data["grasp_types"].items():
            manip_path = configs / "manip" / spec["manip_cfg_file"]
            require(manip_path)
            manip = yaml.safe_load(manip_path.read_text())
            robot_path = configs / "robot" / manip["robot_file"]
            require(robot_path)
            robot = yaml.safe_load(robot_path.read_text())["robot_cfg"]["kinematics"]
            for key in ("collision_spheres", "hand_pose_transfer_path"):
                require(configs / "robot" / robot[key])
            require(configs / "robot" / spec["transfer_file"])
            for key in ("base_cfg_file", "particle_file", "gradient_file"):
                matches = list(configs.rglob(manip[key]))
                assert matches, (key, manip[key])
                for path in matches:
                    require(path)
            urdf = assets / robot["urdf_path"]
            require(urdf)
            for node in ET.parse(urdf).iter("mesh"):
                require(urdf.parent / node.attrib["filename"])
            modes.append({"suite": suite, "mode": name, "robot": manip["robot_file"]})
    print(json.dumps({"python_files": python_count, "yaml_files": yaml_count,
                      "modes": modes, "assets_and_configs": checked}, indent=2))


if __name__ == "__main__":
    main()

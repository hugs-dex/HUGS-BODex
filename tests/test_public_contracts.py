"""Regression checks for the public entrypoints, without a GPU workload."""

import ast
from collections import defaultdict
import importlib.util
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace=None):
    """Load CPU-only definitions without importing the CUDA solver stack."""
    tree = ast.parse((ROOT / path).read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + body, type_ignores=[])), path, "exec"), ns)
    return ns


def test_removed_options_and_legacy_aliases():
    for path, constant in [("src/task/synthesis.py", "INIT_SOURCE_CHOICES"),
                           ("example_grasp/plan_batch_env_human_prior.py", "INIT_SOURCE_ALIASES")]:
        tree = ast.parse((ROOT / path).read_text())
        value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets))
        fn = load_definitions(path, {"canonical_init_source"}, {constant: value})["canonical_init_source"]
        for mode in ("surface_sample", "human"):
            assert fn(mode) == mode
        with pytest.raises(ValueError):
            fn("human_surface")
        if constant == "INIT_SOURCE_ALIASES":
            assert fn("heuristic") == "surface_sample"
            assert fn("human_prior") == "human"
        else:
            with pytest.raises(ValueError):
                fn("human_prior")


def test_assignment_is_unique_and_empty_safe():
    ns = load_definitions("src/task/synthesis.py", {"SynthesisRuntime"}, {"defaultdict": defaultdict})
    runtime = ns["SynthesisRuntime"].__new__(ns["SynthesisRuntime"])
    assert runtime.assign_batches_to_gpus([], [2, 3]) == {2: [], 3: []}
    batches = [{"estimated_cost": i+1, "scene_count": 1, "type_name": str(i), "id": i} for i in range(13)]
    result = runtime.assign_batches_to_gpus(batches, [2, 3])
    assert sorted(b["id"] for group in result.values() for b in group) == list(range(13))
    assert all(result.values())


def test_legacy_worker_propagates_real_child_failure(tmp_path):
    script = tmp_path / "example_grasp"
    script.mkdir()
    (script / "plan_batch_env.py").write_text("print('controlled child failure', flush=True)\nraise SystemExit(23)\n")
    ns = load_definitions("example_grasp/multi_gpu.py", {"worker"}, {"os": os, "sys": sys, "subprocess": subprocess})
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        process = multiprocessing.get_context("fork").Process(target=ns["worker"], args=(2, "grasp", "unused", "unused", str(tmp_path / "worker.log"), "npy", 1, "final", None, False, True))
        process.start()
        process.join(15)
        assert process.exitcode == 23
        assert "controlled child failure" in (tmp_path / "worker.log").read_text()
    finally:
        os.chdir(cwd)


def test_shell_leap_sp_and_failure(tmp_path):
    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit "${CHILD_STATUS:-0}"\n')
    stub.chmod(0o755)
    log = tmp_path / "calls"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], CALL_LOG=str(log))
    command = ["bash", str(ROOT / "scripts/run_all_grasps_multi_gpu.sh"), "--hand", "leap_sp", "--gpus", "2", "3", "-k"]
    subprocess.run(command, env=env, check=True, capture_output=True)
    lines = log.read_text().splitlines()
    assert len(lines) == 5
    assert all("leap_sp/" in line and "-k" in line for line in lines)
    log.write_text("")
    result = subprocess.run(command, env=dict(env, CHILD_STATUS="23"), capture_output=True)
    assert result.returncode != 0
    assert len(log.read_text().splitlines()) == 1


def test_legacy_parent_propagates_workers_and_keeps_logs(tmp_path):
    import yaml
    scripts = tmp_path / "example_grasp"
    scripts.mkdir()
    (scripts / "plan_batch_env.py").write_text("print('intentional failure', flush=True)\nraise SystemExit(19)\n")
    for i in range(2):
        (tmp_path / f"scene{i}.npy").touch()
    config = tmp_path / "manip.yml"
    config.write_text(yaml.safe_dump({"world": {"type": "scene_cfg", "template_path": str(tmp_path / "*.npy"), "start": None, "end": None}, "exp_name": "failure"}))
    result = subprocess.run([sys.executable, str(ROOT / "example_grasp/multi_gpu.py"),
                             "-c", str(config), "-f", str(tmp_path / "results"),
                             "-g", "2", "3"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "GPU workers failed" in result.stderr
    logs = list((tmp_path / "results/runinfo").glob("*_output.txt"))
    assert len(logs) == 2
    assert all("intentional failure" in p.read_text() for p in logs)


def test_artifact_relocation_is_explicit(monkeypatch):
    from curobo.util.artifact_path import resolve_artifact_path
    monkeypatch.setenv("HUGS_PATH_MAP", '{"/old/assets": "/new/assets"}')
    assert resolve_artifact_path("/old/assets/object/a.npy") == "/new/assets/object/a.npy"
    assert resolve_artifact_path("/old/assets_extra/a.npy") == "/old/assets_extra/a.npy"


def test_npy_none_and_skip_without_usd(tmp_path):
    assert importlib.util.find_spec("pxr") is None
    from curobo.util.save_helper import SaveHelper
    model = SimpleNamespace(joint_names=["joint"])
    helper = SaveHelper("unused", str(tmp_path), "grasp", "npy", kin_model=model)
    payload = {"save_prefix": ["scene_"], "robot_pose": np.zeros((1, 2, 1, 1)), "scene_path": ["scene.npy"]}
    assert not helper.exist_piece(["scene_"])
    helper.save_piece(payload)
    assert helper.exist_piece(["scene_"])
    saved = np.load(tmp_path / "scene_grasp.npy", allow_pickle=True).item()
    assert saved["joint_names"] == ["joint"]
    assert saved["robot_pose"].shape == (1, 2, 1, 1)
    helper = SaveHelper("unused", str(tmp_path / "none"), "grasp", "none", kin_model=model)
    helper.save_piece(payload)
    assert not helper.exist_piece(["scene_"])
    assert not (tmp_path / "none").exists()
    for invalid in ("usd", "usd+npy", "invalid"):
        with pytest.raises(ValueError):
            SaveHelper("unused", str(tmp_path), "grasp", invalid, kin_model=model)

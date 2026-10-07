"""Regression checks for the public entrypoints, without a GPU workload."""

import ast
from collections import defaultdict
import importlib.util
from pathlib import Path
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


def test_synthesis_init_sources():
    path = "src/task/synthesis.py"
    constant = "INIT_SOURCE_CHOICES"
    tree = ast.parse((ROOT / path).read_text())
    value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets))
    fn = load_definitions(path, {"canonical_init_source"}, {constant: value})["canonical_init_source"]
    for mode in ("surface_sample", "human"):
        assert fn(mode) == mode
    for mode in ("human_surface", "human_prior", "heuristic"):
        with pytest.raises(ValueError):
            fn(mode)


def test_assignment_is_unique_and_empty_safe():
    ns = load_definitions("src/task/synthesis.py", {"SynthesisRuntime"}, {"defaultdict": defaultdict})
    runtime = ns["SynthesisRuntime"].__new__(ns["SynthesisRuntime"])
    assert runtime.assign_batches_to_gpus([], [2, 3]) == {2: [], 3: []}
    batches = [{"estimated_cost": i+1, "scene_count": 1, "type_name": str(i), "id": i} for i in range(13)]
    result = runtime.assign_batches_to_gpus(batches, [2, 3])
    assert sorted(b["id"] for group in result.values() for b in group) == list(range(13))
    assert all(result.values())


@pytest.mark.parametrize("save_data, horizons", [
    ("all", [0, 1, 2, 3, 4]),
    ("init", [0]),
    ("final", [4]),
    ("select_3", [0, 2, 4]),
    ("pregrasp_and_grasp", [1, 4]),
])
@pytest.mark.parametrize("save_id", [None, [1]])
def test_native_debug_trajectory_selection(save_data, horizons, save_id):
    import torch

    ns = load_definitions("src/task/synthesis.py", {"SynthesisRuntime"}, {"torch": torch})
    runtime = ns["SynthesisRuntime"].__new__(ns["SynthesisRuntime"])
    runtime.args = SimpleNamespace(save_data=save_data, save_id=save_id, save_debug=True)
    poses = torch.arange(2 * 5 * 4).reshape(2, 5, 4)
    hand_points = torch.arange(2 * 5 * 2 * 3).reshape(2, 5, 2, 3).float()
    object_points = torch.arange(2 * 5 * 3 * 3).reshape(2, 5, 3, 3).float()
    stages = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])
    expected = {"hp": hand_points, "grad": hand_points + 1,
                "op": object_points, "debug_posi": object_points + 2,
                "debug_normal": object_points + 3, "contact_stage": stages}
    solver_debug = {key: [list(value.unbind(dim=1))] for key, value in expected.items()}
    solver_debug["steps"] = [[poses[:, :2], poses[:, 2:]]]
    result = SimpleNamespace(debug_info={"solver": solver_debug})
    config = {"grasp_contact_strategy": {"pregrasp_stage": 0, "grasp_stage": 1}}

    selected_poses, debug = runtime.process_grasp_result(result, config)

    indices = [0, 1] if save_id is None else save_id
    torch.testing.assert_close(selected_poses, poses[indices][:, horizons])
    for key, value in expected.items():
        selected = value[indices][:, horizons].reshape((-1,) + value.shape[2:])
        torch.testing.assert_close(debug[key], selected * 100 if key == "grad" else selected)
    world_info = {"world_model": [object()]}
    runtime.attach_result_to_world_info(world_info, result, None, config)
    torch.testing.assert_close(world_info["robot_pose"], selected_poses.unsqueeze(0))
    assert set(world_info["debug_info"]) == set(expected)


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


def test_native_workers_propagate_controlled_failure(tmp_path, monkeypatch):
    from src.task import synthesis
    monkeypatch.setattr(synthesis, '_resolve_hydra_output_dir', lambda: str(tmp_path))
    args = SimpleNamespace(progress=False, profile=False, solver_reuse='batch')
    runtime = synthesis.SynthesisRuntime.__new__(synthesis.SynthesisRuntime)
    runtime.args = args
    batches = [{'estimated_cost': 1, 'scene_count': 1, 'type_name': str(i),
                'manip_cfg_file': 'missing-controlled-test-config.yml', 'type_budget': 1}
               for i in range(2)]
    with pytest.raises(RuntimeError, match='Multi-GPU synthesis failed'):
        runtime.run_multi_gpu({'batches': batches, 'types': []}, [2, 3])
    logs = list((tmp_path/'gpu_logs').glob('gpu_*.log'))
    assert len(logs) == 2
    assert all('Traceback' in p.read_text() for p in logs)


def test_dataset_metadata_relocation(tmp_path, monkeypatch):
    from curobo.util.artifact_path import portable_artifact_metadata, resolve_artifact_path
    old_root, new_root = tmp_path / 'dataset-a', tmp_path / 'dataset-b'
    monkeypatch.setenv('HUGS_DATASET_ROOT', str(old_root))
    monkeypatch.delenv('HUGS_PATH_MAP', raising=False)
    record = {'scene_path': np.array([str(old_root / 'object/DGN_2k/scene_cfg/a.npy')]),
              'world_cfg': {'mesh': {'file_path': str(old_root / 'object/DGN_2k/mesh.obj')}},
              'robot_pose': np.arange(7)}
    saved = portable_artifact_metadata(record)
    assert saved['path_root'] == 'HUGS_DATASET_ROOT'
    assert saved['scene_path'].tolist() == ['object/DGN_2k/scene_cfg/a.npy']
    assert saved['robot_pose'] is record['robot_pose']
    monkeypatch.setenv('HUGS_DATASET_ROOT', str(new_root))
    assert resolve_artifact_path(saved['scene_path'][0]) == str(new_root / saved['scene_path'][0])
    assert resolve_artifact_path(saved['world_cfg']['mesh']['file_path']) == str(new_root / 'object/DGN_2k/mesh.obj')

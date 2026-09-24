import unittest
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]
BASELINE_ROOT = REPOSITORY_ROOT / "src" / "curobo" / "content" / "configs" / "baselines"
MANIP_ROOT = REPOSITORY_ROOT / "src" / "curobo" / "content" / "configs" / "manip"
SHADOW_MANIP_CONFIGS = (
    "sim_shadow/tabletop_two.yml",
    "sim_shadow/tabletop_three.yml",
    "sim_shadow/tabletop_full.yml",
    "sim_dual_dummy_arm_shadow/tabletop_three.yml",
    "sim_dual_dummy_arm_shadow/tabletop_full.yml",
)


def load_yaml(path: Path) -> dict:
    """Load one YAML mapping used by the Issue #89 config tests."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class Issue89BudgetConfigTest(unittest.TestCase):
    def test_learned_prior_budgets_and_replacement(self):
        expected = {
            "issue89_human_shadow_k80.yml": (80, 40, 8),
            "issue89_human_shadow_k160.yml": (160, 80, 16),
        }
        for filename, budget_tuple in expected.items():
            with self.subTest(filename=filename):
                config = load_yaml(BASELINE_ROOT / filename)
                human_prior = config["overrides"]["suite"]["human_prior"]
                self.assertEqual(
                    (
                        human_prior["total_budget"],
                        human_prior["max_type_budget"],
                        human_prior["budget_resolution"],
                    ),
                    budget_tuple,
                )
                self.assertEqual(human_prior["min_type_budget"], 0)
                self.assertEqual(human_prior["score_threshold"], 0.0)
                self.assertEqual(human_prior["budget_rounding_mode"], "round")
                self.assertFalse(human_prior["replacement"])

    def test_random_overrides_scale_all_five_active_types(self):
        for filename, seed_num in (
            ("issue89_random_shadow_seed40.yml", 40),
            ("issue89_random_shadow_seed80.yml", 80),
        ):
            with self.subTest(filename=filename):
                config = load_yaml(BASELINE_ROOT / filename)
                manip = config["overrides"]["manip"]
                self.assertEqual(set(manip), set(SHADOW_MANIP_CONFIGS))
                self.assertEqual({values["seed_num"] for values in manip.values()}, {seed_num})

    def test_default_manipulation_seed_counts_remain_20(self):
        for relative_path in SHADOW_MANIP_CONFIGS:
            with self.subTest(relative_path=relative_path):
                config = load_yaml(MANIP_ROOT / relative_path)
                self.assertEqual(config["seed_num"], 20)


if __name__ == "__main__":
    unittest.main()

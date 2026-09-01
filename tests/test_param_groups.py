from __future__ import annotations

import unittest

try:
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - exercised in torch-free envs
    nn = None

if nn is not None:
    from utils.utils import build_param_lr_groups
else:
    build_param_lr_groups = None

try:
    from training.trainer import WorldModelTrainer
except ModuleNotFoundError:  # pragma: no cover - optional training deps may be absent
    WorldModelTrainer = None


@unittest.skipIf(nn is None, "torch is required for parameter group tests")
class ParamGroupTest(unittest.TestCase):
    def test_bias_and_norm_are_no_decay(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False))

        groups = build_param_lr_groups(model, {"base": 1e-4})

        no_decay = [group for group in groups if group.get("weight_decay") == 0.0]
        decay = [group for group in groups if group.get("weight_decay") != 0.0]
        self.assertTrue(no_decay)
        self.assertTrue(decay)
        self.assertTrue(any(group["name"] == "base.no_decay" for group in no_decay))


@unittest.skipIf(WorldModelTrainer is None, "training dependencies are required")
class TrainerCheckpointConfigTest(unittest.TestCase):
    def test_parse_checkpoint_steps(self):
        self.assertEqual(
            WorldModelTrainer._parse_checkpoint_steps([100, "200", 0, -5, 100]),
            [100, 200],
        )
        self.assertEqual(
            WorldModelTrainer._parse_checkpoint_steps("300, 400  300"),
            [300, 400],
        )
        self.assertEqual(WorldModelTrainer._parse_checkpoint_steps(None), [])


if __name__ == "__main__":
    unittest.main()

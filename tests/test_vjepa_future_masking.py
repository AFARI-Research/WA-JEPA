from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in torch-free envs
    torch = None

if torch is not None:
    from models.vjepa_future_masking import FutureMaskSampler
else:
    FutureMaskSampler = None


@unittest.skipIf(torch is None, "torch is required for V-JEPA masking tests")
class FutureMaskSamplerTest(unittest.TestCase):
    def test_official_like_returns_multiple_future_only_mask_groups(self):
        torch.manual_seed(17)
        sampler = FutureMaskSampler(
            num_history_frames=2,
            num_future_frames=4,
            image_size=(64, 64),
            patch_size=16,
            tubelet_size=2,
            official_mask_configs=[
                {
                    "num_blocks": 2,
                    "spatial_scale": (0.25, 0.25),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (1.0, 1.0),
                },
                {
                    "num_blocks": 1,
                    "spatial_scale": (0.5, 0.5),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (1.0, 1.0),
                },
            ],
        )

        batch = sampler.sample(5, device=torch.device("cpu"))

        self.assertEqual(len(batch.masks_x), 2)
        self.assertEqual(len(batch.masks_y), 2)
        self.assertEqual(batch.mask_indices, [0, 0])
        self.assertEqual(batch.mask_ratios.shape, (2, 5))
        self.assertEqual(batch.full_mask.shape, (2, 5))
        for masks_x, masks_y in zip(batch.masks_x, batch.masks_y):
            self.assertEqual(masks_x.shape[0], 5)
            self.assertEqual(masks_y.shape[0], 5)
            self.assertTrue(torch.all(masks_x[:, : sampler.num_history_tokens] < sampler.num_history_tokens))
            self.assertTrue(torch.all(masks_y >= sampler.num_history_tokens))
            self.assertTrue(any(not torch.equal(masks_y[0], row) for row in masks_y[1:]))

    def test_batch_shapes_are_fixed_with_min_keep_truncation(self):
        torch.manual_seed(23)
        sampler = FutureMaskSampler(
            num_history_frames=2,
            num_future_frames=4,
            image_size=(64, 64),
            patch_size=16,
            tubelet_size=2,
            official_mask_configs=[
                {
                    "num_blocks": 3,
                    "spatial_scale": (0.25, 0.25),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (0.75, 1.5),
                },
            ],
        )

        batch = sampler.sample(8, device=torch.device("cpu"))

        self.assertEqual(len(batch.masks_x), 1)
        self.assertEqual(batch.masks_x[0].shape[0], 8)
        self.assertEqual(batch.masks_y[0].shape[0], 8)
        self.assertTrue(torch.all(batch.mask_ratios == batch.mask_ratios[0, 0]))
        self.assertGreater(batch.masks_y[0].shape[1], 0)
        self.assertGreaterEqual(batch.masks_x[0].shape[1], sampler.num_history_tokens)

    def test_full_future_mask_reuses_full_mask_token(self):
        sampler = FutureMaskSampler(
            num_history_frames=2,
            num_future_frames=2,
            image_size=(32, 32),
            patch_size=16,
            tubelet_size=2,
        )

        batch = sampler.full_future_mask(2, device=torch.device("cpu"))

        self.assertEqual(batch.mask_indices, [0])
        self.assertEqual(batch.num_history_tokens, 4)
        self.assertEqual(batch.num_future_tokens, 4)
        self.assertTrue(bool(torch.all(batch.full_mask)))
        self.assertTrue(bool(torch.all(batch.mask_ratios == 1.0)))
        self.assertTrue(torch.equal(batch.masks_x[0], torch.tensor([[0, 1, 2, 3]] * 2)))
        self.assertTrue(torch.equal(batch.masks_y[0], torch.tensor([[4, 5, 6, 7]] * 2)))

    def test_max_keep_limits_visible_future_context_not_targets(self):
        torch.manual_seed(31)
        sampler = FutureMaskSampler(
            num_history_frames=2,
            num_future_frames=4,
            image_size=(64, 64),
            patch_size=16,
            tubelet_size=2,
            official_mask_configs=[
                {
                    "num_blocks": 1,
                    "spatial_scale": (0.25, 0.25),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (1.0, 1.0),
                    "max_keep": 3,
                }
            ],
        )

        batch = sampler.sample(4, device=torch.device("cpu"))

        self.assertEqual(batch.masks_x[0].shape[1], sampler.num_history_tokens + 3)
        self.assertGreater(batch.masks_y[0].shape[1], 3)

    def test_invalid_configs_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "spatial_scale"):
            FutureMaskSampler(
                num_history_frames=2,
                num_future_frames=2,
                image_size=(32, 32),
                patch_size=16,
                tubelet_size=2,
                official_mask_configs=[
                    {
                        "num_blocks": 1,
                        "spatial_scale": (0.8, 0.2),
                    }
                ],
            )

        with self.assertRaisesRegex(ValueError, "num_blocks"):
            FutureMaskSampler(
                num_history_frames=2,
                num_future_frames=2,
                image_size=(32, 32),
                patch_size=16,
                tubelet_size=2,
                official_mask_configs=[{"num_blocks": 0}],
            )

    def test_empty_batch_fails_fast(self):
        sampler = FutureMaskSampler(
            num_history_frames=2,
            num_future_frames=2,
            image_size=(32, 32),
            patch_size=16,
            tubelet_size=2,
        )

        with self.assertRaisesRegex(ValueError, "batch_size"):
            sampler.sample(0, device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            sampler.full_future_mask(0, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

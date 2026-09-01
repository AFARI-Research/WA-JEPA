from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in torch-free envs
    torch = None

if torch is not None:
    from models.multiview_causal_future_jepa import (
        ModalitySpecificJointAttention,
        MultiViewCausalFutureMaskedJEPA,
    )
else:
    ModalitySpecificJointAttention = None
    MultiViewCausalFutureMaskedJEPA = None

requires_torch = unittest.skipIf(torch is None, "torch is required for MultiViewCausalFutureMaskedJEPA tests")


def _tiny_mv_model(**overrides) -> MultiViewCausalFutureMaskedJEPA:
    save_root = overrides.pop("_save_root", None)
    cfg = {
        "variant": "vit_tiny",
        "use_rope": True,
        "use_sdpa": True,
        "input_hw": [32, 32],
        "num_history_frames": 2,
        "num_future_frames": 2,
        "patch_size": 16,
        "tubelet_size": 2,
        "camera_names": ["cam_l0", "cam_f0", "cam_r0", "cam_b0"],
        "mv_scene_dim": 64,
        "trajectory_horizon": 2,
        "flow_hidden_dim": 64,
        "flow_num_layers": 1,
        "flow_num_heads": 4,
        "ema_start": 0.9,
        "ema_end": 0.9,
    }
    cfg.update(overrides)
    full_cfg = {"data": {"num_future_frames": 2, "num_history_trajectory_steps": 2}}
    if save_root is not None:
        full_cfg["save_root"] = save_root
    model = MultiViewCausalFutureMaskedJEPA(cfg, full_cfg=full_cfg)
    model.eval()
    return model


def _mv_batch(batch_size: int = 2, num_cameras: int = 4) -> dict[str, torch.Tensor]:
    return {
        "history_images": torch.randn(batch_size, 2, num_cameras, 3, 32, 32),
        "future_images": torch.randn(batch_size, 2, num_cameras, 3, 32, 32),
        "ego_status": torch.randn(batch_size, 8),
        "history_trajectory": torch.randn(batch_size, 2, 3),
        "future_trajectory": torch.randn(batch_size, 2, 3),
    }


def _single_view_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "history_images": torch.randn(batch_size, 2, 3, 32, 32),
        "future_images": torch.randn(batch_size, 2, 3, 32, 32),
        "ego_status": torch.randn(batch_size, 8),
        "history_trajectory": torch.randn(batch_size, 2, 3),
        "future_trajectory": torch.randn(batch_size, 2, 3),
    }


@requires_torch
class MultiViewCausalFutureMaskedJEPATest(unittest.TestCase):
    def test_modality_attention_defaults_stop_scene_trajectory_cross_gradients(self):
        torch.manual_seed(0)
        attn = ModalitySpecificJointAttention(16, 4, enable_trajectory_head=True)
        context = torch.randn(2, 3, 16, requires_grad=True)
        scene = torch.randn(2, 5, 16, requires_grad=True)
        traj = torch.randn(2, 2, 16, requires_grad=True)

        _context_out, scene_out, traj_out = attn(context, scene, traj)
        scene_out.float().sum().backward(retain_graph=True)

        self.assertIsNotNone(scene.grad)
        self.assertIsNone(traj.grad)

        context.grad = None
        scene.grad = None
        traj.grad = None
        traj_out.float().sum().backward()

        self.assertIsNone(scene.grad)
        self.assertIsNotNone(traj.grad)

    def test_modality_attention_can_enable_scene_trajectory_cross_gradients(self):
        torch.manual_seed(0)
        attn = ModalitySpecificJointAttention(
            16,
            4,
            enable_trajectory_head=True,
            traj_loss_grad_to_scene_flow=True,
            scene_loss_grad_to_traj_flow=True,
        )
        context = torch.randn(2, 3, 16, requires_grad=True)
        scene = torch.randn(2, 5, 16, requires_grad=True)
        traj = torch.randn(2, 2, 16, requires_grad=True)

        _context_out, scene_out, traj_out = attn(context, scene, traj)
        scene_out.float().sum().backward(retain_graph=True)

        self.assertIsNotNone(traj.grad)
        self.assertGreater(float(traj.grad.abs().sum().item()), 0.0)

        context.grad = None
        scene.grad = None
        traj.grad = None
        traj_out.float().sum().backward()

        self.assertIsNotNone(scene.grad)
        self.assertGreater(float(scene.grad.abs().sum().item()), 0.0)

    def test_modality_attention_forward_matches_single_joint_attention(self):
        torch.manual_seed(0)
        attn = ModalitySpecificJointAttention(32, 4, enable_trajectory_head=True)
        context = torch.randn(2, 3, 32)
        scene = torch.randn(2, 5, 32)
        traj = torch.randn(2, 2, 32)

        context_len = context.size(1)
        scene_len = scene.size(1)
        traj_len = traj.size(1)
        context_qkv = attn.context_proj.pre_attention(context)
        scene_qkv = attn.scene_proj.pre_attention(scene)
        traj_qkv = attn.traj_proj.pre_attention(traj)
        q, k, v = tuple(
            torch.cat([context_qkv[i], scene_qkv[i], traj_qkv[i]], dim=2)
            for i in range(3)
        )
        old = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        old_context, old_scene, old_traj = old.split([context_len, scene_len, traj_len], dim=2)

        def merge_heads(tensor: torch.Tensor) -> torch.Tensor:
            bsz, _, num_tokens, _ = tensor.shape
            return tensor.transpose(1, 2).reshape(bsz, num_tokens, attn.dim)

        old_context = attn.context_proj.post_attention(merge_heads(old_context))
        old_scene = attn.scene_proj.post_attention(merge_heads(old_scene))
        old_traj = attn.traj_proj.post_attention(merge_heads(old_traj))

        new_context, new_scene, new_traj = attn(context, scene, traj)

        self.assertTrue(torch.allclose(new_context, old_context, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(new_scene, old_scene, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(new_traj, old_traj, atol=1e-6, rtol=1e-6))

    def test_multiview_forward_smoke(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        out = model(_mv_batch())
        self.assertEqual(
            set(out["loss"]),
            {
                "loss_jepa_pred",
                "loss_traj",
            },
        )
        for loss in out["loss"].values():
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertEqual(loss.dim(), 0)
        self.assertNotIn("num_cameras", out["other_log"])
        for key in (
            "active_scene_xpred_mse",
            "active_traj_xpred_mse",
            "active_traj_ade",
            "active_traj_fde",
            "active_mask_ratio_mean",
            "active_mask_full_frac",
            "active_context_tokens",
            "active_visible_future_frac",
            "encoder_frozen",
            "target_scene_ema_momentum",
            "active_flow_t_mean",
            "active_flow_t_std",
            "active_scene_token_collapse_cos",
            "active_scene_token_collapse_gap",
            "active_scene_token_teacher_cos",
            "active_scene_token_delta_ratio",
            "active_scene_token_teacher_delta_norm",
        ):
            self.assertIn(key, out["other_log"])
            self.assertTrue(bool(torch.isfinite(out["other_log"][key])))
        self.assertIn(float(out["other_log"]["encoder_frozen"].item()), (0.0, 1.0))
        self.assertNotIn("random_scene_xpred_mse", out["other_log"])
        self.assertNotIn("full_future_scene_xpred_mse", out["other_log"])
        self.assertNotIn("traj_xpred_mse", out["other_log"])
        self.assertNotIn("traj_ade", out["other_log"])
        self.assertNotIn("traj_fde", out["other_log"])

    def test_single_camera_5d_images_use_multiview_path(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(camera_names=["cam_f0"])
        batch = _single_view_batch()

        out = model(batch)
        traj = model.predict_trajectory(batch)

        self.assertEqual(model.num_cameras, 1)
        self.assertEqual(model.predictor.num_scene_tokens, model.mask_sampler.num_future_tokens)
        self.assertEqual(set(out["loss"]), {"loss_jepa_pred", "loss_traj"})
        self.assertEqual(tuple(traj.shape), (2, 2, 3))

    def test_multiview_uses_navsim_driving_condition_adapter_by_default(self):
        model = _tiny_mv_model()

        self.assertEqual(model.driving_condition_adapter_name, "navsim")
        self.assertEqual(model.predictor.trajectory_adapter_name, "navsim")
        self.assertFalse(model.predictor.traj_loss_grad_to_scene_flow)
        self.assertFalse(model.predictor.scene_loss_grad_to_traj_flow)

    def test_multiview_navsim_trajectory_adapter_preserves_checkpoint_keys(self):
        model = _tiny_mv_model()
        state_keys = set(model.state_dict())

        for key in (
            "predictor.noisy_traj_proj.weight",
            "predictor.history_traj_encoder.0.weight",
            "predictor.ego_status_encoder.0.weight",
            "predictor.traj_fusion.weight",
            "predictor.traj_out.3.weight",
            "predictor.traj_type_embed",
            "predictor.traj_pos",
        ):
            self.assertIn(key, state_keys)
        self.assertFalse(any(key.startswith("predictor.trajectory_adapter.") for key in state_keys))

    def test_multiview_bench2drive_driving_condition_adapter_is_reserved(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(driving_condition_adapter="bench2drive")

        with self.assertRaisesRegex(NotImplementedError, "bench2drive"):
            model(_mv_batch())

    def test_multiview_video_pretrain_can_select_reserved_bench2drive_adapter(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(
            enable_trajectory_head=False,
            driving_condition_adapter="bench2drive",
        )
        batch = _mv_batch()
        batch.pop("ego_status")
        batch.pop("history_trajectory")
        batch.pop("future_trajectory")

        out = model(batch)

        self.assertEqual(set(out["loss"]), {"loss_jepa_pred"})

    def test_multiview_video_pretrain_disables_trajectory_modules(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(enable_trajectory_head=False)

        traj_keys = [name for name, _param in model.named_parameters() if "traj" in name or "ego_status" in name]

        self.assertFalse(model.enable_trajectory_head)
        self.assertFalse(model.predictor.enable_trajectory_head)
        self.assertEqual(traj_keys, [])

    def test_multiview_video_pretrain_forward_uses_only_video_fields(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(enable_trajectory_head=False)
        batch = _mv_batch()
        batch.pop("ego_status")
        batch.pop("history_trajectory")
        batch.pop("future_trajectory")

        out = model(batch)

        self.assertEqual(set(out["loss"]), {"loss_jepa_pred"})
        self.assertTrue(bool(torch.isfinite(out["loss"]["loss_jepa_pred"])))
        self.assertIn("active_scene_xpred_mse", out["other_log"])
        self.assertNotIn("loss_traj", out["loss"])
        self.assertNotIn("active_traj_xpred_mse", out["other_log"])
        self.assertNotIn("active_traj_ade", out["other_log"])
        self.assertNotIn("active_traj_fde", out["other_log"])

    def test_multiview_video_pretrain_rejects_trajectory_loss(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(enable_trajectory_head=False)
        model.set_loss_weights({"loss_jepa_pred": 1.0, "loss_traj": 1.0})

        with self.assertRaisesRegex(ValueError, "loss_traj requires"):
            model(_mv_batch())

    def test_multiview_video_pretrain_rejects_trajectory_inference(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(enable_trajectory_head=False)

        with self.assertRaisesRegex(RuntimeError, "predict_trajectory requires"):
            model.predict_trajectory(_mv_batch())

    def test_multiview_scene_projector_keeps_vjepa_token_granularity(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        bsz = 2
        future_steps = model.mask_sampler.future_steps
        history_steps = model.mask_sampler.history_steps
        tokens_per_step = model.mask_sampler.tokens_per_step
        dim = model.target_dim
        future_tokens = torch.randn(
            bsz,
            model.num_cameras,
            future_steps * model.mask_sampler.tokens_per_step,
            dim,
        )
        history_tokens = torch.randn(
            bsz,
            model.num_cameras,
            history_steps * model.mask_sampler.tokens_per_step,
            dim,
        )

        future_scene = model.target_scene_projector(future_tokens)
        history_scene = model.scene_projector(history_tokens)

        self.assertEqual(
            tuple(future_scene.shape),
            (bsz, model.num_cameras, future_steps * tokens_per_step, model.scene_projector.scene_dim),
        )
        self.assertEqual(
            tuple(history_scene.shape),
            (bsz, model.num_cameras, history_steps * tokens_per_step, model.scene_projector.scene_dim),
        )

    def test_multiview_predictor_scene_positioner_uses_absolute_time_offset(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        tokens = torch.randn(
            1,
            model.num_cameras,
            model.mask_sampler.future_steps * model.mask_sampler.tokens_per_step,
            model.target_dim,
        )
        projected = model.target_scene_projector(tokens).reshape(1, -1, model.scene_projector.scene_dim)
        token_indices = torch.arange(
            model.mask_sampler.future_steps * model.mask_sampler.tokens_per_step,
            dtype=torch.long,
        ).view(1, -1)

        future_at_zero = model.predictor.scene_positioner(projected, token_indices, time_offset=0)
        future_at_absolute = model.predictor.scene_positioner(
            projected,
            token_indices,
            time_offset=model.mask_sampler.history_steps,
        )

        self.assertFalse(torch.allclose(future_at_zero, future_at_absolute))

    def test_multiview_zero_weight_losses_still_return_configured_keys(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        model.set_loss_weights(
            {
                "loss_jepa_pred": 0.0,
                "loss_traj": 0.0,
            }
        )
        batch = _mv_batch()
        batch.pop("ego_status")
        batch.pop("history_trajectory")
        batch.pop("future_trajectory")

        out = model(batch)

        self.assertEqual(set(out["loss"]), {"loss_jepa_pred", "loss_traj"})
        self.assertEqual(float(out["loss"]["loss_jepa_pred"].item()), 0.0)
        self.assertEqual(float(out["loss"]["loss_traj"].item()), 0.0)

    def test_multiview_all_masks_compute_trajectory_loss(self):
        torch.manual_seed(0)
        random_model = _tiny_mv_model(full_mask_prob=0.0)
        full_model = _tiny_mv_model(full_mask_prob=1.0)
        batch = _mv_batch()

        random_out = random_model(batch)
        full_out = full_model(batch)

        self.assertGreater(float(random_out["loss"]["loss_traj"].item()), 0.0)
        self.assertGreater(float(full_out["loss"]["loss_traj"].item()), 0.0)
        self.assertEqual(float(full_out["other_log"]["active_mask_full_frac"].item()), 1.0)
        self.assertEqual(
            float(random_out["other_log"]["active_context_tokens"].item()),
            float(random_model.mask_sampler.num_history_tokens),
        )
        self.assertEqual(
            float(full_out["other_log"]["active_context_tokens"].item()),
            float(full_model.mask_sampler.num_history_tokens),
        )

    def test_multiview_epdms_auxiliary_trajectory_losses_are_scalar(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        model.set_loss_weights(
            {
                "loss_jepa_pred": 0.0,
                "loss_traj": 1.0,
                "loss_traj_velocity": 0.1,
                "loss_traj_smooth": 0.05,
                "loss_traj_progress": 0.05,
            }
        )

        out = model(_mv_batch())

        for key in ("loss_traj_velocity", "loss_traj_smooth", "loss_traj_progress"):
            self.assertIn(key, out["loss"])
            self.assertTrue(bool(torch.isfinite(out["loss"][key])))
            self.assertEqual(out["loss"][key].dim(), 0)
            self.assertTrue(out["loss"][key].requires_grad)
            self.assertIn(f"active_{key}", out["other_log"])

    def test_multiview_inactive_warmup_losses_return_scalar_zeros(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        model.set_loss_weights(
            {
                "loss_jepa_pred": 0.5,
                "loss_traj": 1.0,
                "loss_traj_velocity": 0.0,
                "loss_traj_smooth": 0.0,
                "loss_traj_progress": 0.0,
            }
        )

        out = model(_mv_batch())

        for key in ("loss_traj_velocity", "loss_traj_smooth", "loss_traj_progress"):
            self.assertIn(key, out["loss"])
            self.assertEqual(float(out["loss"][key].item()), 0.0)
            self.assertEqual(out["loss"][key].dim(), 0)

    def test_multiview_random_future_tokens_condition_noise_stream_not_context(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(
            full_mask_prob=0.0,
            jepa_mask_configs=[
                {
                    "num_blocks": 1,
                    "spatial_scale": (0.25, 0.25),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (1.0, 1.0),
                }
            ],
        )
        model.eval()
        batch = _mv_batch()
        clip = model._make_full_clip(batch["history_images"], batch["future_images"])
        masks, _ = model._sample_training_masks(batch["history_images"].size(0), device=clip.device)

        with torch.no_grad():
            context_scene, future_condition, context_token_indices = model._encode_scene_inputs(
                clip,
                masks.masks_x[0],
                batch["history_images"].size(0),
            )

        self.assertGreater(masks.masks_x[0].size(1), model.mask_sampler.num_history_tokens)
        self.assertEqual(
            tuple(context_scene.shape),
            (
                batch["history_images"].size(0),
                model.num_cameras * model.mask_sampler.num_history_tokens,
                model.scene_projector.scene_dim,
            ),
        )
        self.assertEqual(tuple(context_token_indices.shape), (batch["history_images"].size(0), model.mask_sampler.num_history_tokens))
        self.assertEqual(
            tuple(future_condition.shape),
            (
                batch["history_images"].size(0),
                model.predictor.num_scene_tokens,
                model.scene_projector.scene_dim,
            ),
        )
        changed_batch = dict(batch)
        changed_batch["future_images"] = batch["future_images"] + 10.0
        changed_clip = model._make_full_clip(changed_batch["history_images"], changed_batch["future_images"])
        with torch.no_grad():
            changed_context, changed_future_condition, _changed_context_indices = model._encode_scene_inputs(
                changed_clip,
                masks.masks_x[0],
                changed_batch["history_images"].size(0),
            )
        self.assertEqual(tuple(changed_context.shape), tuple(context_scene.shape))
        self.assertFalse(torch.allclose(future_condition, changed_future_condition))

    def test_multiview_full_mask_prob_applies_to_whole_batch(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(full_mask_prob=0.5)

        with mock.patch(
            "models.multiview_causal_future_jepa.torch.rand",
            return_value=torch.tensor(0.0),
        ):
            full_masks, full_rows = model._sample_training_masks(3, device=torch.device("cpu"))

        self.assertEqual(len(full_masks.masks_x), 1)
        self.assertTrue(bool(full_rows.all().item()))
        self.assertTrue(bool(full_masks.full_mask.all().item()))

        with mock.patch(
            "models.multiview_causal_future_jepa.torch.rand",
            return_value=torch.tensor(1.0),
        ):
            random_masks, random_rows = model._sample_training_masks(3, device=torch.device("cpu"))

        self.assertEqual(len(random_masks.masks_x), len(model.mask_sampler.official_mask_configs))
        self.assertFalse(bool(random_rows.any().item()))
        self.assertFalse(bool(random_masks.full_mask.any().item()))

    def test_multiview_scene_token_collapse_metrics_compare_teacher(self):
        model = _tiny_mv_model(num_future_frames=4, dynamic_collapse_topk=1)
        tokens = model.num_cameras * model.mask_sampler.future_steps * model.mask_sampler.tokens_per_step
        pred_scene = torch.zeros(1, tokens, 2)
        target_scene = torch.zeros(1, tokens, 2)
        target_scene[:, 0] = torch.tensor([1.0, 0.0])
        target_scene[:, model.mask_sampler.tokens_per_step] = torch.tensor([0.0, 1.0])
        pred_scene[:, 0] = torch.tensor([1.0, 0.0])
        pred_scene[:, model.mask_sampler.tokens_per_step] = torch.tensor([1.0, 0.0])

        pred_cos, gap, teacher_cos, delta_ratio, teacher_delta_norm = (
            model._scene_token_collapse_metrics(
                [{"pred_scene": pred_scene, "target_scene": target_scene}],
                pred_scene,
            )
        )

        expected_teacher_cos = torch.tensor(0.0)
        expected_teacher_delta = torch.tensor(2.0**0.5)
        self.assertTrue(torch.allclose(pred_cos, torch.tensor(1.0), atol=1e-6))
        self.assertTrue(torch.allclose(gap, torch.tensor(1.0) - expected_teacher_cos, atol=1e-6))
        self.assertTrue(torch.allclose(teacher_cos, expected_teacher_cos, atol=1e-6))
        self.assertTrue(torch.allclose(delta_ratio, torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(teacher_delta_norm, expected_teacher_delta, atol=1e-6))

    def test_multiview_navsim_strictly_initializes_from_nuplan_checkpoint(self):
        torch.manual_seed(0)
        nuplan_stage = _tiny_mv_model(
            freeze_encoder_steps=20,
            target_scene_ema_warmup_start=0.5,
            target_scene_ema_warmup_steps=20,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(nuplan_stage.state_dict(), os.path.join(tmpdir, "model_state_dict.pt"))
            navsim_stage = _tiny_mv_model(
                init_from_checkpoint=tmpdir,
                init_checkpoint_allow_partial=False,
                freeze_encoder_steps=0,
                target_scene_ema_warmup_steps=0,
                _save_root=tmpdir,
            )
            report_path = os.path.join(tmpdir, "checkpoint_load_report.jsonl")
            with open(report_path, "r", encoding="utf-8") as f:
                reports = [
                    json.loads(line)
                    for line in f
                    if json.loads(line).get("log_prefix") == "model.init_from_checkpoint"
                ]

        self.assertEqual(navsim_stage.freeze_encoder_steps, 0)
        self.assertEqual(navsim_stage.target_scene_ema_warmup_steps, 0)
        self.assertTrue(reports)
        report = reports[-1]
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["unexpected_count"], 0)
        self.assertEqual(report["shape_mismatch_count"], 0)
        self.assertAlmostEqual(float(report["loaded_numel_ratio"]), 1.0, places=6)
        nuplan_state = nuplan_stage.state_dict()
        navsim_state = navsim_stage.state_dict()
        self.assertEqual(set(nuplan_state), set(navsim_state))
        for key, expected in nuplan_state.items():
            self.assertTrue(torch.allclose(expected, navsim_state[key]), key)

    def test_multiview_navsim_requires_warmup_checkpoint_when_requested(self):
        with self.assertRaises(ValueError):
            _tiny_mv_model(require_init_from_checkpoint=True)

    def test_multiview_init_from_checkpoint_list_loads_in_order(self):
        torch.manual_seed(1)
        first = _tiny_mv_model()
        first_state = {k: v.detach().clone() for k, v in first.state_dict().items()}
        override_key = next(k for k, v in first_state.items() if torch.is_floating_point(v))
        keep_key = next(k for k, v in first_state.items() if k != override_key and torch.is_floating_point(v))
        override_value = torch.full_like(first_state[override_key], 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_a = os.path.join(tmpdir, "stage_a")
            ckpt_b = os.path.join(tmpdir, "stage_b")
            os.makedirs(ckpt_a)
            os.makedirs(ckpt_b)
            torch.save(first_state, os.path.join(ckpt_a, "model_state_dict.pt"))
            torch.save({override_key: override_value}, os.path.join(ckpt_b, "model_state_dict.pt"))

            torch.manual_seed(3)
            loaded = _tiny_mv_model(
                init_from_checkpoint=[ckpt_a, ckpt_b],
                init_checkpoint_allow_partial=False,
                _save_root=tmpdir,
            )
            loaded_state = loaded.state_dict()
            with open(os.path.join(tmpdir, "checkpoint_load_report.jsonl"), encoding="utf-8") as f:
                prefixes = [json.loads(line)["log_prefix"] for line in f if line.strip()]

        self.assertTrue(torch.allclose(loaded_state[override_key], override_value))
        self.assertTrue(torch.allclose(loaded_state[keep_key], first_state[keep_key]))
        self.assertIn("model.init_from_checkpoint[0]", prefixes)
        self.assertIn("model.init_from_checkpoint[1]", prefixes)

    def test_multiview_freeze_encoder_steps_keeps_backbone_no_grad_before_start(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(freeze_encoder_steps=10)
        model.train()
        model.set_train_step(0)
        batch = _mv_batch(batch_size=1)
        clip = model._make_full_clip(batch["history_images"], batch["future_images"])
        masks = model.mask_sampler.full_future_mask(1, device=clip.device)

        context_scene = model._encode_scene_context(clip, masks.masks_x[0], 1)
        context_scene.float().sum().backward()

        self.assertIsNone(model.encoder.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.scene_projector.proj.weight.grad)

    def test_multiview_freeze_encoder_steps_treats_missing_step_as_zero(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(freeze_encoder_steps=10)
        model.train()
        model.set_train_step(None)
        batch = _mv_batch(batch_size=1)
        clip = model._make_full_clip(batch["history_images"], batch["future_images"])
        masks = model.mask_sampler.full_future_mask(1, device=clip.device)

        context_scene = model._encode_scene_context(clip, masks.masks_x[0], 1)
        context_scene.float().sum().backward()

        self.assertIsNone(model.encoder.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.scene_projector.proj.weight.grad)

    def test_multiview_freeze_encoder_steps_allows_backbone_grad_after_start(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(freeze_encoder_steps=10)
        model.train()
        model.set_train_step(10)
        batch = _mv_batch(batch_size=1)
        clip = model._make_full_clip(batch["history_images"], batch["future_images"])
        masks = model.mask_sampler.full_future_mask(1, device=clip.device)

        context_scene = model._encode_scene_context(clip, masks.masks_x[0], 1)
        context_scene.float().sum().backward()

        self.assertIsNotNone(model.encoder.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.scene_projector.proj.weight.grad)

    def test_multiview_predict_trajectory_without_future_images(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        with torch.no_grad():
            pred = model.predict_trajectory(_mv_batch())
        self.assertEqual(tuple(pred.shape), (2, 2, 3))

    def test_multiview_predict_trajectory_is_deterministic_by_default(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        features = _mv_batch()
        with torch.no_grad():
            pred1 = model.predict_trajectory(features)
            pred2 = model.predict_trajectory(features)
        self.assertTrue(torch.allclose(pred1, pred2))

    def test_multiview_trajectory_norm_round_trip_without_clamp(self):
        model = _tiny_mv_model(
            trajectory_norm_mean=[1.0, -2.0, 0.5],
            trajectory_norm_std=[2.0, 4.0, 0.25],
        )
        trajectory = torch.tensor(
            [
                [
                    [5.0, -10.0, 1.5],
                    [101.0, -202.0, 3.5],
                ]
            ]
        )

        normalized = model._normalize_trajectory(trajectory)
        restored = model._denormalize_trajectory(normalized)

        self.assertTrue(torch.allclose(restored, trajectory, atol=1e-6))
        self.assertGreater(float(normalized.abs().max().item()), 10.0)

    def test_multiview_target_scene_projector_tracks_student_with_ema(self):
        torch.manual_seed(0)
        model = _tiny_mv_model()
        with torch.no_grad():
            for param in model.scene_projector.parameters():
                param.add_(1.0)
        before = next(model.target_scene_projector.parameters()).detach().clone()
        model.after_optimizer_step(global_step=1)
        after = next(model.target_scene_projector.parameters()).detach()
        self.assertFalse(torch.allclose(before, after))

    def test_multiview_target_scene_ema_warmup_only_affects_scene_projector(self):
        torch.manual_seed(0)
        model = _tiny_mv_model(
            ema_start=0.9,
            ema_end=0.9,
            target_scene_ema_warmup_start=0.5,
            target_scene_ema_warmup_steps=10,
        )
        encoder_param = next(model.encoder.parameters())
        target_encoder_param = next(model.target_encoder.parameters())
        scene_param = next(model.scene_projector.parameters())
        target_scene_param = next(model.target_scene_projector.parameters())
        with torch.no_grad():
            encoder_param.add_(1.0)
            scene_param.add_(1.0)
            encoder_before = target_encoder_param.detach().clone()
            scene_before = target_scene_param.detach().clone()
            encoder_student = encoder_param.detach().clone()
            scene_student = scene_param.detach().clone()

        model.after_optimizer_step(global_step=0)

        self.assertTrue(
            torch.allclose(
                target_encoder_param.detach(),
                encoder_before * 0.9 + encoder_student * 0.1,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                target_scene_param.detach(),
                scene_before * 0.5 + scene_student * 0.5,
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()

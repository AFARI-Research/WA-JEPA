# Config Guide

This directory contains the training configuration templates. Important
behavior is defined explicitly in YAML rather than left to implicit Python
defaults.

## Templates

Basic configs live under `configs/basic/`:

- `configs/basic/base_training.yaml`: default NuPlan-style base template.
- `configs/basic/nuplan_basic.yaml`: single NuPlan dataset training.
- `configs/basic/navsim_basic.yaml`: single NAVSIM dataset training using the local dataset wrapper.
- `configs/basic/navsim_official_basic.yaml`: single NAVSIM dataset training using official `SceneLoader` / `SceneFilter`.
- `configs/basic/mixed_data_basic.yaml`: mixed dataset training with weighted sampling across sources.

Runnable experiment templates live directly under `configs/`:

- `configs/stage3_navsim_placeholder.yaml`: legacy stage-3 NAVSIM parameters
  paired with a minimal placeholder model target. Use this template when
  developing a replacement model while retaining the previous stage-3 training
  settings.
- `configs/causal_future_jepa_mv_navsim.yaml`: active multi-view V-JEPA 2.1
  causal driving JEPA NAVSIM finetune experiment. It requires NAVSIM paths, an official
  V-JEPA 2.1 checkpoint, and the optional DeepSpeed dependency for the provided
  ViT-L/16 training settings.
- `configs/causal_future_jepa_mv_nuplan.yaml`: optional NuPlan warmup stage for
  the same multi-view model. It trains scene flow and trajectory flow jointly,
  while keeping the original V-JEPA student encoder in no-grad mode for the first
  `freeze_encoder_steps` optimizer steps.

## Top-Level Fields

| Field | Meaning | Common Use |
| --- | --- | --- |
| `seed` | Random seed passed to Accelerate / PyTorch setup. | Keep fixed for comparable runs. |
| `tf32` | Enables CUDA TF32 matmul / cuDNN when available. | Usually `true` for faster training on Ampere+ GPUs. |
| `name` | Broad experiment group. | Used in save path and tracker tags. |
| `experiment_name` | Specific run name. | Used in save path and tracker run name. |
| `save_root` | Root directory for run outputs. | Checkpoints and copied configs go under this path. |
| `checkpoint` | Resume checkpoint path. | Directory containing `model_state_dict.pt` and `state.pt`, or a `.pt` file. |
| `stage2_checkpoint` | Previous stage-2 model checkpoint path kept for stage-3-style experiments. | New model code may use this to initialize from an earlier phase. The base trainer ignores it. |
| `vlm_feature_cache_dir` | Previous VLM feature cache root kept for stage-3-style experiments. | New data/model code may use this if cache loading is reintroduced. The base trainer ignores it. |
| `experiment_tracker` | `none` or `wandb`. | Selects logging backend. Defaults to `none`, so training reports to stdout only. |

Run outputs are saved as:

```text
<save_root>/<user>/<name>/<experiment_name>/<timestamp>/
```

If no user environment variable is found, the `<user>` segment is omitted.

## Logging

### Weights & Biases

Prefer the nested form:

```yaml
wandb:
  project: world-model
  run_name:
  entity:
  mode:
  dir:
```

| Field | Meaning |
| --- | --- |
| `wandb.project` | W&B project. |
| `wandb.run_name` / `wandb.name` | Explicit run name. Defaults to generated run name if empty. |
| `wandb.entity` | W&B team or user. |
| `wandb.mode` | W&B mode, e.g. `offline`, `disabled`, or empty for default. |
| `wandb.dir` | Local W&B directory. Defaults to `<save_root>/wandb`. |

The flat `wandb_project`, `wandb_run_name`, `wandb_entity`, and `wandb_mode`
fields are kept for compatibility.

## Model

```yaml
model:
  target: models.my_model:MyModel
  name:
  dtype: bfloat16
```

| Field | Meaning |
| --- | --- |
| `target` | Import path for the model class. Recommended for new implementations. |
| `name` | Optional registry key if using `MODEL_REGISTRY.register(...)`. |
| `dtype` | Optional model dtype hint for model code. The base trainer does not force-cast from this field. |

`target` classes are constructed as:

```python
MyModel(model_cfg, full_cfg=config)
```

The trainer expects:

```python
{
    "loss": {"main": scalar_tensor},
    "other_log": {"metric": scalar_tensor},  # optional
}
```

`loss` entries must be scalar tensors. `other_log` scalar tensors are logged and
averaged during validation.

### Placeholder Model

`configs/stage3_navsim_placeholder.yaml` uses:

```yaml
model:
  target: models.placeholder:PlaceholderWorldModel
```

This target is intended only for smoke tests. It has one trainable scalar,
returns zero losses matching `trainer.losses`, and produces an all-zero
trajectory. Replace `model.target` with the production model class before
training.

### Causal Future-Masked JEPA

`configs/causal_future_jepa_mv_navsim.yaml` uses:

```yaml
model:
  target: models.multiview_causal_future_jepa:MultiViewCausalFutureMaskedJEPA
  vjepa2_ckpt: ${oc.env:VJEPA2_CKPT,/path/to/vjepa2_1_vitl_dist_vitG_384.pt}
  require_pretrained: true
```

Key fields:

| Field | Meaning | Notes |
| --- | --- | --- |
| `model.vjepa2_ckpt` | Official V-JEPA 2.1 ViT-L/16 checkpoint path. | Required when `require_pretrained` is true. Set `VJEPA2_CKPT`, or edit the config. The default is a placeholder rather than a working path, so a missing checkpoint fails at load instead of training from scratch: `require_pretrained: true` and `min_checkpoint_load_ratio: 1.0` together reject a partial or absent load. |
| `model.encoder_checkpoint_key`, `model.target_encoder_checkpoint_key` | Optional checkpoint subkey overrides for student and EMA target encoders. | Leave unset for V-JEPA 2.1 ViT-L/16 so the loader can try `target_encoder`, `ema_encoder`, then `encoder` and match both official naming variants. |
| `model.copy_target_encoder_from_student` | Initialize EMA target encoder from the loaded student encoder. | Default is `true` unless `target_encoder_checkpoint_key` is explicitly set. This avoids inheriting a separate pretraining EMA lag when the driving JEPA target differs from official V-JEPA pretraining. |
| `model.variant` | Vendored V-JEPA encoder variant, typically `vjepa2_1_vit_large_384`. | Must match the checkpoint size. |
| `model.input_hw` | Runtime image resolution, currently `[256, 512]`. | Order is height, width. |
| `model.pretrained_input_hw` | RoPE interpolation base image size. | The template uses `[256, 256]` to match the official V-JEPA 2.1 RoPE base grid even when loading 384 release checkpoints. |
| `model.image_normalization` | Input conversion before V-JEPA. | `minus_one_to_imagenet` converts dataset `[-1, 1]` tensors to ImageNet-normalized inputs. |
| `model.num_history_frames`, `model.num_future_frames` | Driving clip split. | Current default is 4 history + 8 future frames. |
| `model.use_activation_checkpointing` | Recompute ViT / predictor blocks during backward to reduce memory. | Current speed-oriented template sets it to `false`; turn it back on if ViT-L/16 runs out of memory. |
| `model.ema_start`, `model.ema_end`, `model.ema_total_steps` | Target encoder EMA momentum schedule. | The code uses a linear schedule from `ema_start` to `ema_end` over `ema_total_steps`, matching the V-JEPA training form. Current templates set start=end=`0.99925`, so the effective momentum stays constant while the schedule path is explicit. |
| `model.target_scene_ema_warmup_start`, `model.target_scene_ema_warmup_steps` | Extra linear EMA warmup for newly initialized non-backbone target modules. | In the current multi-view model this applies only to `target_scene_projector`; the V-JEPA backbone target encoder still uses `ema_start/end/total_steps`. NuPlan warmup uses this for newly initialized scene modules; NAVSIM finetune disables it because it should start from the NuPlan checkpoint. |

Multi-view JEPA additions:

| Key | Meaning | Notes |
| --- | --- | --- |
| `model.target: models.multiview_causal_future_jepa:MultiViewCausalFutureMaskedJEPA` | Enables the multi-camera JEPA world model. | See `configs/causal_future_jepa_mv_navsim.yaml`. |
| `model.camera_names` / `data.camera_names` | Ordered camera list. | Current template uses `[cam_l0, cam_f0, cam_r0, cam_b0]`. Set a single camera, for example `[cam_f0]`, to run the same multi-view architecture as a one-camera model. |
| `model.enable_trajectory_head` | Registers and runs the trajectory branch. | Set `false` for video-only stage-1 pretraining; set `true` for NuPlan/NAVSIM driving stages. When disabled, the model only returns `loss_jepa_pred` and does not require driving fields. |
| `model.driving_condition_adapter` | Dataset-specific adapter for driving condition fields and trajectory-flow semantics. | Current implemented value is `navsim`, used by NuPlan and NAVSIM configs. Bench2Drive config is intentionally not added yet. |
| `model.mv_scene_dim` | Hidden size of the lightweight multi-view scene projector. | The projector is newly initialized and only contains `LayerNorm(D_vjepa)` plus `Linear(D_vjepa -> D_scene)`; V-JEPA encoder weights are still loaded from the official checkpoint. Camera/time/spatial embeddings and output norm live inside the predictor, so EMA only tracks the compact projector. Time embeddings use absolute tubelet indices in the full clip, so future teacher tokens start after the history tubelets. |
| `model.flow_hidden_dim`, `model.flow_num_layers`, `model.flow_num_heads` | Scene/trajectory joint-flow x-pred predictor size. | Current direct-token template uses `512` hidden, `12` layers, and `8` heads. Each block uses MMDiT-style modality-specific QKV/out projections, one joint attention over context / scene / trajectory tokens, separate FFNs, flow type embeddings, AdaLN-Zero residual gates, and QK RMSNorm. |
| `model.traj_loss_grad_to_scene_flow`, `model.scene_loss_grad_to_traj_flow` | Directional cross-stream gradient switches inside predictor attention. | Attention is computed per query stream, so trajectory tokens can still read scene tokens and scene tokens can still read trajectory tokens, but the opposite stream's K/V tensors are detached unless the corresponding switch is enabled. Current driving configs use `true/false`: trajectory loss can update scene flow, while scene loss cannot update trajectory flow. |
| `model.trajectory_horizon` | Number of future trajectory points predicted by the trajectory flow. | Current template uses `8`, matching `data.num_future_frames`. |
| `model.full_mask_prob` | Probability of overriding a whole training batch from official-like random future mask to full future mask. | Random and full batches both compute scene flow and trajectory flow losses. Full-mask batches match the inference path most closely and avoid mixed visible-token layouts inside one batch. |
| `model.flow_time_sampling`, `model.flow_logit_mean`, `model.flow_logit_std` | Training time-step distribution for multi-view flow. | Current template uses `logit_normal` with `sigmoid(N(-0.2, 1.6))`, giving broader coverage than the earlier `sigmoid(N(0, 1))` setting and a mild bias toward noisy early-inference states. |
| `model.flow_inference_seed` | Optional deterministic seed for flow inference noise. | Default `0` keeps NAVSIM evaluation reproducible while still starting from Gaussian noise. Set to null only if stochastic test-time sampling is intended. |
| `model.dynamic_collapse_topk` | Number of teacher-dynamic camera-spatial tokens used by scene-token collapse logs. | The metric first selects the camera-spatial tokens with the largest teacher feature change over future time, then compares prediction vs teacher over the future time axis. This matches the single-view `jepa_wm` dynamic-token collapse semantics and avoids full all-pairs logging over every scene token. |
| `model.trajectory_norm_mean`, `model.trajectory_norm_std` | Linear normalization for `[x, y, heading]` trajectory flow. | Defaults mirror the `jq/vla` HMEF `norm_odo` affine range: mean `[31.8, 1.32, 0.095]`, std `[33.37, 21.0, 1.765]`. History/future trajectories are normalized before the flow predictor and denormalized for `predict_trajectory()` output; no clamp is applied. |
| `model.freeze_encoder_steps` | Number of initial optimizer steps that keep the student V-JEPA encoder in no-grad feature-extraction mode. | NuPlan warmup currently uses `20000`; NAVSIM finetune uses `0` because scene projector / predictor should already be trained by the NuPlan stage. If the total NuPlan steps are below this value, the V-JEPA encoder remains frozen for the whole NuPlan stage. Parameters remain in the optimizer, so DeepSpeed/ZeRO does not need to rebuild optimizer state when the encoder unfreezes. |
| `model.init_from_checkpoint`, `model.require_init_from_checkpoint` | Model-only initialization from one or more previous stage checkpoints. | Set `init_from_checkpoint` to the NuPlan warmup `checkpoints/last` directory for NAVSIM finetune, or to a YAML list to load multiple checkpoints in order. Single-checkpoint loads honor `init_checkpoint_allow_partial`; multi-checkpoint loads are partial per checkpoint and later checkpoints override earlier matching tensors. Do not use top-level `checkpoint` unless you want strict trainer resume. |

For `NAVSIM_OFFICIAL`, `max(data.num_history_image_frames,
data.num_history_trajectory_steps)` must match the selected SceneFilter history
length.

### Stage-3 Model Parameter Surface

The stage-3 template retains the previous model-facing parameters so replacement
implementations can use them without consulting repository history.

| Field | Meaning | Notes |
| --- | --- | --- |
| `model.freeze_vlm` | Whether the legacy VLM branch should be frozen. | The placeholder ignores this field; replacement models may define its behavior. |
| `model.use_wan`, `model.use_jepa`, `model.use_vggt`, `model.use_traj` | Switches for optional expert branches from the previous design. | Retained for compatibility and not implemented by the current base code. |
| `model.use_vlm_autoreg_text_trajectory` | Previous VLM autoregressive trajectory switch. | Placeholder ignores it. |
| `model.use_cached_features` | Whether to use cached feature tensors. | Base datasets do not load the legacy cache format. |
| `model.load_history_images_from_cache` | Whether history image paths/images are expected from cache. | Placeholder ignores it. |
| `model.external_vggt_context.*` | Previous external VGGT context encoder settings. | Includes `enabled`, `vggt_model_path`, `pool_size`, and `projector`. |
| `model.external_jepa_context.*` | Previous external JEPA context encoder settings. | Includes checkpoint, variant, dtype, input shape, tubelet/patch settings, and projector. |
| `model.vlm_conditioning.*` | Token groups that should condition the action head. | Kept for the redesigned model's expert routing. |
| `model.vlm.*` | Previous VLM path and action-token layout. | Token counts/steps mirror the old stage-2/stage-3 setup. |
| `model.action_head.*` | Previous HMEF/action-head hyperparameters. | Retained as configuration fields only; the legacy action-head implementation is not included. |
| `model.action_head.fusion_mode` | Previous expert fusion granularity. | Typical values: `global`, `per_step`. |
| `model.action_head.traj_token_mask_*` | Previous traj-token masking augmentation settings. | New model code must implement the behavior if needed. |
| `model.action_head.goal_loss_*` | Previous endpoint/goal auxiliary loss switches. | Loss weight remains under `trainer.losses.goal_loss.weight`. |

## Trainer

| Field | Meaning | Notes |
| --- | --- | --- |
| `batch_size_per_gpu` | Per-process batch size. | Applied to train and val loaders. |
| `max_epochs` | Number of epochs to train. | Used when `> 0`. |
| `max_steps` | Number of optimizer steps to train. | Used only when `max_epochs <= 0`. |
| `eval_only` | Skip training and run validation once. | Requires `checkpoint` for meaningful evaluation. |
| `log_interval` | Training metric logging interval in optimizer steps. | Main process only. |
| `eval_interval` | Validation interval. | `>1` means fixed steps; `0<value<1` means fraction of total steps; `<=0` disables step validation. |
| `eval_max_batches` | Maximum validation batches per validation run. | Keeps validation bounded. |
| `gradient_accumulation_steps` | Accelerate gradient accumulation. | Also used to compute optimizer steps per epoch. |
| `gradient_clipping` | Max grad norm. | Applied before optimizer step. |
| `cached_pin_memory` | Previous cached-feature DataLoader pin-memory switch. | Kept in stage-3 template for compatibility; current generic loader ignores it. |
| `cached_num_workers` | Previous cached-feature DataLoader worker count. | Kept in stage-3 template for compatibility; current generic loader ignores it. |
| `cached_persistent_workers` | Previous cached-feature DataLoader persistent-worker switch. | Kept in stage-3 template for compatibility; current generic loader ignores it. |
| `cached_prefetch_factor` | Previous cached-feature DataLoader prefetch factor. | Kept in stage-3 template for compatibility; current generic loader ignores it. |
| `weight_decay` | AdamW weight decay. | Default template keeps it `0.0`. |
| `learning_rate.base` | Default learning rate. | Used for all trainable params not matched by a more specific entry. |
| `learning_rate.<module_or_glob>` | Per-module or parameter-pattern learning rate. | Example: `backbone: 0.00001` or `head.*: 0.0001`. |
| `lr_warmup_ratio` | Warmup fraction of total training steps. | Cosine schedule after warmup. |
| `freeze_modules` | Module path or list/patterns to freeze. | Example: `backbone` or `[backbone, encoder.*]`. |
| `losses` | Optional loss-weight map. | If empty, all scalar losses are summed. If non-empty, each configured key must be returned by the model as a scalar loss. |
| `best_metric` | Validation metric used for top-k checkpoint selection. | Usually `val_total` or another returned validation key. Missing metrics raise unless `allow_best_metric_fallback` is true. |
| `allow_best_metric_fallback` | Allow fallback from missing `best_metric` to generic loss keys. | Defaults to false to catch metric-name typos. |
| `best_mode` | `min` or `max`. | Controls top-k checkpoint ranking. |
| `save_checkpoint_every_epoch` | Save checkpoint after each completed epoch. | Also saves `last` after validation / final step. |
| `save_checkpoint_steps` | Explicit optimizer-step checkpoints to keep. | List of positive steps, e.g. `[5000, 10000]`. Matching steps save `checkpoints/step_<step>` and are not managed or deleted by the top-k best checkpoint tracker. |
| `mixed_precision` | Accelerate mixed precision mode. | Common values: `"no"`, `"fp16"`, `"bf16"`. |
| `deepspeed_config` | Path to DeepSpeed JSON config, or empty/null to disable plugin. | DeepSpeed is optional; install it separately before enabling this field. Environment `DEEPSPEED_CONFIG` is used only when this field is omitted. |
| `dist_timeout_sec` | Distributed process group timeout. | Useful for slow dataset startup. |

DataLoader settings are read from `data:` for both single-dataset and mixed-dataset
training:

| Field | Meaning | Notes |
| --- | --- | --- |
| `data.dataloader_num_workers` | Train DataLoader worker count. | Defaults to `6` when omitted. |
| `data.val_dataloader_num_workers` | Validation DataLoader worker count. | Defaults to train-style worker count when validation is non-empty. |
| `data.dataloader_prefetch_factor` | PyTorch DataLoader prefetch factor. | Used only when worker count is greater than zero. |
| `data.dataloader_pin_memory` | DataLoader pin-memory switch. | Usually `true` for CUDA training. |
| `data.dataloader_persistent_workers` | Keep worker processes alive across epochs. | Useful for expensive dataset startup; default remains `false`. |

Example weighted losses:

```yaml
trainer:
  losses:
    loss_jepa_pred:
      weight: 1.0
    loss_traj:
      weight: 1.0
```

Inline NAVSIM eval expects the model to implement:

```python
predict_trajectory(features) -> torch.Tensor  # [B, T, 3]
```

Portable checkpoints save model, scheduler, config, step, and epoch. Non-DeepSpeed
runs also save optimizer state. DeepSpeed optimizer state is engine-managed and
is not stored in the portable checkpoint; treat those checkpoints as weight
recovery/export unless you add a DeepSpeed-native engine checkpoint path.
Checkpoint resume is strict by default: model keys must match exactly, and
DeepSpeed portable checkpoints are not accepted for optimizer-state training
resume.

For architecture-changing initialization, use a model-specific partial init
field instead of top-level `checkpoint`. For example, NAVSIM finetune can
start from a NuPlan warmup checkpoint with:

```yaml
model:
  init_from_checkpoint: /path/to/nuplan_warmup/checkpoints/last
  init_checkpoint_allow_partial: true
```

This loads only shape-compatible model weights and does not restore optimizer,
scheduler, epoch, or global step. Missing, unexpected, and shape-mismatched
keys are written under the experiment `save_root` in `checkpoint_load_report.*`
and `checkpoint_load_keys/`.

`init_from_checkpoint` can also be a list:

```yaml
model:
  init_from_checkpoint:
    - /path/to/base_stage/checkpoints/last
    - /path/to/head_or_adapter/checkpoints/last
```

List entries are loaded sequentially. This is useful when different checkpoints
own different parts of the model; if the same tensor appears more than once,
the later checkpoint wins.

## Single Dataset Fields

Single-dataset configs use:

```yaml
data:
  dataset: NUPLAN  # or NAVSIM / NAVSIM_OFFICIAL
```

### Shared Data Fields

| Field | Meaning | Notes |
| --- | --- | --- |
| `frame_height`, `frame_width` | Resize target used by dataset loaders. | |
| `num_history_frames` | Legacy fallback for both image and trajectory history counts when specific fields are absent. | |
| `num_history_image_frames` | Number of history image frames. | |
| `num_history_image_paths` | Number of history image paths to preserve when a model/cache pipeline needs paths rather than decoded tensors. | |
| `num_history_trajectory_steps` | Number of past ego trajectory steps. | |
| `num_future_frames` | Number of future target frames / trajectory steps. | |
| `trajectory_frame_rate_hz` | Future trajectory sampling rate. | |

### NuPlan

| Field | Meaning | Notes |
| --- | --- | --- |
| `nuplan_data_root` | Dataset root containing NuPlan DBs / maps expected by `NuplanDevkitDataset`. | |
| `nuplan_cameras` | NuPlan camera list, e.g. `[CAM_L0, CAM_F0, CAM_R0, CAM_B0]`. | When multiple cameras are configured, the dataset returns images as `[T,V,C,H,W]`. |
| `nuplan_primary_camera` | Camera whose timestamps define the history/future sample times and ego trajectory poses. | Current template uses `CAM_F0`. |
| `camera_names` | Lowercase model/data camera names used by multi-view JEPA, e.g. `[cam_l0, cam_f0, cam_r0, cam_b0]`. | Must match `model.camera_names` after lowercasing. |
| `nuplan_split` | NuPlan split name, e.g. `mini`. | |
| `nuplan_anchor_stride` | Sample stride over available frames. | |
| `nuplan_camera_time_tolerance_us` | Nearest-time tolerance for aligning non-primary cameras to the primary camera timestamp. | Default template uses `50000` us, half of the nominal 10Hz camera interval. |
| `nuplan_split_ratio` | Train/val split ratio inside `NuplanDevkitDataset`. | |
| `nuplan_index_build_workers` | Workers for building dataset index. | |
| `private_data_root` | Reserved/legacy field; currently not used by the reset training entrypoint. | |

### NAVSIM

| Field | Meaning | Notes |
| --- | --- | --- |
| `navsim_log_path` | Train logs path. | |
| `sensor_blobs_path` | Train sensor blobs path. | Falls back to `navsim_log_path` if empty. |
| `navsim_val_log_path` | Val logs path. | Falls back to `navsim_log_path` if empty. |
| `navsim_val_sensor_blobs_path` | Val sensor blobs path. | Falls back to train sensor/log path if empty. |
| `train_scene_filter_yaml` | Optional train scene filter. | |
| `val_scene_filter_yaml` | Optional val scene filter. | Falls back to train filter when needed. |
| `navsim_window_stride` | Train window stride. | |
| `navsim_val_window_stride` | Val window stride. | Falls back to `navsim_window_stride`. |
| `camera_name` | Camera key, usually `cam_f0`. | |

### NAVSIM Official

Uses the same path fields as NAVSIM. `train_scene_filter_yaml` is required
because official loading builds samples from a `SceneFilter`.

| Field | Meaning | Notes |
| --- | --- | --- |
| `navsim_val_holdout_ratio` | Fraction of the train filter's logs held out for validation. | Split by a `blake2b` hash of the log name, so the train and val sides are exact complements and stay identical across ranks, runs, and filter-file regeneration. Requires `log_names` in the train filter YAML, and requires `val_scene_filter_yaml` to be empty or equal to `train_scene_filter_yaml` — a val filter of its own already defines a split, and splitting it again would silently evaluate a fraction of it. Leave unset to validate on whatever `val_scene_filter_yaml` names. |

`navtrain` and `navtest` share no logs, so pointing validation at `navtest`
makes in-training validation a test-split run. The NAVSIM template therefore
holds out 5% of `navtrain` and leaves `navtest` for the final scoring scripts.

## Navigation Command Fields

These are passed to dataset loaders through `nav_command_kwargs_from_mapping`.

| Field | Meaning |
| --- | --- |
| `nav_command_infer_mode` | Mode for inferring navigation command from trajectory/status. Supported: `endpoint`, `arc_length`. |
| `nav_command_forward_m` | Forward distance threshold in meters. |
| `nav_command_lateral_m` | Lateral distance threshold in meters. |
| `nav_command_heading_rad` | Heading threshold in radians. |
| `nav_command_combine` | Strategy for combining navigation command cues. Supported: `or`, `and`, `lat_only`, `heading_only`. |

See `datasets/nav_command_infer.py` for the exact supported modes and combine
behavior.

## Mixed Dataset

Mixed training uses `data.sources` instead of `data.dataset`.

```yaml
data:
  sampling:
    mode: proportional
    samples_per_epoch:
    val_samples_per_epoch:
    weights:
      nuplan_mini: 1.0
      navsim_train: 1.0

  sources:
    nuplan_mini:
      type: NUPLAN
      ...
```

### Shared Values

The top-level fields under `data`:

```yaml
frame_height
frame_width
num_history_image_frames
num_history_trajectory_steps
num_future_frames
trajectory_frame_rate_hz
```

are shared by all sources. In the current loader, shared values take precedence
over per-source values for these keys. Remove a key from the shared section if
you want a source-specific value to take effect.

### Sampling

| Field | Meaning |
| --- | --- |
| `sampling.mode` | `proportional`, `uniform`, or `weights`. |
| `sampling.samples_per_epoch` | Virtual train samples per epoch. Empty means total source length. |
| `sampling.val_samples_per_epoch` | Virtual val samples per validation epoch. Empty means total source val length. |
| `sampling.weights` | Required only when `mode: weights`; keys must match source names. |

Modes:

- `proportional`: source probability follows dataset length.
- `uniform`: each source has equal probability.
- `weights`: use explicit positive weights.

### Source Fields

Each source has:

```yaml
type: NUPLAN  # or NAVSIM / NAVSIM_OFFICIAL
```

Then it uses that dataset type's fields. For NAVSIM sources, `log_names` and
`scene_tokens` can optionally restrict the loaded subset.

## Launch

Basic configurations provide data and trainer defaults but require a concrete
model target before launch. Runnable templates directly under `configs/` already
set `model.target`, but still require dataset paths and, for JEPA, a checkpoint.

```bash
python training/train_world_model.py --config configs/stage3_navsim_placeholder.yaml
python training/train_world_model.py --config configs/causal_future_jepa_mv_navsim.yaml
python training/train_world_model.py --config configs/basic/nuplan_basic.yaml model.target=models.placeholder:PlaceholderWorldModel
```

The shell launcher takes the same config path:

```bash
bash scripts/training/train.sh --config configs/stage3_navsim_placeholder.yaml
```

CLI dotlist overrides are supported:

```bash
python training/train_world_model.py \
  --config configs/basic/nuplan_basic.yaml \
  trainer.batch_size_per_gpu=4 \
  experiment_name=test_run
```

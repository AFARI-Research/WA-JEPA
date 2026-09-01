"""Reusable training loop.

Model contract:
    ``model(batch)`` returns ``{"loss": {"name": scalar_tensor}, "other_log": {...}}``.

Validation defaults to running the same forward pass on the validation loader and
averaging scalar losses. New algorithms can either fit that contract or subclass
``WorldModelTrainer``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    RenderableColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.optim.lr_scheduler import LambdaLR

from training.checkpoint import BestCheckpointTracker, load_checkpoint, save_checkpoint
from utils import wandb_utils
from utils.utils import (
    batch_to_device,
    build_param_lr_groups,
    cfg_get,
    config_to_serializable,
    dict_scalars_to_tensors,
    freeze_modules,
    print_trainable_summary,
    resolve_experiment_tracker,
    set_sampler_epoch,
    unwrap_model_for_non_deepspeed,
    unpack_step_out,
)


def _is_finite_everywhere(loss: torch.Tensor) -> bool:
    ok = torch.isfinite(loss.detach())
    if dist.is_available() and dist.is_initialized():
        ok_i = ok.to(device=loss.device, dtype=torch.int32)
        dist.all_reduce(ok_i, op=dist.ReduceOp.MIN)
        return bool(ok_i.item())
    return bool(ok.item())


def _broadcast_bool(value: bool, device: torch.device) -> bool:
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor(int(value), device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        return bool(flag.item())
    return bool(value)


def _hookable_model(model: nn.Module) -> nn.Module:
    return unwrap_model_for_non_deepspeed(model)


def _prepare_model_batch(model: nn.Module, batch: Any) -> Any:
    prepare_batch = getattr(_hookable_model(model), "prepare_batch", None)
    if callable(prepare_batch):
        return prepare_batch(batch)
    return batch


def _set_model_loss_weights(model: nn.Module, weights: dict[str, float]) -> None:
    setter = getattr(_hookable_model(model), "set_loss_weights", None)
    if callable(setter):
        setter(weights)


def _set_model_train_step(model: nn.Module, step: int | None) -> None:
    setter = getattr(_hookable_model(model), "set_train_step", None)
    if callable(setter):
        setter(step)


def _cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    warmup = max(0, int(num_warmup_steps))
    total = max(1, int(num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup > 0 and current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class WorldModelTrainer:
    """Extensible trainer for supervised model experiments."""

    def __init__(
        self,
        cfg,
        model: nn.Module,
        train_loader,
        val_loader,
        accelerator: Accelerator,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self._val_loader = val_loader
        self.accelerator = accelerator

        trainer_cfg = cfg_get(cfg, "trainer", {})
        self._eval_only = bool(cfg_get(trainer_cfg, "eval_only", False))
        self._log_interval = int(cfg_get(trainer_cfg, "log_interval", 50))
        self._eval_interval_raw = cfg_get(trainer_cfg, "eval_interval", 1000)
        self._eval_interval = 0
        self._max_epochs = int(cfg_get(trainer_cfg, "max_epochs", 0))
        self._max_steps = int(cfg_get(trainer_cfg, "max_steps", 0))
        self._gradient_clipping = float(cfg_get(trainer_cfg, "gradient_clipping", 1.0))
        self._lr_warmup_ratio = float(cfg_get(trainer_cfg, "lr_warmup_ratio", 0.05))
        self._eval_max_batches = int(cfg_get(trainer_cfg, "eval_max_batches", 50))
        self._best_metric = str(cfg_get(trainer_cfg, "best_metric", "val_total"))
        self._best_mode = str(cfg_get(trainer_cfg, "best_mode", "min")).strip().lower()
        self._allow_best_metric_fallback = bool(
            cfg_get(trainer_cfg, "allow_best_metric_fallback", False)
        )
        if self._best_mode not in ("min", "max"):
            self._best_mode = "min"
        self._save_checkpoint_every_epoch = bool(
            cfg_get(trainer_cfg, "save_checkpoint_every_epoch", True)
        )
        self._save_checkpoint_steps = set(
            self._parse_checkpoint_steps(cfg_get(trainer_cfg, "save_checkpoint_steps", []))
        )
        self._saved_checkpoint_steps: set[int] = set()
        self._loss_cfg = cfg_get(trainer_cfg, "losses", {})
        self._save_root = str(cfg_get(cfg, "save_root", "results"))
        self._experiment_tracker = resolve_experiment_tracker(cfg)

        self.global_step = 0
        self._resume_epoch_count = 0
        self._best_tracker: BestCheckpointTracker | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: LambdaLR | None = None
        self.progress = self._make_rich_progress(
            disable=not self.accelerator.is_main_process
        )
        self._train_progress_task: Any | None = None

    @staticmethod
    def _resolve_eval_interval_steps(raw: Any, total_steps: int) -> int:
        if total_steps <= 0:
            return 0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 1000.0
        if value <= 0:
            return 0
        if value > 1.0:
            return max(1, int(value))
        if value < 1.0:
            return max(1, int(round(float(total_steps) * value)))
        return 1

    @staticmethod
    def _parse_checkpoint_steps(raw: Any) -> list[int]:
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            values = [item.strip() for item in raw.replace(",", " ").split()]
        elif isinstance(raw, (int, float)):
            values = [raw]
        else:
            try:
                values = list(raw)
            except TypeError:
                values = [raw]

        steps: set[int] = set()
        for value in values:
            if value is None or value == "":
                continue
            step = int(value)
            if step > 0:
                steps.add(step)
        return sorted(steps)

    def setup(self) -> None:
        cfg = self.cfg
        trainer_cfg = cfg_get(cfg, "trainer", {})

        frozen = freeze_modules(self.model, cfg_get(trainer_cfg, "freeze_modules", None))
        if frozen:
            self.accelerator.print(f"[Trainer] Frozen modules: {frozen}")

        print_trainable_summary(self.model, log_fn=self.accelerator.print)

        lr_cfg = cfg_get(trainer_cfg, "learning_rate", {"base": 1e-4})
        if not isinstance(lr_cfg, dict):
            lr_cfg = {"base": float(lr_cfg)}
        param_groups = build_param_lr_groups(self.model, lr_cfg)
        if not param_groups:
            raise ValueError("Model has no trainable parameters.")

        for i, group in enumerate(param_groups):
            n_params = sum(p.numel() for p in group["params"])
            self.accelerator.print(
                f"  Param group [{i}] name={group.get('name', '?')} "
                f"lr={group['lr']:.3g} numel={n_params / 1e6:.2f}M"
            )

        first_param = next(self.model.parameters(), None)
        use_cuda = first_param is not None and first_param.device.type == "cuda"
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=float(cfg_get(trainer_cfg, "weight_decay", 0.0)),
            fused=use_cuda,
        )

        if self._val_loader is None:
            raise ValueError("Validation loader must be set before trainer.setup().")

        self.accelerator.print("[Trainer] accelerator.prepare starting...")
        self.model, self.optimizer, self.train_loader, self._val_loader = (
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
                self._val_loader,
            )
        )
        self.accelerator.print("[Trainer] accelerator.prepare finished.")
        acc = self.accelerator
        self.accelerator.print(
            f"[Trainer] backend: distributed_type={acc.state.distributed_type!s} "
            f"deepspeed_active={acc.state.distributed_type == DistributedType.DEEPSPEED} "
            f"deepspeed_plugin={'set' if getattr(acc.state, 'deepspeed_plugin', None) else 'none'} "
            f"mixed_precision={acc.mixed_precision!s} "
            f"prepared_model={type(self.model).__name__}"
        )

        steps_per_epoch = len(self.train_loader)
        if not self._eval_only and steps_per_epoch <= 0:
            raise ValueError(
                "Training DataLoader is empty. Check dataset size, "
                "batch_size_per_gpu, and drop_last settings."
            )
        grad_accum = max(1, int(cfg_get(trainer_cfg, "gradient_accumulation_steps", 1)))
        steps_per_epoch_opt = max(1, math.ceil(steps_per_epoch / grad_accum))

        if self._eval_only:
            self._total_steps = 0
        elif self._max_epochs > 0:
            self._total_steps = self._max_epochs * steps_per_epoch_opt
        elif self._max_steps > 0:
            self._total_steps = self._max_steps
        else:
            raise ValueError("Set trainer.max_epochs or trainer.max_steps.")

        self._eval_interval = self._resolve_eval_interval_steps(
            self._eval_interval_raw,
            self._total_steps,
        )

        train_steps = max(self._total_steps, 1)
        warmup = int(train_steps * self._lr_warmup_ratio)
        self.scheduler = _cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=train_steps,
        )
        self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

        self.accelerator.print(
            f"[Trainer] eval_only={self._eval_only} "
            f"steps/epoch={steps_per_epoch_opt} total_steps={self._total_steps} "
            f"eval_interval={self._eval_interval} (raw={self._eval_interval_raw!r}) "
            f"save_checkpoint_steps={sorted(self._save_checkpoint_steps)}"
        )

        save_dir = os.path.join(self._save_root, "checkpoints")
        if self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir
        self._best_tracker = BestCheckpointTracker(save_dir, k=5, mode=self._best_mode)

    def _loss_weight(self, name: str, step: int | None = None) -> float:
        entry = cfg_get(self._loss_cfg, name, {})
        if isinstance(entry, dict):
            weight = float(cfg_get(entry, "weight", 1.0))
            warmup_start = cfg_get(entry, "warmup_start", None)
            warmup_end = cfg_get(entry, "warmup_end", None)
            if warmup_start is None or warmup_end is None:
                return weight
            start = int(warmup_start)
            end = int(warmup_end)
            if end <= start:
                return weight
            current = self.global_step if step is None else int(step)
            if current < start:
                return 0.0
            if current >= end:
                return weight
            return weight * (float(current - start) / float(end - start))
        return 1.0

    def compute_total_loss(self, loss_dict: dict[str, torch.Tensor], step: int | None = None) -> torch.Tensor:
        device = self.accelerator.device
        total = torch.zeros((), device=device)
        if self._loss_cfg:
            missing: list[str] = []
            for name in self._loss_cfg:
                val = loss_dict.get(name)
                if isinstance(val, torch.Tensor) and val.dim() == 0:
                    total = total + self._loss_weight(name, step=step) * val
                else:
                    missing.append(str(name))
            if missing:
                available = ", ".join(sorted(loss_dict.keys())) or "<none>"
                raise KeyError(
                    f"Configured losses missing or non-scalar: {missing}. "
                    f"Available model losses: {available}"
                )
        else:
            used = 0
            for val in loss_dict.values():
                if isinstance(val, torch.Tensor) and val.dim() == 0:
                    total = total + val
                    used += 1
            if used == 0:
                raise ValueError("Model returned no scalar losses.")
        return total

    def _current_loss_weights(self, step: int | None = None) -> dict[str, float]:
        if not self._loss_cfg:
            return {}
        return {str(name): self._loss_weight(str(name), step=step) for name in self._loss_cfg}

    def _checkpoint_selection_metric(self, val_subs: dict[str, float]) -> tuple[str, float]:
        key = self._best_metric.strip() if self._best_metric else "val_total"
        if key in val_subs:
            return key, float(val_subs[key])
        if not self._allow_best_metric_fallback:
            available = ", ".join(sorted(val_subs)) or "<none>"
            raise KeyError(
                f"trainer.best_metric={key!r} was not returned by validation. "
                f"Available metrics: {available}"
            )
        for fallback in ("val_total", "total_loss", "loss"):
            if fallback in val_subs:
                return fallback, float(val_subs[fallback])
        return "val_total", 0.0

    @staticmethod
    def _make_rich_progress(*, disable: bool = False) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            SpinnerColumn(),
            RenderableColumn(),
            disable=disable,
        )

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if len(self._val_loader) == 0:
            return {}

        was_training = self.model.training
        self.model.eval()
        acc = self.accelerator
        total_steps = min(self._eval_max_batches, len(self._val_loader))
        val_task = self.progress.add_task(description="[Val]", total=total_steps)
        sums: dict[str, float] = {}
        counts: dict[str, float] = {}

        try:
            for batch_idx, batch in enumerate(self._val_loader):
                if batch_idx >= self._eval_max_batches:
                    break
                inputs = batch_to_device(batch, acc.device)
                inputs = _prepare_model_batch(self.model, inputs)
                _set_model_train_step(self.model, self.global_step)
                _set_model_loss_weights(self.model, self._current_loss_weights(step=self.global_step))
                batch_size = 1
                if isinstance(inputs, dict):
                    for input_value in inputs.values():
                        if isinstance(input_value, torch.Tensor) and input_value.ndim > 0:
                            batch_size = int(input_value.shape[0])
                            break
                with acc.autocast():
                    loss_dict, other_log = unpack_step_out(self.model(inputs))
                    total_loss = self.compute_total_loss(loss_dict, step=self.global_step)
                metrics = {"total": total_loss, **loss_dict, **other_log}
                for name, value in metrics.items():
                    if not isinstance(value, torch.Tensor) or value.dim() != 0:
                        continue
                    weight = torch.tensor([batch_size], device=acc.device, dtype=torch.float32)
                    gathered = acc.gather_for_metrics(value.detach().reshape(1).float() * weight)
                    gathered_weight = acc.gather_for_metrics(weight)
                    key = f"val_{name}" if not str(name).startswith("val_") else str(name)
                    sums[key] = sums.get(key, 0.0) + float(gathered.sum().item())
                    counts[key] = counts.get(key, 0.0) + float(gathered_weight.sum().item())
                self.progress.update(val_task, advance=1)
        finally:
            self.progress.remove_task(val_task)
            self.model.train(was_training)
        return {
            key: sums[key] / max(counts.get(key, 1), 1)
            for key in sorted(sums)
        }

    def _log_metrics_trackers(
        self,
        step: int,
        loss_dict: dict,
        *,
        other_log: dict | None = None,
        prefix: str = "train",
    ) -> None:
        scalars = {}
        for item in (loss_dict, other_log or {}):
            for key, value in item.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    scalars[key] = value.detach()
        tracker = self._experiment_tracker
        if scalars and tracker == "wandb":
            wandb_utils.log_metrics(step, scalars, series_label=prefix)
        if self.optimizer is None:
            return
        if tracker == "wandb":
            wandb_utils.log_learning_rate(self.optimizer, step)

    def _eval_only_run(self) -> None:
        acc = self.accelerator
        self.progress.start()
        set_sampler_epoch(self._val_loader, self.global_step)
        acc.wait_for_everyone()
        val_subs = self.validate()
        if acc.is_main_process and val_subs:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in val_subs.items())
            acc.print(f"\n[Trainer] eval_only | {parts}")
        self.progress.stop()

    def _after_eval_interval(
        self,
        step_i: int,
        epoch_count: int,
        val_subs: dict[str, float],
    ) -> None:
        acc = self.accelerator
        if not val_subs:
            return
        selection = self._checkpoint_selection_metric(val_subs)
        if acc.is_main_process:
            self._log_metrics_trackers(
                step_i,
                dict_scalars_to_tensors(val_subs),
                prefix="val",
            )
            requested = self._best_metric.strip() if self._best_metric else "val_total"
            if selection[0] != requested:
                acc.print(
                    f"[Trainer] best_metric={requested!r} not found; "
                    f"using {selection[0]!r} for checkpoint selection."
                )
        self._save_checkpoint(step_i, epoch_count, checkpoint_name="last")
        keep_best = (
            self._best_tracker is not None
            and acc.is_main_process
            and self._best_tracker.would_keep(selection[1])
        )
        keep_best = _broadcast_bool(keep_best, acc.device)
        if keep_best:
            ckpt_dir = self._save_checkpoint(step_i, epoch_count)
            if self._best_tracker is not None and acc.is_main_process:
                self._best_tracker.update(ckpt_dir, selection[1])
        if acc.is_main_process:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in val_subs.items())
            acc.print(
                f"\n[Step {step_i}] val [{selection[0]}]={selection[1]:.6f} | {parts}"
            )

    def train(self) -> None:
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Call trainer.setup() before trainer.train().")

        acc = self.accelerator
        if self._eval_only:
            self._eval_only_run()
            return

        model = self.model
        model.train()
        step_i = self.global_step
        epoch_count = self._resume_epoch_count

        self.progress.start()
        self._train_progress_task = self.progress.add_task(
            description="[Train]",
            total=self._total_steps,
        )
        self.progress.reset(self._train_progress_task, completed=step_i)

        previous_time = time.perf_counter()
        while step_i < self._total_steps:
            set_sampler_epoch(self.train_loader, epoch_count)
            for batch in self.train_loader:
                if step_i >= self._total_steps:
                    break

                inputs = batch_to_device(batch, acc.device)
                inputs = _prepare_model_batch(model, inputs)
                _set_model_train_step(model, step_i)
                _set_model_loss_weights(model, self._current_loss_weights(step=step_i))
                _set_model_train_step(model, step_i)
                with acc.accumulate(model):
                    with acc.autocast():
                        loss_dict, other_log = unpack_step_out(self.model(inputs))
                        total_loss = self.compute_total_loss(loss_dict, step=step_i)

                    if not _is_finite_everywhere(total_loss):
                        acc.print(f"[Skip] Loss NaN/Inf at step {step_i}")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    acc.backward(total_loss)

                    if acc.sync_gradients:
                        acc.clip_grad_norm_(model.parameters(), self._gradient_clipping)

                    self.optimizer.step()
                    if acc.sync_gradients:
                        unwrapped = unwrap_model_for_non_deepspeed(model)
                        after_step = getattr(unwrapped, "after_optimizer_step", None)
                        if callable(after_step):
                            after_step(global_step=step_i + 1)
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if not acc.sync_gradients:
                    continue

                step_i += 1
                self.global_step = step_i
                now = time.perf_counter()
                dt = max(now - previous_time, 1e-12)
                previous_time = now
                desc = (
                    f"[Train] ep={epoch_count + 1} | "
                    f"loss={float(total_loss.detach().item()):.4f} | {1.0 / dt:.2f} step/s"
                )
                self.progress.update(self._train_progress_task, advance=1, description=desc)

                if acc.is_main_process and step_i % self._log_interval == 0:
                    self._log_metrics_trackers(
                        step_i,
                        {"total": total_loss.detach(), **loss_dict},
                        other_log=other_log,
                        prefix="train",
                    )

                if (
                    step_i in self._save_checkpoint_steps
                    and step_i not in self._saved_checkpoint_steps
                ):
                    acc.wait_for_everyone()
                    ckpt_dir = self._save_checkpoint(
                        step_i,
                        epoch_count,
                        checkpoint_name=f"step_{step_i}",
                    )
                    self._saved_checkpoint_steps.add(step_i)
                    if acc.is_main_process:
                        acc.print(
                            f"[Trainer] Required checkpoint step={step_i} -> {ckpt_dir}"
                        )
                    acc.wait_for_everyone()

                if (
                    self._eval_interval > 0
                    and step_i % self._eval_interval == 0
                    and len(self._val_loader) > 0
                ):
                    set_sampler_epoch(self._val_loader, step_i)
                    acc.wait_for_everyone()
                    self._after_eval_interval(step_i, epoch_count, self.validate())
            else:
                epoch_count += 1
                if self._save_checkpoint_every_epoch and step_i > 0:
                    acc.wait_for_everyone()
                    ep_dir = self._save_checkpoint(step_i, epoch_count)
                    if acc.is_main_process:
                        acc.print(
                            f"[Trainer] Epoch-end checkpoint "
                            f"(completed_epochs={epoch_count}) -> {ep_dir}"
                        )
                    acc.wait_for_everyone()

        self.progress.stop()
        self._train_progress_task = None
        if step_i > 0:
            self._save_checkpoint(step_i, epoch_count, checkpoint_name="last")

    def _save_checkpoint(
        self,
        step: int,
        epoch: int,
        *,
        checkpoint_name: str | None = None,
    ) -> str:
        return save_checkpoint(
            self.model,
            self.scheduler,
            self.optimizer,
            config_to_serializable(self.cfg),
            self._save_dir,
            step=step,
            epoch=epoch,
            accelerator=self.accelerator,
            checkpoint_name=checkpoint_name,
        )

    def resume_from(self, ckpt_path: str) -> None:
        if self.scheduler is None:
            raise RuntimeError("Call trainer.setup() before resume_from().")
        unwrapped = unwrap_model_for_non_deepspeed(self.model)
        state, start_step = load_checkpoint(
            ckpt_path,
            unwrapped,
            log_fn=self.accelerator.print,
            write_key_reports=self.accelerator.is_main_process,
            key_report_dir=self._save_root,
            strict=True,
        )
        self.global_step = start_step
        _set_model_train_step(self.model, start_step)
        epoch = state.get("epoch")
        self._resume_epoch_count = int(epoch) if epoch is not None else 0
        sched_sd = state.get("scheduler_state_dict")
        if sched_sd is not None:
            self.scheduler.load_state_dict(sched_sd)
            self.accelerator.print("  Scheduler state restored")
        opt_sd = state.get("optimizer_state_dict")
        is_deepspeed = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if is_deepspeed and opt_sd is None and not self._eval_only:
            raise RuntimeError(
                "This portable checkpoint does not contain DeepSpeed optimizer "
                "state. Use a non-DeepSpeed checkpoint for strict resume, or add "
                "a DeepSpeed-native save/load path."
            )
        if opt_sd is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(opt_sd)
            self.accelerator.print("  Optimizer state restored")

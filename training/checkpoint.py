"""Generic checkpoint management."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Callable

import torch

from utils.utils import unwrap_model_for_non_deepspeed


MODEL_CHECKPOINT = "model_state_dict.pt"
STATE_FILE = "state.pt"


def _unwrap_state_dict(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict checkpoint payload, got {type(obj).__name__}")
    for key in ("state_dict", "model_state_dict"):
        if isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def _load_sd_and_state_from_ckpt_path(
    ckpt_path: str,
    map_location: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    state: dict[str, Any] = {}
    if os.path.isdir(path):
        model_path = os.path.join(path, MODEL_CHECKPOINT)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing {MODEL_CHECKPOINT} in {path}")
        sd = _unwrap_state_dict(
            torch.load(model_path, map_location=map_location, weights_only=False)
        )
        state_path = os.path.join(path, STATE_FILE)
        if os.path.isfile(state_path):
            raw = torch.load(state_path, map_location=map_location, weights_only=False)
            state = raw if isinstance(raw, dict) else {}
    elif os.path.isfile(path):
        sd = _unwrap_state_dict(
            torch.load(path, map_location=map_location, weights_only=False)
        )
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return sd, state


def _write_key_diff_reports(
    ckpt_path: str,
    missing: list[str],
    unexpected: list[str],
    log_fn: Callable[[str], None],
    *,
    shape_mismatches: list[dict[str, Any]] | None = None,
    report_dir: str | None = None,
    report_stem: str | None = None,
) -> None:
    path = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    if report_dir:
        out_dir = os.path.normpath(os.path.expanduser(str(report_dir)))
        stem = report_stem or os.path.splitext(os.path.basename(path.rstrip(os.sep)))[0]
    elif os.path.isdir(path):
        out_dir = path
        stem = ""
    else:
        out_dir = os.path.dirname(path)
        stem = os.path.splitext(os.path.basename(path))[0]

    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{stem}_" if stem else ""
    if missing:
        out = os.path.join(out_dir, f"{prefix}missing_keys.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
            f.write("\n")
        log_fn(f"  Saved missing keys ({len(missing)}): {out}")
    if unexpected:
        out = os.path.join(out_dir, f"{prefix}unexpected_keys.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(unexpected))
            f.write("\n")
        log_fn(f"  Saved unexpected keys ({len(unexpected)}): {out}")
    shape_mismatches = shape_mismatches or []
    if shape_mismatches:
        out = os.path.join(out_dir, f"{prefix}shape_mismatches.txt")
        with open(out, "w", encoding="utf-8") as f:
            for item in shape_mismatches:
                f.write(
                    f"{item['key']}\tckpt{tuple(item['checkpoint_shape'])}"
                    f"\tmodel{tuple(item['model_shape'])}\n"
                )
        log_fn(f"  Saved shape mismatches ({len(shape_mismatches)}): {out}")

    if missing or unexpected or shape_mismatches:
        report = {
            "log_prefix": report_stem or "checkpoint.resume",
            "checkpoint_path": path,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "shape_mismatch_count": len(shape_mismatches),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
        }
        with open(os.path.join(out_dir, "checkpoint_load_report.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, sort_keys=True))
            f.write("\n")
        with open(os.path.join(out_dir, "checkpoint_load_report.txt"), "a", encoding="utf-8") as f:
            f.write(f"[{report['log_prefix']}]\n")
            f.write(f"checkpoint: {path}\n")
            f.write(
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"shape_mismatches={len(shape_mismatches)}\n\n"
            )


def _report_state_dict_diff(
    ckpt_path: str,
    log_fn: Callable[[str], None],
    missing: list[str],
    unexpected: list[str],
    *,
    shape_mismatches: list[dict[str, Any]] | None = None,
    write_key_reports: bool,
    key_report_dir: str | None = None,
    key_report_stem: str | None = None,
) -> None:
    shape_mismatches = shape_mismatches or []
    if not missing and not unexpected and not shape_mismatches:
        log_fn("  Checkpoint matched model state dict.")
        return
    if missing:
        log_fn(f"  Missing keys: {len(missing)} - {missing[:8]}")
    if unexpected:
        log_fn(f"  Unexpected keys: {len(unexpected)} - {unexpected[:8]}")
    if shape_mismatches:
        examples = [
            f"{item['key']}: ckpt{tuple(item['checkpoint_shape'])} != model{tuple(item['model_shape'])}"
            for item in shape_mismatches[:8]
        ]
        log_fn(f"  Shape mismatches: {len(shape_mismatches)} - {examples}")
    if write_key_reports:
        _write_key_diff_reports(
            ckpt_path,
            missing,
            unexpected,
            log_fn,
            shape_mismatches=shape_mismatches,
            report_dir=key_report_dir,
            report_stem=key_report_stem,
        )


def _state_dict_key_diff(
    model_state: dict[str, Any],
    checkpoint_state: dict[str, Any],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches: list[dict[str, Any]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_value = model_state[key]
        ckpt_value = checkpoint_state[key]
        if (
            torch.is_tensor(model_value)
            and torch.is_tensor(ckpt_value)
            and tuple(model_value.shape) != tuple(ckpt_value.shape)
        ):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(ckpt_value.shape),
                    "model_shape": list(model_value.shape),
                }
            )
    return missing, unexpected, shape_mismatches


def _should_write_checkpoint(accelerator, save_only_rank0: bool) -> bool:
    return accelerator is None or not save_only_rank0 or bool(accelerator.is_main_process)


def save_checkpoint(
    model,
    scheduler,
    optimizer,
    config,
    save_dir: str,
    step: int,
    epoch: int | None = None,
    accelerator=None,
    save_only_rank0: bool = True,
    checkpoint_name: str | None = None,
) -> str:
    step_int = int(step)
    if checkpoint_name:
        ckpt_name = checkpoint_name
    elif epoch is not None:
        ckpt_name = f"epoch_{int(epoch):04d}_step_{step_int}"
    else:
        ckpt_name = f"step_{step_int}"
    ckpt_dir = os.path.join(save_dir, ckpt_name)

    if accelerator is not None and getattr(accelerator.state, "deepspeed_plugin", None):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
        state_dict = accelerator.get_state_dict(model)
    elif accelerator is not None:
        unwrapped = unwrap_model_for_non_deepspeed(model)
        state_dict = unwrapped.state_dict()
    else:
        unwrapped = unwrap_model_for_non_deepspeed(model)
        state_dict = unwrapped.state_dict()

    payload = {
        "step": step_int,
        "epoch": int(epoch) if epoch is not None else None,
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    is_deepspeed = accelerator is not None and getattr(accelerator.state, "deepspeed_plugin", None)
    if optimizer is not None and not is_deepspeed:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    elif is_deepspeed:
        payload["optimizer_state_dict"] = None
        payload["optimizer_state_note"] = (
            "DeepSpeed optimizer state is engine-managed and is not stored in this "
            "portable model checkpoint."
        )

    if _should_write_checkpoint(accelerator, save_only_rank0):
        if checkpoint_name == "last" and os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(ckpt_dir, MODEL_CHECKPOINT))
        torch.save(payload, os.path.join(ckpt_dir, STATE_FILE))

    return ckpt_dir


def load_checkpoint(
    ckpt_path: str,
    model,
    map_location: Any = "cpu",
    log_fn: Callable[[str], None] = print,
    write_key_reports: bool = True,
    key_report_dir: str | None = None,
    key_report_stem: str | None = None,
    strict: bool = True,
) -> tuple[dict[str, Any], int]:
    path = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    log_fn(f"Loading checkpoint: {path}")
    sd, state = _load_sd_and_state_from_ckpt_path(path, map_location)
    model_state = model.state_dict()
    missing, unexpected, shape_mismatches = _state_dict_key_diff(model_state, sd)
    load_sd = sd
    if shape_mismatches:
        mismatch_keys = {item["key"] for item in shape_mismatches}
        load_sd = {k: v for k, v in sd.items() if k not in mismatch_keys}
    loaded_missing, loaded_unexpected = model.load_state_dict(load_sd, strict=False)
    if not shape_mismatches and list(loaded_missing) != missing:
        missing = list(loaded_missing)
    if not shape_mismatches and list(loaded_unexpected) != unexpected:
        unexpected = list(loaded_unexpected)
    _report_state_dict_diff(
        path,
        log_fn,
        list(missing),
        list(unexpected),
        shape_mismatches=shape_mismatches,
        write_key_reports=write_key_reports,
        key_report_dir=key_report_dir,
        key_report_stem=key_report_stem,
    )
    if strict and (missing or unexpected or shape_mismatches):
        raise RuntimeError(
            "Checkpoint resume is strict by default and model keys did not match: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"shape_mismatches={len(shape_mismatches)}. "
            "Use a dedicated partial-initialization path for architecture changes."
        )

    if state.get("step") is not None:
        start_step = int(state["step"])
    else:
        match = re.search(r"step_(\d+)", os.path.basename(path.rstrip(os.sep)))
        start_step = int(match.group(1)) if match else 0
    log_fn(f"  Resuming from completed step {start_step}")
    if state.get("epoch") is not None:
        log_fn(f"  Checkpoint epoch={int(state['epoch'])}")
    return state, start_step


class BestCheckpointTracker:
    """Keeps the top-k checkpoint directories by one validation metric."""

    def __init__(self, save_dir: str, k: int = 5, mode: str = "min"):
        self.save_dir = save_dir
        self.k = int(k)
        self.mode = mode
        self._higher_is_better = mode == "max"
        self._meta_path = os.path.join(save_dir, "best_checkpoints.json")
        self._entries: list[dict] = []
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, encoding="utf-8") as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._entries = []
        self._sort_entries_inplace()

    def _sort_entries_inplace(self) -> None:
        self._entries.sort(
            key=lambda item: float(item["metric"]),
            reverse=self._higher_is_better,
        )

    def _is_better(self, new_val: float, old_val: float) -> bool:
        return new_val < old_val if self.mode == "min" else new_val > old_val

    def would_keep(self, metric_value: float) -> bool:
        if len(self._entries) < self.k:
            return True
        self._sort_entries_inplace()
        worst_metric = float(self._entries[-1]["metric"])
        return self._is_better(float(metric_value), worst_metric)

    def ranked_entries(self) -> list[dict[str, Any]]:
        self._sort_entries_inplace()
        return [dict(entry) for entry in self._entries]

    def update(self, ckpt_dir: str, metric_value: float) -> bool:
        entry = {"path": os.path.basename(ckpt_dir), "metric": float(metric_value)}
        if len(self._entries) < self.k:
            self._entries.append(entry)
            self._sort_entries_inplace()
            self._persist()
            return True

        self._sort_entries_inplace()
        worst = self._entries[-1]
        if not self._is_better(float(metric_value), float(worst["metric"])):
            return False

        evicted = self._entries.pop()
        self._entries.append(entry)
        self._sort_entries_inplace()
        self._persist()

        old_dir = os.path.join(self.save_dir, evicted["path"])
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)
        return True

    def _persist(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2)

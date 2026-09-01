"""Shared helpers for V-JEPA world-model variants."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _is_rank_zero_process() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return os.environ.get("RANK", "0") in ("", "0")


def _safe_report_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem.strip("._") or "checkpoint"


def _write_checkpoint_load_report(report_dir: str | None, report: dict[str, Any]) -> None:
    if not report_dir or not _is_rank_zero_process():
        return
    out_dir = os.path.normpath(os.path.expanduser(str(report_dir)))
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "checkpoint_load_report.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, sort_keys=True))
        f.write("\n")

    txt_path = os.path.join(out_dir, "checkpoint_load_report.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"[{report['log_prefix']}]\n")
        f.write(f"checkpoint: {report.get('checkpoint_path') or ''}\n")
        f.write(
            "loaded_keys={loaded_keys} loaded_numel_ratio={loaded_numel_ratio:.6f} "
            "missing={missing_count} unexpected={unexpected_count} "
            "shape_mismatches={shape_mismatch_count}\n\n".format(**report)
        )

    detail_dir = os.path.join(out_dir, "checkpoint_load_keys")
    os.makedirs(detail_dir, exist_ok=True)
    stem = _safe_report_stem(str(report["log_prefix"]))
    details = {
        "missing_keys": report["missing_keys"],
        "unexpected_keys": report["unexpected_keys"],
        "shape_mismatches": [
            f"{item['key']}\tckpt{tuple(item['checkpoint_shape'])}\tmodel{tuple(item['model_shape'])}"
            for item in report["shape_mismatches"]
        ],
    }
    for suffix, values in details.items():
        path = os.path.join(detail_dir, f"{stem}.{suffix}.txt")
        with open(path, "w", encoding="utf-8") as f:
            if values:
                f.write("\n".join(str(v) for v in values))
                f.write("\n")


def _as_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (int(value), int(value))
    return (int(value[0]), int(value[1]))


def _strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod.", "model.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _select_checkpoint_state(payload: Any, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        return {}
    for wrapper_key in ("state_dict", "model_state_dict"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, dict):
            selected = _select_checkpoint_state(wrapped, names)
            if selected:
                return selected
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
        prefix = f"{name}."
        prefixed = {
            _strip_prefixes(str(k))[len(prefix):]: v
            for k, v in payload.items()
            if (
                isinstance(k, str)
                and _strip_prefixes(str(k)).startswith(prefix)
                and torch.is_tensor(v)
            )
        }
        if prefixed:
            return prefixed
    if payload and all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
        return payload
    return {}


def _load_state_from_path(path: str) -> dict[str, torch.Tensor]:
    norm_path = os.path.normpath(os.path.expanduser(str(path)))
    if os.path.isdir(norm_path):
        model_path = os.path.join(norm_path, "model_state_dict.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model_state_dict.pt in checkpoint directory: {norm_path}")
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    elif os.path.isfile(norm_path):
        payload = torch.load(norm_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {norm_path}")
    state = _select_checkpoint_state(payload, ("state_dict", "model_state_dict"))
    if not state:
        raise RuntimeError(f"No state_dict tensors found in checkpoint: {norm_path}")
    return state


def _checkpoint_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        path = str(value).strip()
        return [path] if path else []
    if isinstance(value, Sequence):
        paths: list[str] = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError(
                    "model.init_from_checkpoint entries must be paths, "
                    f"got {type(item).__name__}"
                )
            path = str(item).strip()
            if path:
                paths.append(path)
        return paths
    raise TypeError(
        "model.init_from_checkpoint must be a path or a list of paths, "
        f"got {type(value).__name__}"
    )


def _load_submodule_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    log_prefix: str,
    min_load_ratio: float = 0.95,
    allow_partial: bool = False,
    require_non_empty: bool = False,
    checkpoint_path: str | None = None,
    report_dir: str | None = None,
) -> float:
    own = module.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    loaded_numel = 0
    total_numel = sum(t.numel() for t in own.values())
    for raw_key, value in state.items():
        key = _strip_prefixes(str(raw_key))
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            filtered[key] = value
            loaded_numel += own[key].numel()
        elif key in own:
            skipped_shape.append(
                (key, tuple(value.shape), tuple(own[key].shape))
            )
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    unexpected = sorted(_normalized_state_keys(state) - set(own.keys()))
    ratio = loaded_numel / max(total_numel, 1)
    report = {
        "log_prefix": str(log_prefix),
        "checkpoint_path": os.path.normpath(os.path.expanduser(str(checkpoint_path))) if checkpoint_path else None,
        "module_type": type(module).__name__,
        "loaded_keys": len(filtered),
        "loaded_numel": int(loaded_numel),
        "total_numel": int(total_numel),
        "loaded_numel_ratio": float(ratio),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "shape_mismatch_count": len(skipped_shape),
        "missing_keys": [str(k) for k in missing],
        "unexpected_keys": [str(k) for k in unexpected],
        "shape_mismatches": [
            {
                "key": key,
                "checkpoint_shape": list(ckpt_shape),
                "model_shape": list(model_shape),
            }
            for key, ckpt_shape, model_shape in skipped_shape
        ],
    }
    _write_checkpoint_load_report(report_dir, report)
    logger.info(
        f"[{log_prefix}] loaded={len(filtered)} skipped_shape={len(skipped_shape)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"loaded_numel_ratio={ratio:.4f}"
    )
    if skipped_shape:
        examples = "; ".join(
            f"{key}: ckpt{ckpt_shape} != model{model_shape}"
            for key, ckpt_shape, model_shape in skipped_shape[:20]
        )
        suffix = "" if len(skipped_shape) <= 20 else f"; ... +{len(skipped_shape) - 20} more"
        logger.warning(f"[{log_prefix}] shape mismatches: {examples}{suffix}")
    if missing:
        examples = ", ".join(str(k) for k in missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... +{len(missing) - 20} more"
        logger.warning(f"[{log_prefix}] missing model keys after load: {examples}{suffix}")
    if unexpected:
        examples = ", ".join(str(k) for k in unexpected[:20])
        suffix = "" if len(unexpected) <= 20 else f", ... +{len(unexpected) - 20} more"
        logger.warning(f"[{log_prefix}] unexpected checkpoint keys: {examples}{suffix}")
    if require_non_empty and not filtered:
        raise RuntimeError(
            f"{log_prefix} loaded 0 matching tensors from checkpoint: {checkpoint_path}. "
            "Check that this checkpoint contains model weights for the current architecture."
        )
    if not allow_partial and ratio < float(min_load_ratio):
        mismatch_msg = ""
        if skipped_shape:
            examples = "; ".join(
                f"{key}: ckpt{ckpt_shape} != model{model_shape}"
                for key, ckpt_shape, model_shape in skipped_shape[:5]
            )
            mismatch_msg = f" Shape mismatches include: {examples}."
        raise RuntimeError(
            f"{log_prefix} checkpoint load ratio {ratio:.4f} < {min_load_ratio:.4f}. "
            "Check variant/checkpoint compatibility or set model.allow_partial_load=true."
            f"{mismatch_msg}"
        )
    return ratio


def _normalized_state_keys(state: dict[str, torch.Tensor]) -> set[str]:
    return {_strip_prefixes(str(key)) for key in state}


def _chunk_layer_norm(x: torch.Tensor, chunk_dim: int) -> torch.Tensor:
    if x.size(-1) % chunk_dim != 0:
        return F.layer_norm(x, (x.size(-1),))
    chunks = [
        F.layer_norm(x[..., i : i + chunk_dim], (chunk_dim,))
        for i in range(0, x.size(-1), chunk_dim)
    ]
    return torch.cat(chunks, dim=-1)


def _as_float_tuple(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(x) for x in value)


def _module_param_dtype(module: nn.Module | None) -> torch.dtype | None:
    if module is None:
        return None
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    return None


def _module_param_device(module: nn.Module | None) -> torch.device | None:
    if module is None:
        return None
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return None


def _cast_floating_tensor(value: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    if dtype is None or not value.is_floating_point() or value.dtype == dtype:
        return value
    return value.to(dtype=dtype)


def _move_tensor_for_module(
    value: torch.Tensor,
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if device is not None and value.device != device:
        kwargs["device"] = device
    if dtype is not None and value.is_floating_point() and value.dtype != dtype:
        kwargs["dtype"] = dtype
    if not kwargs:
        return value
    return value.to(**kwargs)


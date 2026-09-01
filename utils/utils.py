from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

# 每个 checkpoint 子目录内文件名
MODEL_CHECKPOINT = "model_state_dict.pt"
STATE = "state.pt"


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from dict-like configs (``.get``) or attribute-style objects (``getattr``).

    Works with ``dict``, OmegaConf ``DictConfig``, ``SimpleNamespace``, dataclasses, etc.
    """
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


_STR_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def cfg_model_torch_dtype(model_cfg: Any, *, default: str = "bfloat16") -> torch.dtype:
    """Parse ``model.dtype`` (YAML string) for pretrained weights and dtype alignment."""
    s = str(cfg_get(model_cfg, "dtype", default)).strip().lower()
    return _STR_TO_TORCH_DTYPE.get(s, torch.bfloat16)


def resolve_experiment_tracker(cfg: Any) -> str:
    """Return ``"none"`` or ``"wandb"`` from ``experiment_tracker``.

    An unrecognized value resolves to ``"none"`` instead of raising, so a config written
    for a tracker this build does not ship still trains; it just reports no metrics.
    """
    v = cfg_get(cfg, "experiment_tracker", None)
    if v is not None:
        s = str(v).strip().lower()
        if s == "wandb":
            return s
        if s in ("none", "null", "off", "disabled", "false", "0", ""):
            return "none"
    return "none"


def user_from_env() -> str:
    """Current login / user id used in save paths and experiment tags.

    Prefers environment variables ``whoami`` or ``WHOAMI`` (e.g. ``export whoami=$(whoami)``), then
    ``USER``, ``LOGNAME``, or ``USERNAME``. Returns ``""`` if none are set.
    """
    w = (os.environ.get("whoami") or os.environ.get("WHOAMI") or "").strip()
    if w:
        return w
    return (os.environ.get("USER") or os.environ.get("LOGNAME") or os.environ.get("USERNAME") or "").strip()


def dict_scalars_to_tensors(
    d: Mapping[str, Union[int, float]],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """浅层 ``dict``：值须为 ``int`` / ``float``，转为 0 维 ``Tensor``。"""
    return {k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in d.items()}


def unpack_step_out(step_out: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """解析 ``model(batch)`` 返回：``{"loss": dict, "other_log"?: dict}``。"""
    if not isinstance(step_out, dict) or "loss" not in step_out:
        raise TypeError(
            "model forward must return dict with 'loss' mapping (and optional 'other_log')."
        )
    loss_dict = step_out["loss"]
    if not isinstance(loss_dict, dict):
        raise TypeError("step_out['loss'] must be a dict[str, Tensor].")
    ol = step_out.get("other_log")
    other_log = ol if isinstance(ol, dict) else {}
    return loss_dict, other_log


def unwrap_model_for_non_deepspeed(model: nn.Module) -> nn.Module:
    """Unwrap common parallel/compiled wrappers without importing DeepSpeed.

    ``accelerator.unwrap_model`` imports DeepSpeed whenever it is installed, even
    for non-DeepSpeed runs. Some smoke-test environments intentionally do not
    expose ``CUDA_HOME``, so importing DeepSpeed ops can fail before training
    starts. DeepSpeed-enabled paths should still use Accelerate's unwrap/get
    state dict helpers.
    """
    unwrapped: Any = model
    for _ in range(16):
        next_model = getattr(unwrapped, "module", None)
        if next_model is None:
            break
        unwrapped = next_model
    return getattr(unwrapped, "_orig_mod", unwrapped)


def batch_to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    """将 batch 中 ``Tensor`` 搬到 ``device``，其余键原样保留。"""
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# --- YAML freeze rules and parameter learning-rate groups -------------------
# trainer: freeze_modules and learning_rate entries can target submodule paths
# or parameter-name glob patterns.


def freeze_modules(model: nn.Module, spec: str | list[str] | None) -> list[str]:
    if not spec:
        return []
    if isinstance(spec, str):
        paths = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        paths = list(spec)

    frozen: list[str] = []
    for path in paths:
        has_wildcard = any(c in path for c in ("*", "?", "["))

        if not has_wildcard:
            module = _resolve_module(model, path)
            if module is None:
                print(f"[freeze] warning: module path not found, skipping: {path}")
                continue
            module.requires_grad_(False)
            frozen.append(path)
        else:
            matched = False
            for name, module in model.named_modules():
                if not name:
                    continue
                if fnmatch.fnmatch(name, path):
                    module.requires_grad_(False)
                    if not matched:
                        frozen.append(path)
                        matched = True
            if not matched:
                print(f"[freeze] warning: no modules matched pattern: {path}")
    return frozen


def _resolve_module(model: nn.Module, path: str) -> nn.Module | None:
    m = model
    try:
        for attr in path.split("."):
            m = getattr(m, attr)
        return m
    except AttributeError:
        return None


def _match_params_by_pattern(
    model: nn.Module, pattern: str
) -> list[nn.Parameter]:
    has_wildcard = any(c in pattern for c in ("*", "?", "["))

    if not has_wildcard:
        mod = _resolve_module(model, pattern)
        if mod is not None:
            return [p for p in mod.parameters() if p.requires_grad]
        return []

    matched: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, pattern + ".*"):
            matched.append(param)
    return matched


def build_param_lr_groups(
    model: nn.Module,
    lr_cfg: dict[str, Any],
) -> list[dict]:
    base_lr = float(lr_cfg.get("base", 1e-4))
    used_ids: set[int] = set()
    groups: list[dict] = []
    names_by_id = {id(p): name for name, p in model.named_parameters()}

    def is_no_decay(param: nn.Parameter) -> bool:
        name = names_by_id.get(id(param), "")
        return (
            param.ndim <= 1
            or name.endswith(".bias")
            or any(part in name for part in ("pos_embed", "mask_tokens", "mod_embed", "query", "fill_token"))
        )

    def append_split_group(params: list[nn.Parameter], *, lr: float, name: str) -> None:
        decay = [p for p in params if not is_no_decay(p)]
        no_decay = [p for p in params if is_no_decay(p)]
        if decay:
            groups.append({"params": decay, "lr": float(lr), "name": name})
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": float(lr),
                    "weight_decay": 0.0,
                    "name": f"{name}.no_decay",
                }
            )

    for pattern, lr in lr_cfg.items():
        if pattern == "base":
            continue
        params = _match_params_by_pattern(model, pattern)
        new_params = [p for p in params if id(p) not in used_ids]
        if new_params:
            append_split_group(new_params, lr=float(lr), name=pattern)
            used_ids.update(id(p) for p in new_params)

    other = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in used_ids
    ]
    if other:
        append_split_group(other, lr=base_lr, name="base")

    return groups


def print_trainable_summary(
    model: nn.Module,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    _print = log_fn or print
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _print(
        f"Parameters: total={total / 1e6:.1f}M  trainable={trainable / 1e6:.1f}M  "
        f"frozen={(total - trainable) / 1e6:.1f}M"
    )
    return total, trainable


def save_parameter_name_lists(model: nn.Module, save_dir: str) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, p in model.named_parameters():
        (trainable_names if p.requires_grad else frozen_names).append(name)
    trainable_names.sort()
    frozen_names.sort()
    path_t = os.path.join(save_dir, "trainable_params.txt")
    path_f = os.path.join(save_dir, "frozen_params.txt")
    with open(path_t, "w", encoding="utf-8") as f:
        f.write("\n".join(trainable_names))
        if trainable_names:
            f.write("\n")
    with open(path_f, "w", encoding="utf-8") as f:
        f.write("\n".join(frozen_names))
        if frozen_names:
            f.write("\n")
    return path_t, path_f


def _unwrap_raw_checkpoint(obj: Any) -> dict[str, Any]:
    """``torch.load`` 结果 → 扁平 ``state_dict``。"""
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint 内容须为 dict，实际: {type(obj).__name__}")
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    # 假定整份即为 state_dict（键多为 str，值为 Tensor）
    return obj


def load_training_checkpoint_payload(
    ckpt_path: str,
    map_location: Any = "cpu",
    model_only: bool = False,
    *,
    include_model_state_dict: bool = True,
) -> dict[str, Any]:
    """读取 ``save_training_state`` 产出的 **目录**（内含 ``model_state_dict.pt``、``state.pt``）。

    若为旧版**单文件** ``.pt``（仅权重），仍可读取。返回 ``model_state_dict``、``state``、``transformer_dir=None``。
    """
    p = os.path.normpath(os.path.expanduser(str(ckpt_path).strip()))
    sd: dict[str, Any] = {}
    st: dict[str, Any] = {}

    if os.path.isdir(p):
        mp = os.path.join(p, MODEL_CHECKPOINT)
        sp = os.path.join(p, STATE)
        if include_model_state_dict:
            if not os.path.isfile(mp):
                raise FileNotFoundError(f"目录中缺少 {MODEL_CHECKPOINT}: {p}")
            raw = torch.load(mp, map_location=map_location, weights_only=False)
            sd = _unwrap_raw_checkpoint(raw)
        if not model_only and os.path.isfile(sp):
            st = torch.load(sp, map_location=map_location, weights_only=False)
    elif os.path.isfile(p):
        if not p.endswith(".pt"):
            raise ValueError(f"checkpoint 须为目录或 .pt 文件: {p}")
        if include_model_state_dict:
            raw = torch.load(p, map_location=map_location, weights_only=False)
            sd = _unwrap_raw_checkpoint(raw)
    else:
        raise FileNotFoundError(f"checkpoint 路径不存在: {p}")

    return {
        "model_state_dict": sd,
        "transformer_dir": None,
        "optimizer_state_dict": None,
        "state": st if isinstance(st, dict) else {},
    }


def readable_timestamp():
    """Generate a sortable timestamp for filenames (no weekday)."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def config_to_serializable(obj: Any) -> Any:
    """配置 / 嵌套 dataclass 等转为仅含基本类型与容器的结构（可 JSON / 安全 pickle）。

    用于写入 checkpoint 的 ``state.pt`` 内 ``config`` 字段，避免把自定义类名写进 pickle，
    以免日后删除/重命名 ``utils.config`` 中的类型后无法 ``torch.load``。
    """
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: config_to_serializable(getattr(obj, k))
            for k in obj.__dataclass_fields__
        }
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: config_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [config_to_serializable(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)):
        return obj
    # 其余类型不再原样写入 checkpoint（易绑定模块路径）；落为字符串便于人工排查
    return str(obj)


def set_sampler_epoch(loader, epoch: int) -> None:
    """Call ``DistributedSampler.set_epoch``（及 ``WeightedMixtureDataset.set_epoch``）在嵌套的 DataLoader 链上。"""
    e = int(epoch)
    cur: Any = loader
    for _ in range(16):
        if cur is None:
            break
        sampler = getattr(cur, "sampler", None)
        if sampler is not None and callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(e)
        dataset = getattr(cur, "dataset", None)
        if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
            dataset.set_epoch(e)
        nxt = getattr(cur, "dataloader", None)
        if nxt is None:
            nxt = getattr(cur, "_dataloader", None)
        cur = nxt


def save_training_state(
    model,
    _optimizer,
    scheduler,
    config,
    checkpoints_dir,
    prefix,
    step,
    save_only_on_rank0=False,
    epoch=None,
    accelerator=None,
):
    """多机多卡聚合 ``state_dict`` 后，写入 **子目录**：``model_state_dict.pt`` + ``state.pt``（与原先一致）。

    ``state.pt`` 含 ``scheduler_state_dict``、``config``（可序列化）、``step``、``epoch``、``timestamp``。
    不保存 optimizer。返回该 **checkpoint 目录**路径（配置里 ``checkpoint`` / ``infer_checkpoint`` 填此目录）。
    """
    if accelerator is not None and getattr(accelerator.state, "deepspeed_plugin", None):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
        accelerator.print("Using DeepSpeed to get state dict")
        state_dict = accelerator.get_state_dict(model)
    elif accelerator is not None:
        unwrapped = unwrap_model_for_non_deepspeed(model)
        accelerator.print("Using unwrapped.state_dict() to get state dict")
        state_dict = unwrapped.state_dict()
    else:
        unwrapped = unwrap_model_for_non_deepspeed(model)
        state_dict = unwrapped.state_dict()

    step_int = int(step) if step is not None else 0
    if epoch is not None:
        ckpt_dir = os.path.join(
            checkpoints_dir, f"{prefix}_epoch_{int(epoch)}_step_{step_int}"
        )
    else:
        ckpt_dir = os.path.join(checkpoints_dir, f"{prefix}_step_{step_int}")

    state_payload = {
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config_to_serializable(config),
        "step": step_int,
        "epoch": int(epoch) if epoch is not None else None,
        "timestamp": readable_timestamp(),
    }

    do_write = True
    if save_only_on_rank0:
        if accelerator is not None:
            do_write = bool(accelerator.is_main_process)
        elif not dist.is_available() or not dist.is_initialized():
            do_write = True
        else:
            do_write = dist.get_rank() == 0

    if do_write:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(ckpt_dir, MODEL_CHECKPOINT))
        torch.save(state_payload, os.path.join(ckpt_dir, STATE))
        msg = f"[Checkpoint] saved {ckpt_dir} ({MODEL_CHECKPOINT}, {STATE})"
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)

    return ckpt_dir

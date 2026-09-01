"""Weights & Biases：实验初始化与指标上报（供 Trainer 复用）。

配置建议写在顶层 ``wandb:`` 映射中（也可用旧式扁平 ``wandb_*`` 键）::

    wandb:
      project: my-proj
      entity: null
      run_name: null   # 默认用 {model}_{timestamp}
      mode: null       # 如 offline
      dir: null        # 本地 run 根目录，默认 ``{save_root}/wandb``
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None

import torch

from utils.utils import cfg_get, config_to_serializable


def init_wandb_from_cfg(cfg_dict: Dict[str, Any], default_run_name: str) -> None:
    """从 ``cfg_dict`` 解析 W&B 参数并 ``init``；本地文件默认写入 ``save_root/wandb``。"""
    wb = cfg_get(cfg_dict, "wandb", None)
    save_root = str(cfg_get(cfg_dict, "save_root", "results"))
    if isinstance(wb, dict):
        project = str(cfg_get(wb, "project", cfg_get(cfg_dict, "wandb_project", "world-model")))
        run_name = str(
            cfg_get(wb, "run_name", cfg_get(wb, "name", cfg_get(cfg_dict, "wandb_run_name", default_run_name)))
        )
        entity = cfg_get(wb, "entity", None)
        if entity is None or entity == "":
            entity = cfg_get(cfg_dict, "wandb_entity", None) or None
        else:
            entity = entity or None
        mode = cfg_get(wb, "mode", None)
        if mode is None or mode == "":
            mode = cfg_get(cfg_dict, "wandb_mode", None) or None
        else:
            mode = mode or None
        wandb_dir = cfg_get(wb, "dir", None)
    else:
        project = str(cfg_get(cfg_dict, "wandb_project", "world-model"))
        run_name = str(cfg_get(cfg_dict, "wandb_run_name", default_run_name))
        entity = cfg_get(cfg_dict, "wandb_entity", None) or None
        mode = cfg_get(cfg_dict, "wandb_mode", None) or None
        wandb_dir = None
    if not wandb_dir:
        wandb_dir = os.path.join(save_root, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    init_wandb(
        project,
        config_to_serializable(cfg_dict),
        run_name=run_name,
        entity=entity,
        mode=mode,
        wandb_dir=wandb_dir,
    )


def init_wandb(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
    wandb_dir: Optional[str] = None,
) -> None:
    if wandb is None:
        print("[wandb] 未安装 wandb，跳过初始化（pip install wandb）", flush=True)
        return
    print("[wandb] 开始使用 Weights & Biases 记录实验与指标", flush=True)
    kwargs: Dict[str, Any] = {
        "project": project_name,
        "name": run_name,
        "config": config,
        "reinit": True,
    }
    if entity:
        kwargs["entity"] = entity
    if mode:
        kwargs["mode"] = mode
    if wandb_dir:
        kwargs["dir"] = wandb_dir
    try:
        print("[wandb] 正在初始化 run（连接 api.wandb.ai 或 offline 目录）…", flush=True)
        wandb.init(**kwargs)
        print(f"[wandb] run started: project={project_name}, name={run_name}", flush=True)
    except Exception as e:
        print(f"[wandb] init failed: {e}", flush=True)


def _run_active() -> bool:
    return wandb is not None and wandb.run is not None


def log_metrics(step: int, metrics: Dict[str, Any], series_label: str = ""):
    if not _run_active():
        return
    prefix = series_label or "default"
    payload = {}
    for k, v in metrics.items():
        val = float(v.item() if hasattr(v, "item") else v)
        payload[f"{prefix}/{k}"] = val
    wandb.log(payload, step=step)


def log_learning_rate(optimizer: torch.optim.Optimizer, step: int):
    if not _run_active():
        return
    payload = {
        f"train/learning_rate_group_{i}": float(pg["lr"])
        for i, pg in enumerate(optimizer.param_groups)
    }
    wandb.log(payload, step=step)


def log_system_metrics(step: int):
    if not torch.cuda.is_available() or not _run_active():
        return
    wandb.log(
        {
            "train/gpu_memory_allocated_gb": float(
                torch.cuda.memory_allocated() / 1024**3
            ),
            "train/gpu_memory_reserved_gb": float(
                torch.cuda.memory_reserved() / 1024**3
            ),
        },
        step=step,
    )


def finish_wandb():
    if not _run_active():
        return
    try:
        wandb.finish()
    except Exception as e:
        print(f"[wandb] finish failed: {e}")

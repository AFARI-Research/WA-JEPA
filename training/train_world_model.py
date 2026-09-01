"""Generic training entrypoint for world-model experiments.

Configs can point ``model.target`` at the active Causal Future-Masked JEPA model
or at any other model class that follows the trainer loss contract.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf

from datasets.data_utils import load_data_and_data_loaders, load_mixed_datasets
from models import build_world_model
from training.trainer import WorldModelTrainer
from utils.utils import (
    cfg_get,
    readable_timestamp,
    resolve_experiment_tracker,
    save_parameter_name_lists,
    unwrap_model_for_non_deepspeed,
    user_from_env,
)
from utils.wandb_utils import finish_wandb, init_wandb_from_cfg


def _merged_loader_config(data_cfg, model_cfg, trainer_cfg):
    """Flatten relevant config sections for dataset builders."""
    if not isinstance(data_cfg, dict):
        return data_cfg
    merged: dict = {}
    for section in (data_cfg, model_cfg, trainer_cfg):
        if isinstance(section, dict):
            merged.update({k: v for k, v in section.items() if isinstance(k, str)})
    return SimpleNamespace(**merged)


def _build_accelerator(cfg: dict) -> Accelerator:
    trainer_cfg = cfg_get(cfg, "trainer", {})
    grad_accum = int(cfg_get(trainer_cfg, "gradient_accumulation_steps", 1))
    ds_json = cfg_get(
        trainer_cfg,
        "deepspeed_config",
        os.environ.get("DEEPSPEED_CONFIG", None),
    )
    ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_json) if ds_json else None
    timeout_s = int(
        os.environ.get(
            "WM_DIST_TIMEOUT_SEC",
            str(cfg_get(trainer_cfg, "dist_timeout_sec", 3600)),
        )
    )

    return Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=str(cfg_get(trainer_cfg, "mixed_precision", "no")),
        deepspeed_plugin=ds_plugin,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_s))],
    )


def _make_save_root(cfg_dict: dict) -> None:
    base_save = str(cfg_get(cfg_dict, "save_root", "results"))
    user = user_from_env()
    name = str(cfg_get(cfg_dict, "name", "default")).strip() or "default"
    exp_name = str(cfg_get(cfg_dict, "experiment_name", "run")).strip() or "run"
    timestamp = readable_timestamp()

    cfg_dict["name"] = name
    cfg_dict["experiment_name"] = exp_name
    parts: list[str] = [base_save]
    if user:
        parts.append(user)
    parts.extend([name, exp_name, timestamp])
    cfg_dict["save_root"] = os.path.join(*parts)


def _init_tracker(cfg_dict: dict, timestamp_name: str, is_main: bool) -> str:
    tracker = resolve_experiment_tracker(cfg_dict)
    if is_main:
        print(f"[train] experiment_tracker={tracker}", flush=True)
    if not is_main or tracker == "none":
        return tracker

    if tracker == "wandb":
        init_wandb_from_cfg(cfg_dict, timestamp_name)
    return tracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/basic/base_training.yaml")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        dot_overrides = [s.lstrip("-") for s in args.overrides if "=" in s]
        if dot_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    accelerator = _build_accelerator(cfg_dict)
    set_seed(int(cfg_get(cfg_dict, "seed", 42)))
    is_main = accelerator.is_main_process

    if cfg_get(cfg_dict, "tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _make_save_root(cfg_dict)
    if is_main:
        os.makedirs(cfg_dict["save_root"], exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg_dict["save_root"], "config.yaml"))

    timestamp_name = f"{cfg_dict['experiment_name']}_{os.path.basename(cfg_dict['save_root'])}"
    tracker = _init_tracker(cfg_dict, timestamp_name, is_main)

    accelerator.print("Building model...")
    model = build_world_model(cfg_dict).to(accelerator.device)

    data_cfg = cfg_get(cfg_dict, "data", {})
    trainer_cfg = cfg_get(cfg_dict, "trainer", {})
    model_cfg = cfg_get(cfg_dict, "model", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Config must define a top-level `data:` mapping.")
    if not isinstance(trainer_cfg, dict):
        trainer_cfg = {}

    batch_size = int(cfg_get(trainer_cfg, "batch_size_per_gpu", 2))
    if "sources" in data_cfg:
        accelerator.print(f"Multi-dataset mode: {list(data_cfg['sources'].keys())}")
        mixed = load_mixed_datasets(
            data_cfg,
            batch_size_per_gpu=batch_size,
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
            seed=int(cfg_get(cfg_dict, "seed", 42)),
        )
        accelerator.print(f"Multi-dataset sampling weights={mixed['source_weights']}")
        train_loader, val_loader = mixed["train_loader"], mixed["val_loader"]
    else:
        merged = _merged_loader_config(data_cfg, model_cfg, trainer_cfg)
        train_data, _, train_loader, val_loader = load_data_and_data_loaders(
            dataset=cfg_get(data_cfg, "dataset", "NUPLAN"),
            batch_size=batch_size,
            config=merged,
        )
        if len(train_data) == 0:
            raise ValueError("Training set is empty")

    trainer = WorldModelTrainer(
        cfg=cfg_dict,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
    )
    trainer.setup()

    ckpt = cfg_get(cfg_dict, "checkpoint", None)
    if ckpt:
        trainer.resume_from(ckpt)

    if is_main:
        unwrapped = unwrap_model_for_non_deepspeed(model)
        trainable_path, frozen_path = save_parameter_name_lists(
            unwrapped,
            cfg_dict["save_root"],
        )
        accelerator.print(f"Saved parameter lists: {trainable_path}, {frozen_path}")

    trainer.train()

    if is_main and tracker == "wandb":
        finish_wandb()


if __name__ == "__main__":
    main()

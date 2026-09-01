"""Unified launcher for NAVSIM v1 and v2 evaluation.

The selected NAVSIM checkout is injected through ``PYTHONPATH`` so all
``import navsim`` calls in worldmodel and NAVSIM resolve to the same devtool.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eval.navsim_v2_eval import (
    DEFAULT_DATA_SPLIT_NAME,
    DEFAULT_TRAIN_TEST_SPLIT,
    NavsimV2EvalConfig,
    build_env,
    build_navsim_v2_command,
    default_original_data_split,
    default_synthetic_data_split,
)


@dataclass
class NavsimEvalConfig:
    navsim_version: str
    navsim_v1_root: Path | None
    navsim_v2_root: Path | None
    worldmodel_config: Path
    checkpoint: Path | None
    openscene_data_root: Path
    metric_cache_path: Path
    output_dir: Path
    train_test_split: str = DEFAULT_TRAIN_TEST_SPLIT
    data_split_name: str = DEFAULT_DATA_SPLIT_NAME
    experiment_name: str = "worldmodel_navsim_eval"
    device: str = "cuda"
    python_executable: str = sys.executable
    dry_run: bool = False
    v1_split_name: str = "navtest"
    original_sensor_path: Path | None = None
    navsim_log_path: Path | None = None
    sensor_blobs_path: Path | None = None
    synthetic_sensor_path: Path | None = None
    synthetic_scenes_path: Path | None = None
    verbose: bool = False
    extra_overrides: list[str] | None = None
    cache_only: bool = False
    one_stage: bool = False


def _env_path(name: str, value: Path | None, *, required: bool = False) -> Path | None:
    if value is not None:
        return value.expanduser()
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser()
    if required:
        raise SystemExit(f"{name} is required.")
    return None


def select_navsim_root(cfg: NavsimEvalConfig) -> Path:
    version = cfg.navsim_version.lower()
    if version == "v1":
        root = cfg.navsim_v1_root
    elif version == "v2":
        root = cfg.navsim_v2_root
    else:
        raise ValueError("navsim_version must be 'v1' or 'v2'")
    if root is None:
        raise ValueError(f"navsim_{version}_root is required for --navsim-version {version}")
    return root


def build_navsim_v1_command(cfg: NavsimEvalConfig, repo_root: Path) -> list[str]:
    navsim_root = select_navsim_root(cfg)
    navsim_log_path = cfg.navsim_log_path or (
        cfg.openscene_data_root / "navsim_logs" / "test"
    )
    sensor_blobs_path = cfg.sensor_blobs_path or (
        cfg.openscene_data_root / "sensor_blobs" / "test"
    )
    if cfg.checkpoint is None:
        raise SystemExit("--checkpoint is required for NAVSIM v1 scoring.")
    command = [
        cfg.python_executable,
        str(navsim_root / "navsim/planning/script/run_pdm_score.py"),
        f"train_test_split={cfg.v1_split_name}",
        f"experiment_name={cfg.experiment_name}",
        f"output_dir={cfg.output_dir}",
        f"navsim_log_path={navsim_log_path}",
        f"sensor_blobs_path={sensor_blobs_path}",
        f"metric_cache_path={cfg.metric_cache_path}",
        "agent._target_=eval.navsim_agent.WorldModelNavsimAgent",
        f"+agent.config_path={cfg.worldmodel_config}",
        f"+agent.checkpoint_path={cfg.checkpoint}",
    ]
    if cfg.extra_overrides:
        command.extend(cfg.extra_overrides)
    return command


def build_navsim_eval_command(cfg: NavsimEvalConfig, repo_root: Path) -> list[str]:
    if cfg.navsim_version.lower() == "v1":
        return build_navsim_v1_command(cfg, repo_root)
    navsim_root = select_navsim_root(cfg)
    command = build_navsim_v2_command(
        NavsimV2EvalConfig(
            navsim_devkit_root=navsim_root,
            worldmodel_config=cfg.worldmodel_config,
            checkpoint=cfg.checkpoint,
            openscene_data_root=cfg.openscene_data_root,
            metric_cache_path=cfg.metric_cache_path,
            output_dir=cfg.output_dir,
            train_test_split=cfg.train_test_split,
            data_split_name=cfg.data_split_name,
            experiment_name=cfg.experiment_name,
            device=cfg.device,
            python_executable=cfg.python_executable,
            original_sensor_path=cfg.original_sensor_path,
            navsim_log_path=cfg.navsim_log_path,
            synthetic_sensor_path=cfg.synthetic_sensor_path,
            synthetic_scenes_path=cfg.synthetic_scenes_path,
            verbose=cfg.verbose,
            one_stage=cfg.one_stage,
        )
    )
    if cfg.extra_overrides:
        command.extend(cfg.extra_overrides)
    return command


def build_metric_cache_command(cfg: NavsimEvalConfig) -> list[str]:
    navsim_root = select_navsim_root(cfg)
    is_v1 = cfg.navsim_version.lower() == "v1"
    split = cfg.v1_split_name if is_v1 else cfg.train_test_split
    original_split = "test" if is_v1 else (
        cfg.data_split_name or default_original_data_split(cfg.train_test_split)
    )
    synthetic_split = default_synthetic_data_split(cfg.train_test_split)
    navsim_log_path = cfg.navsim_log_path or (
        cfg.openscene_data_root / "navsim_logs" / original_split
    )
    synthetic_scenes_path = cfg.synthetic_scenes_path or (
        cfg.openscene_data_root / synthetic_split / "synthetic_scene_pickles"
    )
    command = [
        cfg.python_executable,
        str(navsim_root / "navsim/planning/script/run_metric_caching.py"),
        f"train_test_split={split}",
        f"navsim_log_path={navsim_log_path}",
        f"synthetic_scenes_path={synthetic_scenes_path}",
        f"metric_cache_path={cfg.metric_cache_path}",
    ]
    if cfg.extra_overrides:
        command.extend(cfg.extra_overrides)
    return command


def _parse_extra_overrides(items: list[str] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for item in items:
        out.extend(part for part in item.split() if part)
    return out


def parse_args(argv: Sequence[str] | None = None) -> NavsimEvalConfig:
    parser = argparse.ArgumentParser(description="Run NAVSIM v1 or v2 evaluation.")
    parser.add_argument("--navsim-version", choices=("v1", "v2"), required=True)
    parser.add_argument("--navsim-v1-root", type=Path, default=None)
    parser.add_argument("--navsim-v2-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--openscene-data-root", type=Path, default=None)
    parser.add_argument("--metric-cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/navsim_eval"))
    parser.add_argument("--train-test-split", default=DEFAULT_TRAIN_TEST_SPLIT)
    parser.add_argument("--data-split-name", default=DEFAULT_DATA_SPLIT_NAME)
    parser.add_argument("--experiment-name", default="worldmodel_navsim_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--v1-split-name", default="navtest")
    parser.add_argument("--navsim-log-path", type=Path, default=None)
    parser.add_argument("--sensor-blobs-path", type=Path, default=None)
    parser.add_argument("--original-sensor-path", type=Path, default=None)
    parser.add_argument("--synthetic-sensor-path", type=Path, default=None)
    parser.add_argument("--synthetic-scenes-path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--one-stage",
        action="store_true",
        help="For NAVSIM v2, use run_pdm_score_one_stage.py instead of the two-stage scorer.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override for the devkit scorer. Can be repeated.",
    )
    args = parser.parse_args(argv)

    version = args.navsim_version.lower()
    default_root_env = "NAVSIM_V1_DEVKIT_ROOT" if version == "v1" else "NAVSIM_V2_DEVKIT_ROOT"
    fallback_root = _env_path("NAVSIM_DEVKIT_ROOT", None)
    selected_root = _env_path(default_root_env, None) or fallback_root
    navsim_v1_root = args.navsim_v1_root or (selected_root if version == "v1" else None)
    navsim_v2_root = args.navsim_v2_root or (selected_root if version == "v2" else None)

    if args.metric_cache_path is not None:
        metric_default = args.metric_cache_path.expanduser()
    elif version == "v1" and os.environ.get("NAVSIM_EXP_ROOT"):
        metric_default = Path(os.environ["NAVSIM_EXP_ROOT"]).expanduser() / "metric_cache"
    elif version == "v2" and os.environ.get("NAVSIM_V2_METRIC_CACHE"):
        metric_default = Path(os.environ["NAVSIM_V2_METRIC_CACHE"]).expanduser()
    else:
        metric_default = args.output_dir.expanduser() / "metric_cache"

    return NavsimEvalConfig(
        navsim_version=version,
        navsim_v1_root=navsim_v1_root.expanduser() if navsim_v1_root else None,
        navsim_v2_root=navsim_v2_root.expanduser() if navsim_v2_root else None,
        worldmodel_config=args.config.expanduser(),
        checkpoint=args.checkpoint.expanduser() if args.checkpoint else None,
        openscene_data_root=_env_path("OPENSCENE_DATA_ROOT", args.openscene_data_root, required=True),
        metric_cache_path=metric_default,
        output_dir=args.output_dir.expanduser(),
        train_test_split=args.train_test_split,
        data_split_name=args.data_split_name,
        experiment_name=args.experiment_name,
        device=args.device,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
        v1_split_name=args.v1_split_name,
        original_sensor_path=args.original_sensor_path.expanduser() if args.original_sensor_path else None,
        navsim_log_path=args.navsim_log_path.expanduser() if args.navsim_log_path else None,
        sensor_blobs_path=args.sensor_blobs_path.expanduser() if args.sensor_blobs_path else None,
        synthetic_sensor_path=args.synthetic_sensor_path.expanduser() if args.synthetic_sensor_path else None,
        synthetic_scenes_path=args.synthetic_scenes_path.expanduser() if args.synthetic_scenes_path else None,
        verbose=args.verbose,
        extra_overrides=_parse_extra_overrides(args.override),
        cache_only=args.cache_only,
        one_stage=args.one_stage,
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    navsim_root = select_navsim_root(cfg)
    command = (
        build_metric_cache_command(cfg)
        if cfg.cache_only
        else build_navsim_eval_command(cfg, repo_root)
    )
    env = build_env(repo_root, navsim_root)
    env["NAVSIM_DEVKIT_ROOT"] = str(navsim_root)
    print(f"NAVSIM_DEVKIT_ROOT={navsim_root}", flush=True)
    print(f"PYTHONPATH={env['PYTHONPATH']}", flush=True)
    print(" ".join(command), flush=True)
    if cfg.dry_run:
        return 0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

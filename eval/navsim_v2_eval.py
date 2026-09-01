"""Standalone NAVSIM v2 evaluation launcher.

This module builds the official NAVSIM v2 ``run_pdm_score.py`` command while
pointing Hydra at this repository's ``WorldModelNavsimAgent`` adapter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_TRAIN_TEST_SPLIT = "navhard_two_stage"
DEFAULT_DATA_SPLIT_NAME = ""


def default_original_data_split(train_test_split: str) -> str:
    """Return the OpenScene split backing a NAVSIM train/test split."""

    mapping = {
        "navhard_two_stage": "test",
        "warmup_two_stage": "test",
        "private_test_hard_two_stage": "private_test_hard",
        "navtest": "test",
        "test": "test",
        "mini": "mini",
        "trainval": "trainval",
    }
    return mapping.get(str(train_test_split), str(train_test_split))


def default_synthetic_data_split(train_test_split: str) -> str:
    return str(train_test_split)


@dataclass
class NavsimV2EvalConfig:
    navsim_devkit_root: Path
    worldmodel_config: Path
    checkpoint: Path | None
    openscene_data_root: Path
    metric_cache_path: Path
    output_dir: Path
    train_test_split: str = DEFAULT_TRAIN_TEST_SPLIT
    data_split_name: str = DEFAULT_DATA_SPLIT_NAME
    experiment_name: str = "worldmodel_navsim_v2"
    device: str = "cuda"
    python_executable: str = sys.executable
    original_sensor_path: Path | None = None
    navsim_log_path: Path | None = None
    synthetic_sensor_path: Path | None = None
    synthetic_scenes_path: Path | None = None
    verbose: bool = False
    dry_run: bool = False
    one_stage: bool = False


def _path_override(key: str, value: Path | str) -> str:
    return f"{key}={value}"


def _append_override(key: str, value: Path | str) -> str:
    return f"+{key}={value}"


def _checkpoint_override(checkpoint: Path | None) -> str:
    return (
        "+agent.checkpoint_path=null"
        if checkpoint is None
        else f"+agent.checkpoint_path={checkpoint}"
    )


def build_navsim_v2_command(cfg: NavsimV2EvalConfig) -> list[str]:
    """Build a NAVSIM v2 PDM-score command from explicit paths."""

    script_name = "run_pdm_score_one_stage.py" if cfg.one_stage else "run_pdm_score.py"
    navsim_script = cfg.navsim_devkit_root / "navsim/planning/script" / script_name
    original_data_split = cfg.data_split_name or default_original_data_split(cfg.train_test_split)
    synthetic_data_split = default_synthetic_data_split(cfg.train_test_split)
    navsim_log_path = cfg.navsim_log_path or (
        cfg.openscene_data_root / "navsim_logs" / original_data_split
    )
    original_sensor_path = cfg.original_sensor_path or (
        cfg.openscene_data_root / "sensor_blobs" / original_data_split
    )
    synthetic_sensor_path = cfg.synthetic_sensor_path or (
        cfg.openscene_data_root / synthetic_data_split / "sensor_blobs"
    )
    synthetic_scenes_path = cfg.synthetic_scenes_path or (
        cfg.openscene_data_root / synthetic_data_split / "synthetic_scene_pickles"
    )

    return [
        cfg.python_executable,
        str(navsim_script),
        _path_override("train_test_split", cfg.train_test_split),
        _path_override("experiment_name", cfg.experiment_name),
        _path_override("output_dir", cfg.output_dir),
        _path_override("navsim_log_path", navsim_log_path),
        _path_override("original_sensor_path", original_sensor_path),
        _path_override("synthetic_sensor_path", synthetic_sensor_path),
        _path_override("synthetic_scenes_path", synthetic_scenes_path),
        _path_override("metric_cache_path", cfg.metric_cache_path),
        "agent._target_=eval.navsim_agent.WorldModelNavsimAgent",
        _append_override("agent.config_path", cfg.worldmodel_config),
        _checkpoint_override(cfg.checkpoint),
        _append_override("agent.device", cfg.device),
        "agent.trajectory_sampling._target_=nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
        "agent.trajectory_sampling.time_horizon=4",
        "agent.trajectory_sampling.interval_length=0.5",
        f"verbose={str(cfg.verbose).lower()}",
    ]


def build_env(
    repo_root: Path,
    navsim_devkit_root: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that lets NAVSIM import this repository."""

    env = dict(os.environ if base_env is None else base_env)
    existing = env.get("PYTHONPATH")
    prefix = f"{repo_root}:{navsim_devkit_root}"
    env["PYTHONPATH"] = prefix if not existing else f"{prefix}:{existing}"
    return env


def _env_path(name: str, value: Path | None, *, required: bool = True) -> Path | None:
    if value is not None:
        return value.expanduser()
    env_value = os.environ.get(name)
    if env_value:
        return Path(env_value).expanduser()
    if required:
        raise SystemExit(f"{name} is required; set it or pass the matching CLI flag.")
    return None


def parse_args(argv: Sequence[str] | None = None) -> NavsimV2EvalConfig:
    parser = argparse.ArgumentParser(description="Run standalone NAVSIM v2 EPDMS evaluation.")
    parser.add_argument("--navsim-devkit-root", type=Path, default=None)
    parser.add_argument("--openscene-data-root", type=Path, default=None)
    parser.add_argument("--metric-cache-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True, help="Worldmodel YAML config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Worldmodel checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/navsim_v2_eval"))
    parser.add_argument("--train-test-split", default=DEFAULT_TRAIN_TEST_SPLIT)
    parser.add_argument("--data-split-name", default=DEFAULT_DATA_SPLIT_NAME)
    parser.add_argument("--experiment-name", default="worldmodel_navsim_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--original-sensor-path", type=Path, default=None)
    parser.add_argument("--navsim-log-path", type=Path, default=None)
    parser.add_argument("--synthetic-sensor-path", type=Path, default=None)
    parser.add_argument("--synthetic-scenes-path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    parser.add_argument(
        "--one-stage",
        action="store_true",
        help="Use NAVSIM v2 run_pdm_score_one_stage.py, e.g. for navtest v2-style EPDMS.",
    )
    args = parser.parse_args(argv)

    cfg = NavsimV2EvalConfig(
        navsim_devkit_root=_env_path("NAVSIM_DEVKIT_ROOT", args.navsim_devkit_root),
        worldmodel_config=args.config.expanduser(),
        checkpoint=args.checkpoint.expanduser() if args.checkpoint else None,
        openscene_data_root=_env_path("OPENSCENE_DATA_ROOT", args.openscene_data_root),
        metric_cache_path=_env_path("NAVSIM_V2_METRIC_CACHE", args.metric_cache_path),
        output_dir=args.output_dir.expanduser(),
        train_test_split=args.train_test_split,
        data_split_name=args.data_split_name,
        experiment_name=args.experiment_name,
        device=args.device,
        python_executable=args.python_executable,
        original_sensor_path=args.original_sensor_path.expanduser() if args.original_sensor_path else None,
        navsim_log_path=args.navsim_log_path.expanduser() if args.navsim_log_path else None,
        synthetic_sensor_path=args.synthetic_sensor_path.expanduser() if args.synthetic_sensor_path else None,
        synthetic_scenes_path=args.synthetic_scenes_path.expanduser() if args.synthetic_scenes_path else None,
        verbose=args.verbose,
        dry_run=args.dry_run,
        one_stage=args.one_stage,
    )
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    command = build_navsim_v2_command(cfg)
    repo_root = Path(__file__).resolve().parents[1]
    print(" ".join(command), flush=True)
    if cfg.dry_run:
        return 0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        command,
        env=build_env(repo_root, cfg.navsim_devkit_root),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

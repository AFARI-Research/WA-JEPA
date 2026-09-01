"""Run NAVSIM scoring with a precomputed trajectory cache."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from eval.navsim_v2_eval import build_env


def _path_override(key: str, value: Path | str) -> str:
    return f"{key}={value}"


def _append_override(key: str, value: Path | str) -> str:
    return f"+{key}={value}"


def build_command(args: argparse.Namespace) -> list[str]:
    if args.navsim_version == "v2" or args.one_stage:
        script = "run_pdm_score_one_stage.py"
    else:
        script = "run_pdm_score.py"
    script_path = args.navsim_root / "navsim/planning/script" / script
    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_path} does not exist. navtest scoring needs one-stage scoring; "
            "use a NAVSIM devkit that provides run_pdm_score_one_stage.py or set --no-one-stage "
            "with a two-stage split/cache."
        )
    command = [
        args.python_executable,
        str(script_path),
        "train_test_split=navtest",
        _path_override("experiment_name", args.experiment_name),
        _path_override("output_dir", args.output_dir),
        _path_override("metric_cache_path", args.metric_cache_path),
        "agent._target_=eval.cached_trajectory_navsim_agent.CachedTrajectoryNavsimAgent",
        f"+agent.trajectory_cache_path={args.trajectory_cache_path}",
        "agent.trajectory_sampling._target_=nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
        "agent.trajectory_sampling.time_horizon=4",
        "agent.trajectory_sampling.interval_length=0.5",
    ]
    if args.navsim_version == "v2":
        command.extend(
            [
                _path_override("navsim_log_path", args.openscene_root / "navsim_logs/test"),
                _path_override("original_sensor_path", args.openscene_root / "sensor_blobs/test"),
                "worker=single_machine_thread_pool",
                "worker.max_workers=1",
                "traffic_agents=non_reactive",
                "traffic_agents_policy.non_reactive._target_=navsim.traffic_agents_policies.log_replay_traffic_agents.LogReplayTrafficAgents",
            ]
        )
    else:
        command.extend(
            [
                _path_override("navsim_log_path", args.openscene_root / "navsim_logs/test"),
                _append_override(args.v1_sensor_path_key, args.openscene_root / "sensor_blobs/test"),
            ]
        )
    command.extend(args.override)
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-version", choices=("v1", "v2"), required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--openscene-root", type=Path, required=True)
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--trajectory-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default="worldmodel_cached_trajectory")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--v1-sensor-path-key", default="sensor_blobs_path")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--one-stage", dest="one_stage", action="store_true", default=True)
    parser.add_argument("--no-one-stage", dest="one_stage", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    command = build_command(args)
    env = build_env(repo_root, args.navsim_root)
    env["NAVSIM_DEVKIT_ROOT"] = str(args.navsim_root)
    env["OPENSCENE_DATA_ROOT"] = str(args.openscene_root)
    print(f"NAVSIM_DEVKIT_ROOT={args.navsim_root}", flush=True)
    print(f"PYTHONPATH={env['PYTHONPATH']}", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

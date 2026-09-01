"""Export worldmodel NAVSIM navtest trajectories once for reusable scoring."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

from eval.navsim_v2_eval import build_env


def _scene_filter_path(navsim_root: Path) -> Path:
    return navsim_root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"


def _load_scene_filter(navsim_root: Path) -> tuple[list[str], list[str]]:
    with _scene_filter_path(navsim_root).open() as f:
        data = yaml.safe_load(f)
    log_names = list(data.get("log_names") or [])
    tokens = list(data.get("tokens") or [])
    if not log_names or not tokens:
        raise RuntimeError(f"navtest.yaml must contain log_names and tokens: {_scene_filter_path(navsim_root)}")
    if len(log_names) != len(set(log_names)):
        raise RuntimeError("Duplicate navtest log_names.")
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("Duplicate navtest tokens.")
    return log_names, tokens


def _visible_devices(num_gpus: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(num_gpus)]
    if len(devices) < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs, but only {len(devices)} are visible: {devices}")
    return devices[:num_gpus]


def _shard_logs(log_names: Sequence[str], num_shards: int) -> list[list[str]]:
    return [
        list(log_names[rank * len(log_names) // num_shards : (rank + 1) * len(log_names) // num_shards])
        for rank in range(num_shards)
    ]


def _merge_rank_pickles(rank_paths: Sequence[Path], output_path: Path, expected_tokens: Sequence[str]) -> None:
    trajectories = {}
    for rank_path in rank_paths:
        with open(rank_path, "rb") as f:
            payload = pickle.load(f)
        rank_trajectories = payload.get("trajectories", payload)
        overlap = set(trajectories).intersection(rank_trajectories)
        if overlap:
            raise RuntimeError(f"Duplicate tokens while merging {rank_path}: {sorted(overlap)[:10]}")
        trajectories.update(rank_trajectories)

    expected = set(expected_tokens)
    actual = set(trajectories)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Incomplete trajectory cache: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"first_missing={missing[:10]} first_unexpected={unexpected[:10]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "format": "worldmodel_navsim_trajectory_cache_v1",
                "tokens": list(expected_tokens),
                "trajectories": trajectories,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[navsim-trajectory-cache] wrote {len(trajectories)} trajectories -> {output_path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--openscene-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--rank-output", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run_worker(args: argparse.Namespace) -> int:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from eval.navsim_agent import WorldModelNavsimAgent

    log_names, _ = _load_scene_filter(args.navsim_root)
    log_shards = _shard_logs(log_names, args.num_gpus)
    rank_log_names = log_shards[int(args.rank)]

    scene_filter = instantiate(OmegaConf.load(_scene_filter_path(args.navsim_root)))
    scene_filter.log_names = rank_log_names
    agent = WorldModelNavsimAgent(
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        device="cuda",
        config_overrides=args.override,
    )
    agent.initialize()
    scene_loader = SceneLoader(
        original_sensor_path=args.openscene_root / "sensor_blobs/test",
        data_path=args.openscene_root / "navsim_logs/test",
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    trajectories = {}
    for index, token in enumerate(scene_loader.tokens):
        trajectory = agent.compute_trajectory(scene_loader.get_agent_input_from_token(token))
        trajectories[str(token)] = trajectory.poses.astype("float32")
        if (index + 1) % 100 == 0:
            print(f"[navsim-trajectory-cache] rank={args.rank} {index + 1}/{len(scene_loader.tokens)}", flush=True)

    if args.rank_output is None:
        raise RuntimeError("--rank-output is required in worker mode")
    args.rank_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.rank_output, "wb") as f:
        pickle.dump({"trajectories": trajectories}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[navsim-trajectory-cache] rank={args.rank} wrote {len(trajectories)} -> {args.rank_output}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rank is not None:
        return _run_worker(args)
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    _, expected_tokens = _load_scene_filter(args.navsim_root)
    devices = _visible_devices(args.num_gpus)
    shard_dir = args.output_path.parent / "trajectory_shards"
    rank_paths = [shard_dir / f"rank_{rank:02d}.pkl" for rank in range(args.num_gpus)]

    processes = []
    for rank, (device, rank_path) in enumerate(zip(devices, rank_paths)):
        rank_path.unlink(missing_ok=True)
        command = [
            args.python_executable,
            "-m",
            "eval.navsim_export_trajectory_cache",
            "--navsim-root",
            str(args.navsim_root),
            "--openscene-root",
            str(args.openscene_root),
            "--output-path",
            str(args.output_path),
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--num-gpus",
            str(args.num_gpus),
            "--rank",
            str(rank),
            "--rank-output",
            str(rank_path),
        ]
        for override in args.override:
            command.extend(["--override", override])
        print(f"[navsim-trajectory-cache] rank={rank} gpu={device}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        env = build_env(repo_root, args.navsim_root)
        env["CUDA_VISIBLE_DEVICES"] = device
        log_path = shard_dir / f"rank_{rank:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w")
        processes.append((subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT), log_file, log_path))

    if args.dry_run:
        return 0

    failures = []
    for process, log_file, log_path in processes:
        code = process.wait()
        log_file.close()
        if code != 0:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
            failures.append(f"{log_path}: exit {code}\n{tail}")
    if failures:
        raise RuntimeError("Trajectory export workers failed:\n" + "\n\n".join(failures))

    _merge_rank_pickles(rank_paths, args.output_path, expected_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

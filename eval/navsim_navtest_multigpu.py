"""Multi-GPU launcher for NAVSIM navtest one-stage EPDMS evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

from eval.navsim_v2_eval import NavsimV2EvalConfig, build_env, build_navsim_v2_command


def _visible_devices(num_gpus: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(num_gpus)]
    if len(devices) < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs, but CUDA_VISIBLE_DEVICES exposes "
            f"only {len(devices)}: {devices}"
        )
    return devices[:num_gpus]


def _load_scene_filter(navsim_root: Path) -> tuple[list[str], list[str]]:
    scene_filter_path = (
        navsim_root
        / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    with scene_filter_path.open() as f:
        scene_filter = yaml.safe_load(f)
    log_names = list(scene_filter.get("log_names") or [])
    tokens = list(scene_filter.get("tokens") or [])
    if not log_names:
        raise RuntimeError(f"No log_names found in {scene_filter_path}")
    if len(log_names) != len(set(log_names)):
        raise RuntimeError(f"Duplicate log_names found in {scene_filter_path}")
    if not tokens:
        raise RuntimeError(f"No tokens found in {scene_filter_path}")
    if len(tokens) != len(set(tokens)):
        raise RuntimeError(f"Duplicate tokens found in {scene_filter_path}")
    return log_names, tokens


def _hydra_list(items: Sequence[str]) -> str:
    return json.dumps(list(items), separators=(",", ":"))


def _merge_rank_csvs(
    rank_dirs: Sequence[Path],
    output_dir: Path,
    expected_tokens: Sequence[str],
) -> Path:
    import pandas as pd

    frames = []
    for rank_dir in rank_dirs:
        csv_files = sorted(rank_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(
                f"Expected exactly one result CSV in {rank_dir}, found {csv_files}"
            )
        frame = pd.read_csv(csv_files[0])
        frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
        frame = frame[frame["token"] != "average_all_frames"]
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    duplicated_tokens = merged.loc[merged["token"].duplicated(), "token"].tolist()
    if duplicated_tokens:
        raise RuntimeError(
            f"Duplicate scenario tokens across rank CSVs: {duplicated_tokens[:10]}"
        )
    actual_tokens = set(merged["token"])
    expected_token_set = set(expected_tokens)
    missing_tokens = sorted(expected_token_set - actual_tokens)
    unexpected_tokens = sorted(actual_tokens - expected_token_set)
    if missing_tokens or unexpected_tokens:
        raise RuntimeError(
            "Incomplete NAVSIM navtest merge: "
            f"expected={len(expected_token_set)} actual={len(actual_tokens)} "
            f"missing={len(missing_tokens)} unexpected={len(unexpected_tokens)}; "
            f"first_missing={missing_tokens[:10]} "
            f"first_unexpected={unexpected_tokens[:10]}"
        )
    score_columns = [
        column
        for column in merged.columns
        if column not in {"token", "valid"}
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    average_row = {
        column: float("nan")
        for column in merged.columns
    }
    average_row["token"] = "average_all_frames"
    average_row["valid"] = bool(merged["valid"].all())
    for column in score_columns:
        average_row[column] = float(merged[column].mean(skipna=True))
    merged = pd.concat([merged, pd.DataFrame([average_row])], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{datetime.now():%Y.%m.%d.%H.%M.%S}.csv"
    merged.to_csv(output_path, index=False)
    print(
        f"[navtest-multigpu] merged {len(merged) - 1} scenes | "
        f"score={average_row.get('score', float('nan')):.6f} | {output_path}",
        flush=True,
    )
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--openscene-root", type=Path, required=True)
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--experiment-name", default="worldmodel_navtest_epdms")
    parser.add_argument(
        "--traffic-policy",
        default="navsim.traffic_agents_policies.log_replay_traffic_agents.LogReplayTrafficAgents",
    )
    parser.add_argument(
        "--ranks",
        default="",
        help="Comma-separated local GPU ranks to rerun, for example 3 or 1,3.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip evaluation and merge the existing rank CSV files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    devices = _visible_devices(args.num_gpus)
    log_names, expected_tokens = _load_scene_filter(args.navsim_root)
    log_shards = [
        log_names[start:end]
        for start, end in (
            (
                rank * len(log_names) // args.num_gpus,
                (rank + 1) * len(log_names) // args.num_gpus,
            )
            for rank in range(args.num_gpus)
        )
        if end > start
    ]
    rank_dirs = [args.output_dir / "shards" / f"rank_{rank:02d}" for rank in range(len(log_shards))]
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    selected_ranks = list(range(len(log_shards)))
    if args.ranks:
        try:
            selected_ranks = sorted(
                {int(item.strip()) for item in args.ranks.split(",") if item.strip()}
            )
        except ValueError as error:
            raise ValueError(f"Invalid --ranks value: {args.ranks}") from error
        invalid_ranks = [
            rank for rank in selected_ranks if rank < 0 or rank >= len(log_shards)
        ]
        if not selected_ranks or invalid_ranks:
            raise ValueError(
                f"--ranks must select values in [0, {len(log_shards) - 1}], "
                f"got {selected_ranks}"
            )

    if args.merge_only:
        _merge_rank_csvs(rank_dirs, args.output_dir, expected_tokens)
        return 0

    for rank in selected_ranks:
        device = devices[rank]
        rank_log_names = log_shards[rank]
        rank_dir = rank_dirs[rank]
        rank_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in [*rank_dir.glob("*.csv"), rank_dir / "worker.log"]:
            stale_path.unlink(missing_ok=True)
        cfg = NavsimV2EvalConfig(
            navsim_devkit_root=args.navsim_root,
            worldmodel_config=args.config,
            checkpoint=args.checkpoint,
            openscene_data_root=args.openscene_root,
            metric_cache_path=args.metric_cache_path,
            output_dir=rank_dir,
            train_test_split="navtest",
            data_split_name="test",
            experiment_name=f"{args.experiment_name}_rank{rank}",
            device="cuda",
            python_executable=args.python_executable,
            one_stage=True,
        )
        command = build_navsim_v2_command(cfg)
        command.extend(
            [
                "worker=single_machine_thread_pool",
                "worker.max_workers=1",
                "traffic_agents=non_reactive",
                f"traffic_agents_policy.non_reactive._target_={args.traffic_policy}",
                f"train_test_split.scene_filter.log_names={_hydra_list(rank_log_names)}",
            ]
        )
        print(
            f"[navtest-multigpu] rank={rank} gpu={device} logs={len(rank_log_names)}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue

        env = build_env(repo_root, args.navsim_root)
        env["CUDA_VISIBLE_DEVICES"] = device
        log_file = (rank_dir / "worker.log").open("w")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_file, rank_dir))

    if args.dry_run:
        return 0

    failures = []
    for process, log_file, rank_dir in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failures.append((rank_dir, return_code))
    if failures:
        details = []
        for rank_dir, return_code in failures:
            log_path = rank_dir / "worker.log"
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            details.append(f"{rank_dir}: exit {return_code}\n{tail}")
        raise RuntimeError("NAVSIM GPU shards failed:\n" + "\n\n".join(details))

    _merge_rank_csvs(rank_dirs, args.output_dir, expected_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

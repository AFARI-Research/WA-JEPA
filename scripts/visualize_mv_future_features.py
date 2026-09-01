"""Visualize multi-view future scene tokens without leaking GT into inference.

For each validation sample, this script writes:

    output/sample_xxx/<camera>/future_step_YY.png
    output/sample_xxx/multiview_future.mp4

Each PNG has four rows:
    1. EMA teacher target future tokens, PCA-to-RGB
    2. Inference-time predicted future tokens, PCA-to-RGB
    3. Raw GT future image for the first frame covered by this tubelet
    4. Raw GT future image for the second frame covered by this tubelet

The prediction pass uses only history images and deployment-time driving inputs.
GT future images are used only after prediction to compute teacher/reference
features and to render the raw image rows.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except ModuleNotFoundError:  # allow --help in lightweight utility environments
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _merged_loader_config(data_cfg: Any, model_cfg: Any, trainer_cfg: Any) -> SimpleNamespace:
    merged: dict[str, Any] = {}
    for section in (data_cfg, model_cfg, trainer_cfg):
        if isinstance(section, dict):
            merged.update({k: v for k, v in section.items() if isinstance(k, str)})
    return SimpleNamespace(**merged)


def _sanitize(value: Any, fallback: str) -> str:
    text = str(value if value not in (None, "") else fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def _field_at(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        item = value[index]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _ensure_multiview(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:
        return images.unsqueeze(2)
    if images.ndim == 6:
        return images
    raise ValueError(f"Expected [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(images.shape)}")


def _image_to_uint8(image: torch.Tensor) -> np.ndarray:
    img = image.detach().to("cpu", dtype=torch.float32)
    if img.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {tuple(img.shape)}")
    if float(img.min().item()) < -0.05:
        img = img.mul(0.5).add(0.5)
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).contiguous().numpy() * 255.0).round().astype(np.uint8)


def _rgb_feature_to_uint8(feature_rgb: torch.Tensor) -> np.ndarray:
    arr = feature_rgb.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _fit_sample_pca_rgb(pred_sample: torch.Tensor, target_sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_sample.detach().to("cpu", dtype=torch.float32)
    target = target_sample.detach().to("cpu", dtype=torch.float32)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    pred_flat = pred.reshape(-1, pred.size(-1))
    target_flat = target.reshape(-1, target.size(-1))
    mean = pred_flat.mean(dim=0, keepdim=True)
    centered = pred_flat - mean
    cov = centered.T @ centered / max(centered.size(0) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    basis = eigvecs[:, -3:]
    pred_projected = torch.matmul(pred_flat - mean, basis)
    target_projected = torch.matmul(target_flat - mean, basis)
    lo = torch.quantile(pred_projected, 0.01, dim=0)
    hi = torch.quantile(pred_projected, 0.99, dim=0)
    denom = (hi - lo).clamp_min(1e-6)
    pred_rgb = ((pred_projected - lo) / denom).clamp(0.0, 1.0).reshape(*pred.shape[:-1], 3)
    target_rgb = ((target_projected - lo) / denom).clamp(0.0, 1.0).reshape(*target.shape[:-1], 3)
    return pred_rgb, target_rgb


def _make_step_figure(
    *,
    teacher_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    title: str,
) -> Image.Image:
    raw_a_img = Image.fromarray(_image_to_uint8(raw_a), mode="RGB")
    raw_b_img = Image.fromarray(_image_to_uint8(raw_b), mode="RGB")
    target_size = raw_a_img.size
    teacher_img = Image.fromarray(_rgb_feature_to_uint8(teacher_rgb), mode="RGB").resize(
        target_size,
        resample=Image.Resampling.NEAREST,
    )
    pred_img = Image.fromarray(_rgb_feature_to_uint8(pred_rgb), mode="RGB").resize(
        target_size,
        resample=Image.Resampling.NEAREST,
    )
    font = ImageFont.load_default()
    title_h = 30
    label_h = 24
    rows = [
        ("teacher target tokens (PCA RGB)", teacher_img),
        ("predicted future tokens, inference path (PCA RGB)", pred_img),
        ("raw future frame A", raw_a_img),
        ("raw future frame B", raw_b_img),
    ]
    width = target_size[0]
    height = title_h + sum(label_h + img.size[1] for _label, img in rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)
    y = title_h
    for label, img in rows:
        draw.rectangle((0, y, width, y + label_h), fill=(245, 245, 245))
        draw.text((8, y + 6), label, fill=(20, 20, 20), font=font)
        y += label_h
        canvas.paste(img, (0, y))
        y += img.size[1]
    return canvas


def _save_step_figure(
    out_path: Path,
    *,
    teacher_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    title: str,
) -> Image.Image:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = _make_step_figure(
        teacher_rgb=teacher_rgb,
        pred_rgb=pred_rgb,
        raw_a=raw_a,
        raw_b=raw_b,
        title=title,
    )
    canvas.save(out_path)
    return canvas


def _camera_video_slots(camera_names: list[str]) -> tuple[int, int, list[int | None]]:
    lowered = [str(name).lower() for name in camera_names]
    if {"cam_l0", "cam_f0", "cam_r0", "cam_b0"}.issubset(set(lowered)) and len(camera_names) <= 4:
        slots: list[int | None] = [None] * 6
        for cam_i, name in enumerate(lowered):
            if name == "cam_l0":
                slots[0] = cam_i
            elif name == "cam_f0":
                slots[1] = cam_i
            elif name == "cam_r0":
                slots[2] = cam_i
            elif name == "cam_b0":
                slots[4] = cam_i
        return 2, 3, slots

    cols = int(np.ceil(np.sqrt(max(len(camera_names), 1))))
    rows = int(np.ceil(max(len(camera_names), 1) / cols))
    slots = list(range(len(camera_names))) + [None] * (rows * cols - len(camera_names))
    return rows, cols, slots


def _resize_tile(tile: Image.Image, tile_width: int | None) -> Image.Image:
    if tile_width is None or tile_width <= 0 or tile.size[0] == tile_width:
        return tile
    height = max(1, int(round(tile.size[1] * float(tile_width) / float(tile.size[0]))))
    return tile.resize((int(tile_width), height), resample=Image.Resampling.BILINEAR)


def _make_multiview_video_frame(
    *,
    step_tiles: list[Image.Image],
    camera_names: list[str],
    sample_name: str,
    step_i: int,
    tile_width: int | None,
) -> Image.Image:
    tiles = [_resize_tile(tile, tile_width) for tile in step_tiles]
    tile_w = max(tile.size[0] for tile in tiles)
    tile_h = max(tile.size[1] for tile in tiles)
    rows, cols, slots = _camera_video_slots(camera_names)
    font = ImageFont.load_default()
    gap = 8
    title_h = 34
    width = cols * tile_w + (cols + 1) * gap
    height = title_h + rows * tile_h + (rows + 1) * gap
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f"{sample_name} | future tubelet {step_i:02d}", fill=(20, 20, 20), font=font)
    for slot_i, cam_i in enumerate(slots):
        row = slot_i // cols
        col = slot_i % cols
        x = gap + col * (tile_w + gap)
        y = title_h + gap + row * (tile_h + gap)
        draw.rectangle((x, y, x + tile_w, y + tile_h), fill=(232, 232, 232))
        if cam_i is None:
            continue
        tile = tiles[cam_i]
        paste_x = x + (tile_w - tile.size[0]) // 2
        paste_y = y + (tile_h - tile.size[1]) // 2
        canvas.paste(tile, (paste_x, paste_y))
        label = camera_names[cam_i] if cam_i < len(camera_names) else f"camera_{cam_i}"
        draw.rectangle((paste_x, paste_y, paste_x + min(tile.size[0], 96), paste_y + 18), fill=(255, 255, 255))
        draw.text((paste_x + 4, paste_y + 4), label, fill=(20, 20, 20), font=font)
    return canvas


def _write_mp4(out_path: Path, frames: list[Image.Image], *, fps: float, codec: str) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0].convert("RGB")
    width, height = first.size

    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None

    if cv2 is not None:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*str(codec)[:4].ljust(4)),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {out_path}")
        try:
            for frame in frames:
                if frame.size != (width, height):
                    frame = frame.resize((width, height), resample=Image.Resampling.BILINEAR)
                arr = np.asarray(frame.convert("RGB"))
                writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Writing MP4 requires opencv-python/cv2 or imageio in the runtime."
        ) from exc

    writer = imageio.get_writer(str(out_path), fps=float(fps))
    try:
        for frame in frames:
            if frame.size != (width, height):
                frame = frame.resize((width, height), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(frame.convert("RGB"))
            writer.append_data(arr)
    finally:
        writer.close()


def _build_val_loader(cfg_dict: dict[str, Any], batch_size: int, num_workers: int):
    from datasets.data_utils import load_data_and_data_loaders, load_mixed_datasets
    from utils.utils import cfg_get

    data_cfg = cfg_get(cfg_dict, "data", {})
    model_cfg = cfg_get(cfg_dict, "model", {})
    trainer_cfg = dict(cfg_get(cfg_dict, "trainer", {}) or {})
    trainer_cfg["batch_size_per_gpu"] = batch_size
    trainer_cfg["dataloader_num_workers"] = num_workers
    trainer_cfg["val_dataloader_num_workers"] = num_workers
    if isinstance(data_cfg, dict) and "sources" in data_cfg:
        mixed = load_mixed_datasets(
            data_cfg,
            batch_size_per_gpu=batch_size,
            num_processes=1,
            process_index=0,
            seed=int(cfg_get(cfg_dict, "seed", 42)),
        )
        return mixed["val_loader"]
    merged = _merged_loader_config(data_cfg, model_cfg, trainer_cfg)
    _train_data, _val_data, _train_loader, val_loader = load_data_and_data_loaders(
        dataset=cfg_get(data_cfg, "dataset", "NUPLAN"),
        batch_size=batch_size,
        config=merged,
    )
    return val_loader


def _maybe_make_random_loader(loader: Any, *, batch_size: int, num_workers: int, seed: int | None) -> Any:
    if seed is None:
        return loader
    if not hasattr(loader, "dataset"):
        raise TypeError("Random sampling requires a DataLoader with a dataset attribute")
    from torch.utils.data import DataLoader, Subset
    from datasets.data_utils import nuplan_collate_fn

    dataset = loader.dataset
    total = len(dataset)
    order = list(range(total))
    rng = random.Random(int(seed))
    rng.shuffle(order)
    subset = Subset(dataset, order)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=nuplan_collate_fn,
    )


def _load_cfg(path: str, overrides: list[str]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    dot_overrides = [item.lstrip("-") for item in overrides if "=" in item]
    if dot_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="feature_vis/mv_future_features")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--no-video", action="store_true", help="Disable per-sample multiview MP4 export.")
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--video-tile-width", type=int, default=384)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--strict", action="store_true", help="Strict checkpoint load.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if torch is None:
        raise ModuleNotFoundError("This script requires torch. Run it in the training environment.")
    from models import build_world_model
    from training.checkpoint import load_checkpoint
    from utils.utils import cfg_get

    cfg_dict = _load_cfg(args.config, args.overrides)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = build_world_model(cfg_dict).to(device)
    checkpoint = args.checkpoint or cfg_get(cfg_dict, "checkpoint", None)
    if checkpoint:
        load_checkpoint(
            checkpoint,
            model,
            map_location=device,
            strict=bool(args.strict),
            write_key_reports=False,
        )
    model.eval()

    val_loader = _build_val_loader(cfg_dict, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = _maybe_make_random_loader(
        val_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.random_seed,
    )
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(val_loader):
            if saved >= args.num_samples:
                break
            if "future_images" not in batch:
                raise KeyError("Batch must contain future_images for teacher/reference visualization")
            result = model.predict_future_scene_for_visualization(batch, include_teacher=True)
            pred_scene = result["pred_scene"].detach().to("cpu", dtype=torch.float32)
            target_scene = result["target_scene"].detach().to("cpu", dtype=torch.float32)
            future_images = _ensure_multiview(batch["future_images"]).detach().to("cpu", dtype=torch.float32)

            bsz = pred_scene.size(0)
            num_cameras = int(result["num_cameras"])
            future_steps = int(result["future_steps"])
            tokens_per_step = int(result["tokens_per_step"])
            grid_h = int(result["grid_h"])
            grid_w = int(result["grid_w"])
            tubelet_size = int(result["tubelet_size"])
            camera_names = list(result["camera_names"])
            pred_view = pred_scene.reshape(bsz, num_cameras, future_steps, tokens_per_step, -1)
            target_view = target_scene.reshape(bsz, num_cameras, future_steps, tokens_per_step, -1)

            for sample_i in range(bsz):
                if saved >= args.num_samples:
                    break
                sample_index = _field_at(batch, "index", sample_i)
                scene_token = _field_at(batch, "scene_token", sample_i)
                sample_name = _sanitize(
                    scene_token,
                    f"sample_{saved:06d}" if sample_index is None else f"index_{sample_index}",
                )
                sample_dir = out_root / sample_name
                pred_rgb, target_rgb = _fit_sample_pca_rgb(pred_view[sample_i], target_view[sample_i])
                pred_rgb = pred_rgb.reshape(num_cameras, future_steps, grid_h, grid_w, 3)
                target_rgb = target_rgb.reshape(num_cameras, future_steps, grid_h, grid_w, 3)
                video_step_tiles: list[list[Image.Image]] = [[] for _ in range(future_steps)]

                for cam_i in range(num_cameras):
                    cam_name = camera_names[cam_i] if cam_i < len(camera_names) else f"camera_{cam_i}"
                    cam_dir = sample_dir / _sanitize(cam_name, f"camera_{cam_i}")
                    for step_i in range(future_steps):
                        frame_a = step_i * tubelet_size
                        frame_b = min(frame_a + 1, future_images.size(1) - 1)
                        title = (
                            f"{sample_name} | {cam_name} | future tubelet {step_i:02d} "
                            f"(frames {frame_a},{frame_b})"
                        )
                        step_figure = _save_step_figure(
                            cam_dir / f"future_step_{step_i:02d}.png",
                            teacher_rgb=target_rgb[cam_i, step_i],
                            pred_rgb=pred_rgb[cam_i, step_i],
                            raw_a=future_images[sample_i, frame_a, cam_i],
                            raw_b=future_images[sample_i, frame_b, cam_i],
                            title=title,
                        )
                        video_step_tiles[step_i].append(step_figure)
                if not args.no_video:
                    video_frames = [
                        _make_multiview_video_frame(
                            step_tiles=video_step_tiles[step_i],
                            camera_names=camera_names,
                            sample_name=sample_name,
                            step_i=step_i,
                            tile_width=args.video_tile_width,
                        )
                        for step_i in range(future_steps)
                    ]
                    _write_mp4(
                        sample_dir / "multiview_future.mp4",
                        video_frames,
                        fps=args.video_fps,
                        codec=args.video_codec,
                    )
                saved += 1
                print(f"[visualize] saved {sample_dir}", flush=True)

    print(f"[visualize] done: saved {saved} samples to {out_root}", flush=True)


if __name__ == "__main__":
    main()

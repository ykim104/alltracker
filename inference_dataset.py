#!/usr/bin/env python3
"""
Batch AllTracker point-track visualization for LeRobot-style camera folders.

Walks ``--dataset-root`` and mirrors:

  ``**/observation.images.image/*.mp4``  ->  ``**/observation.points.image/*.mp4``
  ``**/observation.images.wrist_image/*.mp4``  ->  ``**/observation.points.wrist_image/*.mp4``

Run from the alltracker repository root so imports resolve::

  cd /path/to/alltracker
  python inference_dataset.py --dataset-root /path/to/FastWAM/data/libero_mujoco3.3.2/LIBERO-fastwam

Default output for LeRobot / FastWAM: **moving points on a black background** (``bkg_opacity=0``,
``min_motion_px=1``). Use ``--rgb-background`` for dimmed-RGB demo-style overlays.

Uses the same preprocessing and forward pass as ``demo.py`` (logic duplicated here on purpose
so ``demo.py`` stays a single self-contained script).

Input MP4s are often AV1 (LeRobot). OpenCV may spam HW decode errors; this script defaults to
``ffmpeg`` software decode when ``ffprobe``/``ffmpeg`` are on PATH (see ``--video-backend``).
"""

from __future__ import annotations

import os

# OpenCV must not use broken HW AV1 paths (common on Linux). Set before first ``import cv2``
# (including via ``demo``).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import utils.basic
import utils.improc
from demo import draw_pts_gpu
from nets.alltracker import Net

import cv2  # noqa: E402  (after ``demo`` so OPENCV_* env applies to first cv2 import in ``demo``)

# LIBERO data pairs
DIR_PAIRS = (
    ("observation.images.image", "observation.points.image"),
    ("observation.images.wrist_image", "observation.points.wrist_image"),
)

# Robotwin data pairs
DIR_PAIRS = (
    ("observation.images.cam_high", "observation.points.cam_high"),
    ("observation.images.cam_left_wrist", "observation.points.cam_left_wrist"),
    ("observation.images.cam_right_wrist", "observation.points.cam_right_wrist"),
)


def _ffprobe_size_fps(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError("ffprobe failed for %s: %s" % (path, out.stderr.strip()))
    j = json.loads(out.stdout)
    if not j.get("streams"):
        raise RuntimeError("ffprobe: no video stream in %s" % path)
    s = j["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    fr = s.get("r_frame_rate") or s.get("avg_frame_rate") or "30/1"
    num, den = fr.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    return w, h, fps


def _read_mp4_rgb_frames_ffmpeg(path: Path, max_frames: int) -> tuple[list[np.ndarray], int]:
    """Decode full clip to RGB uint8 HWC via ffmpeg (software decode; works for AV1)."""
    w, h, fps = _ffprobe_size_fps(path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-hwaccel",
        "none",
        "-i",
        str(path),
    ]
    if max_frames:
        cmd += ["-vframes", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg decode failed for %s (exit %s): %s"
            % (path, proc.returncode, proc.stderr.decode("utf-8", errors="replace")[:500])
        )
    raw = proc.stdout
    stride = w * h * 3
    if len(raw) % stride != 0:
        raise RuntimeError("ffmpeg output size mismatch for %s" % path)
    n = len(raw) // stride
    frames = []
    for i in range(n):
        chunk = raw[i * stride : (i + 1) * stride]
        frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3).copy())
    return frames, int(round(fps))


def _read_mp4_rgb_frames_pyav(path: Path, max_frames: int) -> tuple[list[np.ndarray], int]:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate
        fps = float(rate) if rate is not None and rate > 0 else 30.0
        for i, frame in enumerate(container.decode(stream)):
            if max_frames and i >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError("No frames in %s" % path)
    return frames, int(round(fps))


def _read_mp4_rgb_frames(path: Path, max_frames: int, backend: str) -> tuple[list[np.ndarray], int]:
    """Return list of RGB uint8 (H,W,3) and nominal FPS."""
    if backend == "opencv":
        from demo import read_mp4

        rgbs, framerate = read_mp4(str(path))
        if not rgbs:
            raise ValueError("No frames in %s" % path)
        if max_frames:
            rgbs = rgbs[:max_frames]
        return rgbs, framerate
    if backend == "pyav":
        return _read_mp4_rgb_frames_pyav(path, max_frames)
    if backend == "ffmpeg":
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("ffmpeg backend requires ffmpeg and ffprobe on PATH")
        return _read_mp4_rgb_frames_ffmpeg(path, max_frames)

    # auto
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        try:
            return _read_mp4_rgb_frames_ffmpeg(path, max_frames)
        except Exception as e:
            print("  ffmpeg read failed (%s), trying PyAV/OpenCV..." % e)
    try:
        return _read_mp4_rgb_frames_pyav(path, max_frames)
    except ImportError:
        print("  PyAV not installed, trying OpenCV...")
    except Exception as e:
        print("  PyAV read failed (%s), trying OpenCV..." % e)
    from demo import read_mp4

    rgbs, framerate = read_mp4(str(path))
    if not rgbs:
        raise ValueError("No frames in %s" % path)
    if max_frames:
        rgbs = rgbs[:max_frames]
    return rgbs, framerate


def _load_rgb_tensor(
    mp4_path: Path, image_size: int, max_frames: int, video_backend: str
) -> tuple[torch.Tensor, int]:
    rgbs, framerate = _read_mp4_rgb_frames(mp4_path, max_frames, video_backend)
    if not rgbs:
        raise ValueError("No frames in %s" % mp4_path)
    h, w = rgbs[0].shape[:2]
    if max_frames:
        rgbs = rgbs[:max_frames]
    scale = min(int(image_size) / h, int(image_size) / w)
    h, w = int(h * scale), int(w * scale)
    h, w = h // 8 * 8, w // 8 * 8
    rgbs = [cv2.resize(rgb, dsize=(w, h), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    tensors = [torch.from_numpy(rgb).permute(2, 0, 1) for rgb in rgbs]
    return torch.stack(tensors, dim=0).unsqueeze(0).float(), framerate


@torch.no_grad()
def _infer_point_frames_rgb_batched(
    rgbs: torch.Tensor,
    lengths: list[int],
    model: torch.nn.Module,
    device: str,
    query_frame: int,
    inference_iters: int,
    rate: int,
    conf_thr: float,
    min_motion_px: float,
    bkg_opacity: float,
) -> list[np.ndarray]:
    """Run ONE forward pass over a padded batch of ``b`` videos.

    ``rgbs`` is ``(b, t, 3, h, w)`` where ``t`` is the (padded) max length and every item shares
    ``(h, w)``. ``lengths[i]`` is the real frame count of video ``i`` (padding is the repeated last
    frame). Returns a list of ``b`` uint8 ``(Ti, H, W, 3)`` arrays, each cropped to its real length.
    """
    b, t, c, h, w = rgbs.shape
    assert c == 3
    rgbs = rgbs.to(device, non_blocking=True)

    grid_xy = utils.basic.gridcloud2d(1, h, w, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, h, w)

    t0 = time.time()
    flows_e, visconf_maps_e, _, _ = model.forward_sliding(
        rgbs[:, query_frame:],
        iters=inference_iters,
        sw=None,
        is_training=False,
    )
    traj_maps_e = flows_e.to(device) + grid_xy
    visconf_maps_e = visconf_maps_e.to(device)
    if query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = model.forward_sliding(
            rgbs[:, : query_frame + 1].flip([1]),
            iters=inference_iters,
            sw=None,
            is_training=False,
        )
        backward_traj_maps_e = backward_flows_e.to(device) + grid_xy
        backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1]
        backward_visconf_maps_e = backward_visconf_maps_e.to(device).flip([1])[:, :-1]
        traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1)
        visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1)

    outs: list[np.ndarray] = []
    for bi in range(b):
        ti = int(lengths[bi])
        tm = traj_maps_e[bi : bi + 1, :ti]
        vc = visconf_maps_e[bi : bi + 1, :ti]
        trajs_e = tm[:, :, :, ::rate, ::rate].reshape(1, ti, 2, -1).permute(0, 1, 3, 2)
        visconfs_e = vc[:, :, :, ::rate, ::rate].reshape(1, ti, 2, -1).permute(0, 1, 3, 2)

        xy0 = trajs_e[0, 0].cpu().numpy()
        colors = utils.improc.get_2d_colors(xy0, h, w)

        vis_draw = visconfs_e[0, :, :, 1] > conf_thr
        if min_motion_px > 0:
            q = min(max(query_frame, 0), ti - 1)
            ref = trajs_e[0, q : q + 1]
            max_disp = (trajs_e[0] - ref).norm(dim=-1).max(dim=0).values
            vis_draw = vis_draw & (max_disp >= min_motion_px).unsqueeze(0)

        frames = draw_pts_gpu(
            rgbs[bi, :ti],
            trajs_e[0],
            vis_draw,
            colors,
            rate=rate,
            bkg_opacity=bkg_opacity,
        )
        outs.append(frames)

    elapsed = time.time() - t0
    print(
        "  batch forward+draw: %.2fs  (b=%d, padded_T=%d, lengths=%s)"
        % (elapsed, b, t, [int(x) for x in lengths])
    )
    return outs


def _collate_batch(
    items: list[tuple[Path, Path, torch.Tensor, int]],
) -> tuple[torch.Tensor, list[int]]:
    """Stack same-resolution clips into ``(b, maxT, c, h, w)``; pad short clips with last frame."""
    lengths = [it[2].shape[1] for it in items]
    max_t = max(lengths)
    _, _, c, h, w = items[0][2].shape
    batch = torch.zeros((len(items), max_t, c, h, w), dtype=items[0][2].dtype)
    for i, it in enumerate(items):
        clip = it[2][0]  # (T, c, h, w)
        ti = clip.shape[0]
        batch[i, :ti] = clip
        if ti < max_t:
            batch[i, ti:] = clip[ti - 1 : ti]  # repeat last frame as padding
    return batch, lengths


def _decode_pairs(
    pairs: list[tuple[Path, Path]],
    image_size: int,
    max_frames: int,
    backend: str,
    workers: int,
    lookahead: int,
):
    """Generator that decodes upcoming videos in background threads (overlaps ffmpeg with GPU).

    Yields ``(src, dst, rgbs(1,T,c,h,w) float cpu, fps)`` in input order. ffmpeg/PyAV decode runs
    in subprocesses/threads that release the GIL, so the GPU stays fed while clips are decoded.
    """
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    futs: deque = deque()
    it = iter(pairs)

    def submit_next() -> bool:
        try:
            src, dst = next(it)
        except StopIteration:
            return False
        futs.append((src, dst, ex.submit(_load_rgb_tensor, src, image_size, max_frames, backend)))
        return True

    try:
        for _ in range(max(1, lookahead)):
            if not submit_next():
                break
        while futs:
            src, dst, fut = futs.popleft()
            submit_next()
            try:
                rgbs, fps = fut.result()
            except Exception as e:  # decode failure: skip this clip, keep going
                print("  ERROR decoding %s: %s" % (src, e))
                continue
            yield src, dst, rgbs, fps
    finally:
        ex.shutdown(wait=True)


def _probe_native_meta(path: Path) -> tuple[int, int, int]:
    """Return (width, height, approx_frame_count) for grouping/sorting. Best-effort; 0s on failure."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,duration,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        s = json.loads(out.stdout)["streams"][0]
        w, h = int(s["width"]), int(s["height"])
        nf = s.get("nb_frames")
        if nf and str(nf).isdigit() and int(nf) > 0:
            n = int(nf)
        else:
            dur = float(s.get("duration") or 0.0)
            num, den = (s.get("r_frame_rate") or "30/1").split("/")
            fps = float(num) / float(den) if float(den) else 30.0
            n = int(dur * fps)
        return w, h, n
    except Exception:
        return 0, 0, 0


def _output_is_complete(out_path: Path) -> bool:
    """True if ``out_path`` looks like a finished MP4 (safe to skip on resume)."""
    if not out_path.is_file() or out_path.stat().st_size < 512:
        return False
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,width,height,duration",
            "-of",
            "json",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if out.returncode != 0:
        return False
    try:
        streams = json.loads(out.stdout).get("streams") or []
        if not streams:
            return False
        s = streams[0]
        if int(s.get("width") or 0) <= 0 or int(s.get("height") or 0) <= 0:
            return False
        return float(s.get("duration") or 0.0) > 0.01
    except Exception:
        return False


def _write_mp4(frames_thwc: np.ndarray, out_path: Path, fps: int) -> None:
    """Write MP4 atomically (``.tmp`` then rename) for safe preemptible resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tlen, h, w, c = frames_thwc.shape
    assert c == 3

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            "%dx%d" % (w, h),
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(tmp_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for ti in range(tlen):
                proc.stdin.write(frames_thwc[ti].astype(np.uint8, copy=False).tobytes())
        finally:
            proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        proc.wait()
        if proc.returncode == 0:
            os.replace(tmp_path, out_path)
            return
        if tmp_path.exists():
            tmp_path.unlink()
        print(
            "  ffmpeg H.264 encode failed (exit %s), falling back to OpenCV: %s"
            % (proc.returncode, err.decode("utf-8", errors="replace")[:300])
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open %s" % tmp_path)
    for ti in range(tlen):
        bgr = cv2.cvtColor(frames_thwc[ti], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    os.replace(tmp_path, out_path)


def _load_weights(model: torch.nn.Module, ckpt_init: str, tiny: bool, device: str) -> None:
    import utils.saveload

    if ckpt_init:
        utils.saveload.load(
            None,
            ckpt_init,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        print("Loaded weights from", ckpt_init)
    else:
        if tiny:
            url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
        else:
            url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=True)
        print("Loaded weights from", url)

    model.to(device)
    for _, p in model.named_parameters():
        p.requires_grad = False
    model.eval()


def _iter_episode_mp4s(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Return sorted (input_mp4, output_mp4) paths (global order is stable for sharding)."""
    out: list[tuple[Path, Path]] = []
    for in_leaf, out_leaf in DIR_PAIRS:
        for in_dir in dataset_root.rglob(in_leaf):
            if not in_dir.is_dir() or in_dir.name != in_leaf:
                continue
            out_dir = in_dir.parent / out_leaf
            for mp4 in sorted(in_dir.glob("*.mp4")):
                out.append((mp4, out_dir / mp4.name))
    out.sort(key=lambda sd: str(sd[0].resolve()))
    return out


def _select_pairs(
    dataset_root: Path,
    num_shards: int,
    shard_id: int,
    skip_existing: bool,
    limit: int,
) -> list[tuple[Path, Path]]:
    pairs = _iter_episode_mp4s(dataset_root)
    if num_shards > 1:
        pairs = pairs[shard_id::num_shards]
    if skip_existing:
        before = len(pairs)
        pairs = [(s, d) for (s, d) in pairs if not _output_is_complete(d)]
        print("skip-existing: %d/%d remain on this shard." % (len(pairs), before))
    if limit:
        pairs = pairs[:limit]
    return pairs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root to search (e.g. .../LIBERO-fastwam or .../libero_object_no_noops_lerobot)",
    )
    p.add_argument("--ckpt-init", type=str, default="", help="Optional checkpoint; default downloads HF weights")
    p.add_argument("--query-frame", type=int, default=0)
    p.add_argument("--image-size", type=int, default=512, help="Max side for resize (lower saves GPU memory)")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap frames per episode (0 = full episode; required for long LIBERO clips)",
    )
    p.add_argument("--inference-iters", type=int, default=4)
    p.add_argument("--window-len", type=int, default=16)
    p.add_argument("--rate", type=int, default=8, help="Draw every Rth point in H,W (sparser grid)")
    p.add_argument("--conf-thr", type=float, default=0.1)
    p.add_argument(
        "--min-motion-px",
        type=float,
        default=1.0,
        help="Only draw points that move at least this many pixels from --query-frame (0 = all confident points)",
    )
    p.add_argument(
        "--bkg-opacity",
        type=float,
        default=0.0,
        help="RGB background weight behind tracks (0=black, 0.5=dimmed RGB like demo.py)",
    )
    p.add_argument(
        "--points-only",
        action="store_true",
        help="Alias for --bkg-opacity 0 (default for this script)",
    )
    p.add_argument(
        "--rgb-background",
        action="store_true",
        help="Dim RGB behind all confident points (demo style; sets bkg_opacity=0.5, min_motion_px=0 unless set)",
    )
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output mp4 already exists")
    p.add_argument("--limit", type=int, default=0, help="Process at most N episodes (0 = all)")
    p.add_argument("--device", type=str, default="cuda:0", help="CUDA device for this process, e.g. cuda:0 / cuda:1")
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Videos per forward pass. Clips are grouped by resolution and padded to the batch's "
        "max length. Raise until GPU memory fills (you have lots of headroom).",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the episode list into N disjoint shards. Launch one process per shard "
        "(optionally one GPU each via --device) to scale across cores/GPUs.",
    )
    p.add_argument("--shard-id", type=int, default=0, help="Which shard this process handles (0..num-shards-1).")
    p.add_argument(
        "--decode-workers",
        type=int,
        default=3,
        help="Background threads decoding upcoming videos while the GPU runs.",
    )
    p.add_argument(
        "--encode-workers",
        type=int,
        default=2,
        help="Background threads encoding finished output videos.",
    )
    p.add_argument(
        "--sort-by-length",
        action="store_true",
        help="Probe frame counts and sort clips so each batch has similar lengths (less padding waste). "
        "Strongly recommended when clip lengths vary a lot.",
    )
    p.add_argument(
        "--max-batch-frames",
        type=int,
        default=0,
        help="Cap on (batch_size * padded_length) per forward pass. When clips vary in length, this "
        "shrinks the batch for long clips so a short clip is never padded out to a huge length "
        "(saves GPU memory + wasted compute). 0 = rely only on --batch-size.",
    )
    p.add_argument(
        "--video-backend",
        choices=("auto", "ffmpeg", "pyav", "opencv"),
        default="auto",
        help="How to decode input MP4s. ``ffmpeg`` uses software decode (best for AV1). "
        "``auto`` tries ffmpeg, then PyAV, then OpenCV.",
    )
    return p


def run(args: argparse.Namespace) -> None:
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        sys.exit("Not a directory: %s" % root)
    if not (0 <= args.shard_id < args.num_shards):
        sys.exit("--shard-id must be in [0, --num-shards)")

    torch.set_grad_enabled(False)
    device = args.device
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    if args.tiny:
        model = Net(args.window_len, use_basicencoder=True, no_split=True)
    else:
        model = Net(args.window_len)
    _load_weights(model, args.ckpt_init, args.tiny, device)

    if args.num_shards > 1:
        print("Shard %d/%d" % (args.shard_id, args.num_shards))
    pairs = _select_pairs(root, args.num_shards, args.shard_id, args.skip_existing, args.limit)
    if not pairs and args.num_shards <= 1:
        all_pairs = _iter_episode_mp4s(root)
        if not all_pairs:
            print("No episodes found under", root)
            print("Expected subfolders named", " or ".join(a for a, _ in DIR_PAIRS))
            return
    if not pairs:
        print("Nothing to do on this shard.")
        return
    print("Processing %d episode(s) on this shard." % len(pairs))

    # Sort so each batch holds similar-resolution / similar-length clips (minimizes padding waste).
    if args.sort_by_length:
        print("Probing %d clip(s) for length-aware batching..." % len(pairs))
        with ThreadPoolExecutor(max_workers=max(4, args.decode_workers * 2)) as pex:
            metas = list(pex.map(lambda sd: _probe_native_meta(sd[0]), pairs))
        order = sorted(range(len(pairs)), key=lambda i: (metas[i][0], metas[i][1], metas[i][2]))
        pairs = [pairs[i] for i in order]

    bkg = float(args.bkg_opacity)
    if args.points_only:
        bkg = 0.0
    min_motion_px = float(args.min_motion_px)
    if args.rgb_background:
        bkg = 0.5
        min_motion_px = 0.0
    max_frames = args.max_frames if args.max_frames else 0
    print(
        "  render: bkg_opacity=%.2f min_motion_px=%.2f max_frames=%s batch_size=%d device=%s"
        % (bkg, min_motion_px, max_frames or "all", args.batch_size, device)
    )

    total = len(pairs)
    encode_ex = ThreadPoolExecutor(max_workers=max(1, args.encode_workers))
    pending: deque = deque()
    processed = 0

    def reap(block: bool = False) -> None:
        while pending and (block or pending[0][0].done()):
            fut, dst = pending.popleft()
            fut.result()
            print("  wrote", dst)

    def run_batch(items: list[tuple[Path, Path, torch.Tensor, int]]) -> None:
        nonlocal processed
        batch, lengths = _collate_batch(items)
        frames_list = _infer_point_frames_rgb_batched(
            batch,
            lengths,
            model,
            device,
            query_frame=args.query_frame,
            inference_iters=args.inference_iters,
            rate=args.rate,
            conf_thr=args.conf_thr,
            min_motion_px=min_motion_px,
            bkg_opacity=bkg,
        )
        for (src, dst, _, fps), frames in zip(items, frames_list):
            pending.append((encode_ex.submit(_write_mp4, frames, dst, int(fps)), dst))
        processed += len(items)
        print("[%d/%d] processed (batch of %d)" % (processed, total, len(items)))
        reap(block=False)

    stream = _decode_pairs(
        pairs,
        args.image_size,
        max_frames,
        args.video_backend,
        workers=args.decode_workers,
        lookahead=args.batch_size + args.decode_workers,
    )

    def would_exceed_budget(buf_items, new_len: int) -> bool:
        if not args.max_batch_frames or not buf_items:
            return False
        lens = [it[2].shape[1] for it in buf_items] + [new_len]
        return (len(lens) * max(lens)) > args.max_batch_frames

    buf: list[tuple[Path, Path, torch.Tensor, int]] = []
    buf_shape: tuple[int, ...] | None = None
    for src, dst, rgbs, fps in stream:
        shape = tuple(rgbs.shape[2:])  # (c, h, w) must match within a batch
        new_len = rgbs.shape[1]
        if buf and (
            shape != buf_shape
            or len(buf) >= args.batch_size
            or would_exceed_budget(buf, new_len)
        ):
            run_batch(buf)
            buf = []
        buf_shape = shape
        buf.append((src, dst, rgbs, fps))
    if buf:
        run_batch(buf)

    reap(block=True)
    encode_ex.shutdown(wait=True)
    print("Done.")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

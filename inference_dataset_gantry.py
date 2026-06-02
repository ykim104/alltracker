#!/usr/bin/env python3
"""
Beaker / Gantry entry point for distributed AllTracker dataset inference.

Each worker gets a disjoint shard of the globally sorted episode list. Re-run the same
job after preemption with ``--skip-existing`` (default here): completed outputs are
validated via ffprobe; partial ``.tmp`` files and corrupt MP4s are reprocessed.

Shard selection (first match wins):
  1. ``--shard-id`` / ``--num-shards`` CLI
  2. ``BEAKER_REPLICA_RANK`` / ``BEAKER_REPLICA_COUNT`` (multi-replica Gantry jobs)
  3. ``ALLTRACKER_SHARD_ID`` / ``ALLTRACKER_NUM_SHARDS`` env vars

For a single node with N GPUs, ``scripts/beaker/run_inference.sh`` launches N processes
with ``--shard-id 0..N-1`` and ``--num-shards N`` instead of using replica env vars.

Example (local multi-GPU on one machine)::

  NUM_GPUS=4 bash scripts/beaker/run_inference.sh

Example (Gantry default: 10 replicas × 1 GPU)::

  bash scripts/beaker/launch_inference_gantry.sh --user-name yejink \\
    --dataset-root /weka/oe-training/yejink/data/robotwin2.0/robotwin2.0

Example (single node, N GPUs)::

  bash scripts/beaker/launch_inference_gantry.sh ... --mode multi-gpu --replicas 1 --gpus 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from inference_dataset import build_parser, run  # noqa: E402


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _apply_gantry_shard_defaults(args: argparse.Namespace) -> None:
    """Fill shard fields from Beaker replica env when CLI left them at defaults."""
    replica_rank = _env_int("BEAKER_REPLICA_RANK")
    replica_count = _env_int("BEAKER_REPLICA_COUNT")
    if replica_rank is not None and replica_count is not None:
        if args.shard_id == 0 and args.num_shards == 1:
            args.shard_id = replica_rank
            args.num_shards = replica_count
            print(
                "Using Beaker replica shard: rank=%d count=%d"
                % (args.shard_id, args.num_shards)
            )
            return

    env_shard = _env_int("ALLTRACKER_SHARD_ID")
    env_shards = _env_int("ALLTRACKER_NUM_SHARDS")
    if env_shard is not None and env_shards is not None:
        if args.shard_id == 0 and args.num_shards == 1:
            args.shard_id = env_shard
            args.num_shards = env_shards
            print(
                "Using ALLTRACKER shard: id=%d num=%d"
                % (args.shard_id, args.num_shards)
            )


def main() -> None:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess even when a valid output MP4 already exists (default: skip complete outputs).",
    )
    args = parser.parse_args()

    if not args.no_skip_existing:
        args.skip_existing = True

    gpu_id = _env_int("ALLTRACKER_GPU_ID")
    if gpu_id is not None and args.device == "cuda:0":
        args.device = "cuda:%d" % gpu_id

    _apply_gantry_shard_defaults(args)

    if args.num_shards < 1:
        sys.exit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        sys.exit("--shard-id must be in [0, --num-shards)")

    print(
        "gantry worker: shard=%d/%d device=%s dataset=%s skip_existing=%s"
        % (
            args.shard_id,
            args.num_shards,
            args.device,
            args.dataset_root,
            args.skip_existing,
        )
    )
    run(args)


if __name__ == "__main__":
    main()

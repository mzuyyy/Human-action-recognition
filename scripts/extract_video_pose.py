#!/usr/bin/env python3
"""Extract a single dominant COCO-17 person sequence from a video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.constants import MMPOSE_CHECKPOINT
from src.inference.pose_estimator import (
    DominantPersonPosePipeline,
)
from src.inference.video_pose import (
    extract_video_pose,
    resolve_device,
    save_pose_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/example_pose.pkl')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--detector-threshold', type=float, default=0.5)
    parser.add_argument('--pose-config', type=Path)
    parser.add_argument('--pose-checkpoint', default=MMPOSE_CHECKPOINT)
    parser.add_argument('--progress-interval', type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    pipeline = DominantPersonPosePipeline(
        device=device, detector_threshold=args.detector_threshold,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint)
    payload = extract_video_pose(
        args.video, pipeline, progress_interval=args.progress_interval)
    save_pose_payload(payload, args.output)
    summary = {
        **payload['metadata'],
        **payload['latency'],
        'keypoint_shape': list(payload['keypoint'].shape),
        'keypoint_score_shape': list(payload['keypoint_score'].shape),
        'output': str(args.output.resolve()),
        'device': device,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

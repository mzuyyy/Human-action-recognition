#!/usr/bin/env python3
"""Run the frozen Joint ST-GCN stream on an extracted pose sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.action_recognizer import STGCNStream, top_k
from src.inference.constants import (
    JOINT_CHECKPOINT,
    JOINT_CONFIG,
)
from src.inference.skeleton_preprocessor import (
    pose_arrays_from_payload,
)
from src.inference.video_pose import (
    load_pose_payload,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose', type=Path)
    parser.add_argument(
        '--checkpoint', type=Path,
        default=PROJECT_ROOT / JOINT_CHECKPOINT)
    parser.add_argument(
        '--config', type=Path, default=PROJECT_ROOT / JOINT_CONFIG)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_pose_payload(args.pose)
    keypoints, scores, image_shape = pose_arrays_from_payload(payload)
    end = args.end_frame if args.end_frame is not None else keypoints.shape[1]
    if not 0 <= args.start_frame < end <= keypoints.shape[1]:
        raise ValueError(
            f'invalid frame range [{args.start_frame}, {end}) for '
            f'{keypoints.shape[1]} frames')
    keypoints = keypoints[:, args.start_frame:end]
    scores = scores[:, args.start_frame:end]
    device = resolve_device(args.device)
    model = STGCNStream(
        'joint', args.checkpoint, config=args.config, device=device,
        project_root=PROJECT_ROOT)
    output, preprocess_ms = model(keypoints, scores, image_shape)
    result = {
        'stream': 'joint',
        'source_pose': str(args.pose.resolve()),
        'frame_range': [args.start_frame, end],
        'raw_keypoint_shape': list(keypoints.shape),
        'model_input_shape': list(output.input_shape),
        'preprocessing_ms': preprocess_ms,
        'inference_ms': output.latency_ms,
        'top1': top_k(output.scores, 1)[0],
        'top5': top_k(output.scores, 5),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

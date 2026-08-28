#!/usr/bin/env python3
"""Verify COCO-17 MMPose inference on one detected person image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.constants import (
    COCO_KEYPOINT_NAMES,
    DETECTOR_NAME,
    MMPOSE_CHECKPOINT,
    POSE_MODEL_NAME,
)
from src.inference.pose_estimator import (
    DominantPersonPosePipeline,
    draw_pose,
)
from src.inference.video_pose import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input', nargs='?', type=Path,
        default=PROJECT_ROOT / 'assets/test_person.jpg')
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/test_pose.jpg')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--detector-threshold', type=float, default=0.5)
    parser.add_argument('--pose-config', type=Path)
    parser.add_argument('--pose-checkpoint', default=MMPOSE_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2
    import mmpose

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f'cannot read test image: {args.input}')
    device = resolve_device(args.device)
    pipeline = DominantPersonPosePipeline(
        device=device, detector_threshold=args.detector_threshold,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint)
    pose = pipeline(image, frame_index=0)
    if not pose.detected:
        raise RuntimeError('no person detected in test image')
    if pose.keypoints.shape != (17, 2):
        raise RuntimeError(f'wrong keypoint shape: {pose.keypoints.shape}')
    if pose.keypoint_scores.shape != (17,):
        raise RuntimeError(
            f'wrong keypoint score shape: {pose.keypoint_scores.shape}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), draw_pose(image, pose)):
        raise RuntimeError(f'failed to write {args.output}')
    environment = {
        'mmpose_version': mmpose.__version__,
        'pose_model': POSE_MODEL_NAME,
        'person_detector': DETECTOR_NAME,
        'layout': 'coco',
        'num_joints': len(COCO_KEYPOINT_NAMES),
        'keypoint_names': list(COCO_KEYPOINT_NAMES),
        'keypoints_shape': list(pose.keypoints.shape),
        'keypoint_scores_shape': list(pose.keypoint_scores.shape),
        'device': device,
    }
    (args.output.parent / 'pose_environment.json').write_text(
        json.dumps(environment, indent=2))
    print('MMPose version:', mmpose.__version__)
    print('COCO joint order:', ', '.join(COCO_KEYPOINT_NAMES))
    print('keypoints shape:', pose.keypoints.shape)
    print('keypoint scores shape:', pose.keypoint_scores.shape)
    print('saved:', args.output)


if __name__ == '__main__':
    main()

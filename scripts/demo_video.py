#!/usr/bin/env python3
"""Offline video -> MMPose -> Joint/Motion ST-GCN -> annotated MP4."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.action_recognizer import JointMotionFusion
from src.inference.constants import (
    CLIP_LEN,
    JOINT_CHECKPOINT,
    JOINT_CONFIG,
    MMPOSE_CHECKPOINT,
    MOTION_CHECKPOINT,
)
from src.inference.pose_estimator import (
    DominantPersonPosePipeline,
)
from src.inference.video_action import (
    annotate_video,
    build_latency_report,
    infer_pose_windows,
    save_json,
)
from src.inference.video_pose import (
    extract_video_pose,
    load_pose_payload,
    resolve_device,
    save_pose_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'artifacts/demo/demo_output.mp4')
    parser.add_argument(
        '--pose-file', type=Path,
        help='Reuse an existing pose pickle instead of extracting poses.')
    parser.add_argument(
        '--save-pose', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/example_pose.pkl')
    parser.add_argument(
        '--joint-checkpoint', type=Path,
        default=PROJECT_ROOT / JOINT_CHECKPOINT)
    parser.add_argument(
        '--motion-checkpoint', type=Path,
        default=PROJECT_ROOT / MOTION_CHECKPOINT)
    parser.add_argument(
        '--action-config', type=Path, default=PROJECT_ROOT / JOINT_CONFIG)
    parser.add_argument('--pose-config', type=Path)
    parser.add_argument('--pose-checkpoint', default=MMPOSE_CHECKPOINT)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--detector-threshold', type=float, default=0.5)
    parser.add_argument('--keypoint-threshold', type=float, default=0.3)
    parser.add_argument('--window-frames', type=int, default=CLIP_LEN)
    parser.add_argument('--window-stride', type=int, default=CLIP_LEN // 2)
    parser.add_argument('--ema-momentum', type=float, default=0.8)
    parser.add_argument('--show-stream-scores', action='store_true')
    parser.add_argument(
        '--latency-report', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/latency_report.json')
    parser.add_argument(
        '--window-predictions', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/window_predictions.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.pose_file is not None:
        payload = load_pose_payload(args.pose_file)
        source = Path(payload['metadata']['source_video']).resolve()
        if source != args.input.expanduser().resolve():
            raise RuntimeError(
                f'pose file belongs to {source}, not '
                f'{args.input.expanduser().resolve()}')
    else:
        pose_pipeline = DominantPersonPosePipeline(
            device=device, detector_threshold=args.detector_threshold,
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint)
        payload = extract_video_pose(args.input, pose_pipeline)
        save_pose_payload(payload, args.save_pose)
        del pose_pipeline
        gc.collect()
        if device.startswith('cuda'):
            import torch

            torch.cuda.empty_cache()

    action_model = JointMotionFusion(
        joint_checkpoint=args.joint_checkpoint,
        motion_checkpoint=args.motion_checkpoint,
        config=args.action_config, device=device, project_root=PROJECT_ROOT)
    action_started = time.perf_counter()
    windows = infer_pose_windows(
        payload, action_model, window_frames=args.window_frames,
        stride=args.window_stride, ema_momentum=args.ema_momentum)
    action_inference_wall_time = time.perf_counter() - action_started
    latency = build_latency_report(
        payload, windows, stride=args.window_stride,
        action_inference_wall_time_sec=action_inference_wall_time)
    save_json(latency, args.latency_report)
    save_json(windows, args.window_predictions)
    annotate_video(
        args.input, args.output, payload, windows,
        pipeline_fps=latency['estimated_pipeline_fps'],
        show_stream_scores=args.show_stream_scores,
        keypoint_threshold=args.keypoint_threshold)
    final = windows[-1]['smoothed_top1']
    print(json.dumps({
        'device': device,
        'frames': payload['metadata']['frame_count'],
        'windows': len(windows),
        'final_smoothed_prediction': final,
        'annotated_video': str(args.output.resolve()),
        'pose_file': str((args.pose_file or args.save_pose).resolve()),
        'latency_report': str(args.latency_report.resolve()),
        'window_predictions': str(args.window_predictions.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()

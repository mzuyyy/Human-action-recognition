#!/usr/bin/env python3
"""Build the final video-inference milestone report from measured artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.constants import (
    DETECTOR_NAME,
    FUSION_JOINT_WEIGHT,
    FUSION_MOTION_WEIGHT,
    JOINT_CHECKPOINT,
    MOTION_CHECKPOINT,
    POSE_MODEL_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pose-environment', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/pose_environment.json')
    parser.add_argument(
        '--mmpose-patch', type=Path,
        default=PROJECT_ROOT
        / 'artifacts/inference/mmpose_lite_patch.json')
    parser.add_argument(
        '--preprocessing-verification', type=Path,
        default=PROJECT_ROOT
        / 'artifacts/inference/preprocessing_verification.json')
    parser.add_argument(
        '--latency', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/latency_report.json')
    parser.add_argument(
        '--domain-gap', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/domain_gap_test.csv')
    parser.add_argument(
        '--demo-video', type=Path,
        default=PROJECT_ROOT / 'artifacts/demo/demo_output.mp4')
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT
        / 'artifacts/inference/video_inference_report.md')
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f'expected JSON object: {path}')
    return data


def as_float(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'invalid {name}: {value!r}') from error


def main() -> None:
    args = parse_args()
    pose = read_json(args.pose_environment)
    mmpose_patch = read_json(args.mmpose_patch)
    preprocessing = read_json(args.preprocessing_verification)
    latency = read_json(args.latency)
    if preprocessing.get('status') != 'pass':
        raise RuntimeError('preprocessing compatibility did not pass')
    if (mmpose_patch.get('status') != 'pass'
            or mmpose_patch.get('selected_pose_model_affected') is not False):
        raise RuntimeError('MMPose lite compatibility check did not pass')
    if not args.demo_video.is_file() or args.demo_video.stat().st_size == 0:
        raise FileNotFoundError(args.demo_video)
    if not args.domain_gap.is_file():
        raise FileNotFoundError(args.domain_gap)
    with args.domain_gap.open(newline='') as stream:
        videos = list(csv.DictReader(stream))
    if len(videos) < 5:
        raise RuntimeError(
            f'final report requires at least 5 domain-gap videos, got '
            f'{len(videos)}')
    correct = sum(row.get('correct', '').strip().casefold() == 'true'
                  for row in videos)
    incorrect = len(videos) - correct

    weak_pose = [
        row for row in videos
        if row.get('qualitative_pose_quality') != 'good'
    ]
    wrong = [row for row in videos
             if row.get('correct', '').strip().casefold() != 'true']
    issue_parts = []
    if weak_pose:
        issue_parts.append(
            f'{len(weak_pose)} clip(s) had mixed/poor heuristic pose quality')
    if wrong:
        pairs = ', '.join(
            f"{row['expected_action']} -> {row['predicted_action']}"
            for row in wrong)
        issue_parts.append(f'observed action errors: {pairs}')
    if not issue_parts:
        issue_parts.append(
            'no error in this small selected set; this is not evidence of '
            'general real-world accuracy')
    issues = '; '.join(issue_parts)

    detection = as_float(latency['detection_ms_per_frame'], 'detection')
    pose_ms = as_float(latency['pose_ms_per_frame'], 'pose')
    skeleton = as_float(
        latency['skeleton_preprocessing_ms_per_window'], 'preprocessing')
    joint = as_float(latency['joint_stgcn_ms_per_window'], 'joint ST-GCN')
    motion = as_float(latency['motion_stgcn_ms_per_window'], 'motion ST-GCN')
    total = as_float(
        latency['estimated_total_ms_per_frame'], 'estimated total')
    fps = as_float(latency['estimated_pipeline_fps'], 'estimated FPS')
    measured_total = as_float(
        latency['measured_pipeline_ms_per_frame'], 'measured total')
    measured_fps = as_float(
        latency['measured_pipeline_fps'], 'measured FPS')

    if weak_pose:
        recommendation = (
            'Inspect detector/pose failures and identity stability on a larger '
            'video set; add tracking only if slot instability is confirmed.')
    elif incorrect:
        recommendation = (
            'Expand the real-video domain-gap set and inspect the failed '
            'skeleton sequences before changing or retraining a model.')
    else:
        recommendation = (
            'Validate on a larger, more diverse labeled video set before any '
            'ONNX/TensorRT optimization or architecture change.')

    input_shape = tuple(preprocessing['batched_model_input_shape'])
    report = f"""POSE MODEL
----------
Model: {POSE_MODEL_NAME}
Person detector: {DETECTOR_NAME}
MMPose version: {pose['mmpose_version']}
Compatibility: unused EDPose/RTMO registries disabled; selected MobileNetV2 + HeatmapHead unchanged
Layout: {pose['layout'].upper()}
Number of joints: {pose['num_joints']}

TRAINING COMPATIBILITY
----------------------
Training skeleton layout: COCO
Online skeleton layout: {pose['layout'].upper()}
Temporal preprocessing: official MMAction2 UniformSampleFrames(clip_len=100, num_clips=1, test_mode=True, seed=255); overlapping 100-frame windows with 50-frame stride
Joint Motion: official GenSkeFeat(feats=['jm']) before temporal sampling
Final input tensor shape: {input_shape}

ACTION MODELS
-------------
Joint checkpoint: {JOINT_CHECKPOINT}
Motion checkpoint: {MOTION_CHECKPOINT}
Fusion weights: Joint={FUSION_JOINT_WEIGHT:.1f}, Joint Motion={FUSION_MOTION_WEIGHT:.1f}

VIDEO TEST
----------
Number of videos: {len(videos)}
Correct: {correct}
Incorrect: {incorrect}
Observed domain-gap issues: {issues}

LATENCY
-------
Detection: {detection:.3f} ms/frame
Pose: {pose_ms:.3f} ms/frame
Skeleton preprocessing: {skeleton:.3f} ms/window
Joint ST-GCN: {joint:.3f} ms/window
Motion ST-GCN: {motion:.3f} ms/window
Total: {measured_total:.3f} measured ms/frame ({measured_fps:.2f} FPS); stage-sum estimate {total:.3f} ms/frame ({fps:.2f} FPS)

OUTPUTS
-------
Demo video: artifacts/demo/demo_output.mp4
Pose extraction: scripts/extract_video_pose.py
Preprocessor: src/inference/skeleton_preprocessor.py
Inference script: scripts/demo_video.py
Latency report: artifacts/inference/latency_report.json

LIMITATIONS
-----------
- The detector selects one dominant person independently per frame; it does not track identity.
- The model recognizes only the 60 NTU60 labels and may not transfer reliably to unconstrained videos.
- Missing detections become zero-confidence skeleton frames; camera motion, occlusion, framing, and pose noise create a domain gap.
- Measured pipeline latency includes video decoding, detection, pose, preprocessing, and action inference; it excludes one-time model downloads/initialization and MP4 rendering.
- The {len(videos)}-video qualitative test is too small and selected to estimate general real-world accuracy.

NEXT RECOMMENDED STEP
---------------------
{recommendation}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)
    print('saved:', args.output)


if __name__ == '__main__':
    main()

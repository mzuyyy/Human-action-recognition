#!/usr/bin/env python3
"""Evaluate at least five labeled real-video clips without accuracy claims."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.inference.action_recognizer import (
    JointMotionFusion,
    top_k,
)
from src.inference.constants import (
    CLIP_LEN,
    JOINT_CHECKPOINT,
    JOINT_CONFIG,
    MMPOSE_CHECKPOINT,
    MOTION_CHECKPOINT,
    NTU60_CLASS_NAMES,
)
from src.inference.pose_estimator import (
    DominantPersonPosePipeline,
)
from src.inference.video_action import infer_pose_windows
from src.inference.video_pose import (
    extract_video_pose,
    load_pose_payload,
    resolve_device,
    save_pose_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/domain_gap_test.csv')
    parser.add_argument(
        '--pose-cache-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/inference/domain_gap_poses')
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
    parser.add_argument('--window-frames', type=int, default=CLIP_LEN)
    parser.add_argument('--window-stride', type=int, default=CLIP_LEN // 2)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'video_path', 'expected_action'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f'manifest must contain columns {sorted(required)}')
    if len(rows) < 5:
        raise ValueError(
            f'domain-gap test requires at least 5 videos, got {len(rows)}')
    return rows


def action_id(name: str) -> int:
    normalized = name.strip().casefold()
    matches = [
        index for index, action in enumerate(NTU60_CLASS_NAMES)
        if action.casefold() == normalized
    ]
    if len(matches) != 1:
        raise ValueError(
            f'expected_action must exactly match one NTU60 name: {name!r}')
    return matches[0]


def pose_quality(payload: dict) -> tuple[str, str, float]:
    detection_rate = float(payload['latency']['detection_rate'])
    scores = np.asarray(payload['keypoint_score'])[0]
    detected = np.asarray(payload['detection_score']) > 0
    mean_score = float(np.mean(scores[detected])) if np.any(detected) else 0.0
    if detection_rate >= 0.9 and mean_score >= 0.6:
        quality = 'good'
    elif detection_rate >= 0.7 and mean_score >= 0.4:
        quality = 'mixed'
    else:
        quality = 'poor'
    notes = (
        f'detection_rate={detection_rate:.3f}; '
        f'mean_keypoint_confidence={mean_score:.3f}; '
        'quality label is heuristic, not a ground-truth pose metric')
    return quality, notes, mean_score


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    device = resolve_device(args.device)
    pose_pipeline = DominantPersonPosePipeline(
        device=device, detector_threshold=args.detector_threshold,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint)
    args.pose_cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for index, row in enumerate(rows, start=1):
        video_path = Path(row['video_path']).expanduser()
        if not video_path.is_absolute():
            video_path = args.manifest.parent / video_path
        video_path = video_path.resolve()
        action_id(row['expected_action'])  # fail before expensive extraction
        print(f'domain-gap video {index}/{len(rows)}: {video_path}')
        payload = extract_video_pose(video_path, pose_pipeline)
        pose_path = args.pose_cache_dir / f'{index:02d}_{video_path.stem}.pkl'
        save_pose_payload(payload, pose_path)
        extracted.append((row, video_path, pose_path))

    # Release detection/pose networks before loading the two action streams.
    # This keeps the workflow comfortable on a 16 GB Kaggle T4 and does not
    # mix model initialization time into per-stage inference measurements.
    del pose_pipeline
    gc.collect()
    if device.startswith('cuda'):
        import torch

        torch.cuda.empty_cache()
    action_model = JointMotionFusion(
        joint_checkpoint=args.joint_checkpoint,
        motion_checkpoint=args.motion_checkpoint,
        config=args.action_config, device=device, project_root=PROJECT_ROOT)

    outputs = []
    for row, video_path, pose_path in extracted:
        payload = load_pose_payload(pose_path)
        expected_id = action_id(row['expected_action'])
        windows = infer_pose_windows(
            payload, action_model, window_frames=args.window_frames,
            stride=args.window_stride)
        mean_scores = np.mean(
            [window['fusion_scores'] for window in windows], axis=0)
        prediction = top_k(mean_scores, 1)[0]
        quality, pose_notes, mean_pose_score = pose_quality(payload)
        manifest_notes = row.get('notes', '').strip()
        notes = pose_notes
        if manifest_notes:
            notes = f'{pose_notes}; reviewer_notes={manifest_notes}'
        outputs.append({
            'video_path': str(video_path),
            'expected_id': expected_id,
            'expected_action': NTU60_CLASS_NAMES[expected_id],
            'predicted_id': prediction['class_id'],
            'predicted_action': prediction['action'],
            'confidence': prediction['probability'],
            'correct': prediction['class_id'] == expected_id,
            'qualitative_pose_quality': quality,
            'detection_rate': payload['latency']['detection_rate'],
            'mean_keypoint_confidence': mean_pose_score,
            'windows': len(windows),
            'notes': notes,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(outputs[0])
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outputs)
    correct = sum(bool(row['correct']) for row in outputs)
    print(
        f'domain-gap clips: {len(outputs)}, correct: {correct}, '
        f'incorrect: {len(outputs) - correct}')
    print('saved:', args.output)
    print(
        'This small, selected set is qualitative and must not be reported as '
        'general real-world accuracy.')


if __name__ == '__main__':
    main()

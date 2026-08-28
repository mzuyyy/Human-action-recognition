#!/usr/bin/env python3
"""Prove that online preprocessing matches the two successful val configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mmengine.config import Config

from src.inference.constants import (
    CLIP_LEN,
    NUM_JOINTS,
    NUM_PERSONS,
    UNIFORM_SAMPLE_SEED,
)
from src.inference.skeleton_preprocessor import (
    SkeletonPreprocessor,
)

JOINT_CONFIG = PROJECT_ROOT / 'configs/stgcn_ntu60_xsub_baseline.py'
MOTION_CONFIG = (
    PROJECT_ROOT / 'configs/stgcn_ntu60_xsub_joint_motion_80e.py')
OUTPUT = PROJECT_ROOT / 'artifacts/inference/preprocessing_verification.json'


def compact_pipeline(config_path: Path) -> list[dict]:
    cfg = Config.fromfile(str(config_path))
    pipeline = cfg.val_dataloader.dataset.pipeline
    return [dict(step) for step in pipeline]


def assert_pipeline(pipeline: list[dict], feature: str) -> None:
    expected_types = [
        'PreNormalize2D',
        'GenSkeFeat',
        'UniformSampleFrames',
        'PoseDecode',
        'FormatGCNInput',
        'PackActionInputs',
    ]
    if [step['type'] for step in pipeline] != expected_types:
        raise RuntimeError(
            f'validation transform order changed: {pipeline}')
    feature_step = pipeline[1]
    if (feature_step.get('dataset') != 'coco'
            or feature_step.get('feats') != [feature]):
        raise RuntimeError(f'wrong GenSkeFeat config: {feature_step}')
    sample_step = pipeline[2]
    if (sample_step.get('clip_len') != CLIP_LEN
            or sample_step.get('num_clips', 1) != 1
            or sample_step.get('test_mode') is not True
            or sample_step.get('seed', UNIFORM_SAMPLE_SEED)
            != UNIFORM_SAMPLE_SEED):
        raise RuntimeError(f'wrong validation sampler: {sample_step}')
    format_step = pipeline[4]
    if (format_step.get('num_person') != NUM_PERSONS
            or format_step.get('mode', 'zero') != 'zero'):
        raise RuntimeError(f'wrong GCN formatter: {format_step}')


def synthetic_pose() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    frames = 137
    height, width = 360, 640
    time_axis = np.arange(frames, dtype=np.float32)[None, :, None]
    joint_axis = np.arange(NUM_JOINTS, dtype=np.float32)[None, None, :]
    keypoints = np.empty((1, frames, NUM_JOINTS, 2), dtype=np.float32)
    keypoints[..., 0] = 180 + 0.7 * time_axis + 2.0 * joint_axis
    keypoints[..., 1] = 80 + 0.4 * time_axis + 1.5 * joint_axis
    scores = np.broadcast_to(
        0.55 + 0.4 * joint_axis / (NUM_JOINTS - 1),
        (1, frames, NUM_JOINTS)).astype(np.float32).copy()
    return keypoints, scores, (height, width)


def normalized_features(
        keypoints: np.ndarray, scores: np.ndarray,
        image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    normalized = keypoints.copy()
    normalized[..., 0] = (normalized[..., 0] - width / 2) / (width / 2)
    normalized[..., 1] = (normalized[..., 1] - height / 2) / (height / 2)
    return np.concatenate((normalized, scores[..., None]), axis=-1)


def main() -> None:
    joint_pipeline = compact_pipeline(JOINT_CONFIG)
    motion_pipeline = compact_pipeline(MOTION_CONFIG)
    assert_pipeline(joint_pipeline, 'j')
    assert_pipeline(motion_pipeline, 'jm')

    keypoints, scores, image_shape = synthetic_pose()
    joint = SkeletonPreprocessor('joint')(
        keypoints, scores, image_shape)
    motion = SkeletonPreprocessor('joint_motion')(
        keypoints, scores, image_shape)
    expected_shape = (1, NUM_PERSONS, CLIP_LEN, NUM_JOINTS, 3)
    if joint.data.shape != expected_shape or motion.data.shape != expected_shape:
        raise RuntimeError('unexpected final GCN input shape')
    np.testing.assert_array_equal(joint.frame_indices, motion.frame_indices)

    features = normalized_features(keypoints, scores, image_shape)
    indices = joint.frame_indices
    expected_joint = features[:, indices]
    np.testing.assert_allclose(
        joint.data[0, 0], expected_joint[0], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(joint.data[0, 1], 0)

    expected_motion = np.zeros_like(features)
    expected_motion[:, :-1, :, :2] = np.diff(
        features[..., :2], axis=1)
    expected_motion[:, :-1, :, 2] = (
        features[:, :-1, :, 2] + features[:, 1:, :, 2]) / 2
    expected_motion = expected_motion[:, indices]
    np.testing.assert_allclose(
        motion.data[0, 0], expected_motion[0], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(motion.data[0, 1], 0)

    report = {
        'status': 'pass',
        'joint_config': str(JOINT_CONFIG.relative_to(PROJECT_ROOT)),
        'motion_config': str(MOTION_CONFIG.relative_to(PROJECT_ROOT)),
        'joint_validation_pipeline': joint_pipeline,
        'motion_validation_pipeline': motion_pipeline,
        'raw_shape': list(keypoints.shape),
        'sampled_frame_indices': indices.tolist(),
        'sample_shape': list(joint.data.shape),
        'batched_model_input_shape': [1, *joint.data.shape],
        'motion_generation': (
            "official GenSkeFeat(feats=['jm']) before temporal sampling"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

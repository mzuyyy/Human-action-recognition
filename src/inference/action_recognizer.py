"""Frozen ST-GCN stream inference and fixed Joint/Motion fusion."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    CLIP_LEN,
    FUSION_JOINT_WEIGHT,
    FUSION_MOTION_WEIGHT,
    JOINT_CHECKPOINT,
    JOINT_CONFIG,
    MOTION_CHECKPOINT,
    NTU60_CLASS_NAMES,
    NUM_CLASSES,
    NUM_CLIPS,
    NUM_JOINTS,
    NUM_PERSONS,
)
from .skeleton_preprocessor import SkeletonPreprocessor, Stream

# These are trusted MMEngine checkpoints produced by this project. MMEngine
# 0.10.x does not pass weights_only explicitly under PyTorch 2.6+.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


@dataclass(frozen=True)
class StreamOutput:
    scores: np.ndarray
    latency_ms: float
    input_shape: tuple[int, ...]


@dataclass(frozen=True)
class FusionOutput:
    joint_scores: np.ndarray
    motion_scores: np.ndarray
    fusion_scores: np.ndarray
    joint_ms: float
    motion_ms: float
    fusion_ms: float
    joint_preprocess_ms: float
    motion_preprocess_ms: float
    input_shape: tuple[int, ...]


def _synchronize(device: str) -> None:
    if device.startswith('cuda'):
        import torch

        torch.cuda.synchronize()


def _project_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class STGCNStream:
    """One frozen ST-GCN model with a training-compatible input stream."""

    def __init__(
            self, stream: Stream, checkpoint: str | Path,
            config: str | Path = JOINT_CONFIG, device: str = 'cuda:0',
            project_root: str | Path | None = None):
        import torch
        from mmaction.registry import MODELS
        from mmaction.utils import register_all_modules
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint

        self.torch = torch
        self.stream = stream
        self.device = device
        self.project_root = Path(
            project_root or Path(__file__).resolve().parents[2]).resolve()
        self.config_path = _project_path(config, self.project_root)
        self.checkpoint_path = _project_path(checkpoint, self.project_root)
        register_all_modules(init_default_scope=True)
        config_obj = Config.fromfile(str(self.config_path))
        self.model = MODELS.build(config_obj.model)
        load_checkpoint(
            self.model, str(self.checkpoint_path), map_location='cpu')
        self.model.to(device).eval()
        self.preprocessor = SkeletonPreprocessor(stream)

    def infer_preprocessed(self, sample: np.ndarray) -> StreamOutput:
        sample = np.asarray(sample, dtype=np.float32)
        expected = (NUM_CLIPS, NUM_PERSONS, CLIP_LEN, NUM_JOINTS, 3)
        if sample.shape != expected:
            raise ValueError(
                f'{self.stream} sample shape must be {expected}, got '
                f'{sample.shape}')
        _synchronize(self.device)
        started = time.perf_counter()
        batch = self.torch.from_numpy(sample).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            # BaseRecognizer._forward unwraps RecognizerGCN.extract_feat and
            # returns only the head logits in tensor mode.
            logits = self.model(batch, mode='tensor', stage='head')
            if not self.torch.is_tensor(logits):
                raise RuntimeError(
                    'RecognizerGCN head output contract changed: expected '
                    f'a tensor, got {type(logits)}')
            probabilities = self.torch.softmax(logits, dim=-1)
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        if probabilities.shape != (1, NUM_CLASSES):
            raise RuntimeError(
                f'{self.stream} model returned {probabilities.shape}, '
                f'expected (1, {NUM_CLASSES})')
        scores = probabilities[0].detach().cpu().numpy().astype(np.float64)
        if not np.allclose(scores.sum(), 1.0, atol=1e-6):
            raise RuntimeError(f'{self.stream} scores do not sum to one')
        return StreamOutput(
            scores=scores, latency_ms=latency_ms,
            input_shape=tuple(batch.shape))

    def __call__(
            self, keypoints: np.ndarray, keypoint_scores: np.ndarray,
            image_shape: tuple[int, int]) -> tuple[StreamOutput, float]:
        started = time.perf_counter()
        sample = self.preprocessor(
            keypoints, keypoint_scores, image_shape).data
        preprocess_ms = (time.perf_counter() - started) * 1000
        return self.infer_preprocessed(sample), preprocess_ms


class JointMotionFusion:
    """Two frozen ST-GCN streams with the validated fixed 0.5/0.5 fusion."""

    def __init__(
            self, joint_checkpoint: str | Path = JOINT_CHECKPOINT,
            motion_checkpoint: str | Path = MOTION_CHECKPOINT,
            config: str | Path = JOINT_CONFIG, device: str = 'cuda:0',
            project_root: str | Path | None = None):
        if (FUSION_JOINT_WEIGHT != 0.5
                or FUSION_MOTION_WEIGHT != 0.5):
            raise RuntimeError('only frozen 0.5/0.5 fusion is permitted')
        self.joint = STGCNStream(
            'joint', joint_checkpoint, config=config, device=device,
            project_root=project_root)
        self.motion = STGCNStream(
            'joint_motion', motion_checkpoint, config=config, device=device,
            project_root=project_root)

    def __call__(
            self, keypoints: np.ndarray, keypoint_scores: np.ndarray,
            image_shape: tuple[int, int]) -> FusionOutput:
        joint, joint_preprocess_ms = self.joint(
            keypoints, keypoint_scores, image_shape)
        motion, motion_preprocess_ms = self.motion(
            keypoints, keypoint_scores, image_shape)
        if joint.input_shape != motion.input_shape:
            raise RuntimeError(
                f'stream input shapes differ: {joint.input_shape} vs '
                f'{motion.input_shape}')
        started = time.perf_counter()
        fusion = (
            FUSION_JOINT_WEIGHT * joint.scores
            + FUSION_MOTION_WEIGHT * motion.scores)
        fusion_ms = (time.perf_counter() - started) * 1000
        if fusion.shape != (NUM_CLASSES,):
            raise RuntimeError(f'fusion returned invalid shape {fusion.shape}')
        if not np.allclose(fusion.sum(), 1.0, atol=1e-6):
            raise RuntimeError('fusion scores do not sum to one')
        return FusionOutput(
            joint_scores=joint.scores,
            motion_scores=motion.scores,
            fusion_scores=fusion,
            joint_ms=joint.latency_ms,
            motion_ms=motion.latency_ms,
            fusion_ms=fusion_ms,
            joint_preprocess_ms=joint_preprocess_ms,
            motion_preprocess_ms=motion_preprocess_ms,
            input_shape=joint.input_shape,
        )


class EMASmoother:
    """Probability-space exponential moving average without learned state."""

    def __init__(self, momentum: float = 0.8):
        if not 0 <= momentum < 1:
            raise ValueError('EMA momentum must be in [0, 1)')
        self.momentum = momentum
        self.value: np.ndarray | None = None

    def update(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (NUM_CLASSES,):
            raise ValueError(f'expected {NUM_CLASSES} scores, got {scores.shape}')
        if self.value is None:
            self.value = scores.copy()
        else:
            self.value = (
                self.momentum * self.value
                + (1 - self.momentum) * scores)
        self.value /= self.value.sum()
        return self.value.copy()


def top_k(scores: np.ndarray, k: int = 5) -> list[dict]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (NUM_CLASSES,):
        raise ValueError(f'expected {NUM_CLASSES} scores, got {scores.shape}')
    if not 1 <= k <= NUM_CLASSES:
        raise ValueError(f'k must be in [1, {NUM_CLASSES}]')
    indices = np.argsort(-scores, kind='stable')[:k]
    return [
        {
            'class_id': int(index),
            'action': NTU60_CLASS_NAMES[index],
            'probability': float(scores[index]),
        }
        for index in indices
    ]

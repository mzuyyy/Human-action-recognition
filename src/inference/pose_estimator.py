"""Lightweight person detection and official MMPose COCO-17 inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    DETECTOR_NAME,
    MMPOSE_CHECKPOINT,
    MMPOSE_CONFIG_NAME,
    MMPOSE_VERSION,
    NUM_JOINTS,
    POSE_MODEL_NAME,
)


def _synchronize(device: str) -> None:
    if device.startswith('cuda'):
        import torch

        torch.cuda.synchronize()


def resolve_mmpose_config(explicit: str | Path | None = None) -> Path:
    """Resolve the packaged MMPose config without relying on the CWD."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    import mmpose

    package = Path(mmpose.__file__).resolve().parent
    candidates = (
        package / '.mim' / MMPOSE_CONFIG_NAME,
        package.parent / MMPOSE_CONFIG_NAME,
        Path('/kaggle/working/mmpose') / MMPOSE_CONFIG_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'MMPose MobileNetV2 config was not packaged; searched: '
        f'{[str(path) for path in candidates]}')


@dataclass(frozen=True)
class Detection:
    bbox: np.ndarray | None
    score: float
    latency_ms: float


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    bbox: np.ndarray
    detection_score: float
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    detection_ms: float
    pose_ms: float

    @property
    def detected(self) -> bool:
        return bool(self.detection_score > 0)


class TorchvisionPersonDetector:
    """COCO person detector that does not require compiled MMCV ops."""

    name = DETECTOR_NAME

    def __init__(self, device: str = 'cuda:0', score_threshold: float = 0.5):
        if not 0 < score_threshold < 1:
            raise ValueError('detector score threshold must be in (0, 1)')
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        self.torch = torch
        self.device = device
        self.score_threshold = score_threshold
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, progress=True).to(device).eval()

    def __call__(self, frame_bgr: np.ndarray) -> Detection:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f'expected BGR image HxWx3, got {frame_bgr.shape}')
        _synchronize(self.device)
        started = time.perf_counter()
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        image = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        image = image.to(self.device)
        with self.torch.inference_mode():
            output = self.model([image])[0]
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000

        boxes = output['boxes'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy()
        keep = np.flatnonzero((labels == 1) & (scores >= self.score_threshold))
        if keep.size == 0:
            return Detection(None, 0.0, latency_ms)
        candidate_boxes = boxes[keep]
        areas = np.maximum(0, candidate_boxes[:, 2] - candidate_boxes[:, 0])
        areas *= np.maximum(0, candidate_boxes[:, 3] - candidate_boxes[:, 1])
        # The largest confident person is the dominant slot. No tracker is
        # introduced in this milestone.
        selected_local = int(np.argmax(areas * scores[keep]))
        selected = int(keep[selected_local])
        return Detection(
            boxes[selected].astype(np.float32), float(scores[selected]),
            latency_ms)


class MMPoseTopDownEstimator:
    """Official MMPose MobileNetV2 COCO-17 top-down estimator."""

    name = POSE_MODEL_NAME
    checkpoint = MMPOSE_CHECKPOINT

    def __init__(
            self, device: str = 'cuda:0', config: str | Path | None = None,
            checkpoint: str = MMPOSE_CHECKPOINT):
        import mmpose
        from mmengine.config import Config
        from mmpose.apis import inference_topdown, init_model

        if mmpose.__version__ != MMPOSE_VERSION:
            raise RuntimeError(
                f'MMPose {MMPOSE_VERSION} is required, got '
                f'{mmpose.__version__}')
        self.device = device
        self.inference_topdown = inference_topdown
        self.config = resolve_mmpose_config(config)
        self.checkpoint = checkpoint
        config_object = Config.fromfile(str(self.config))
        # Test-time horizontal flip doubles pose compute. Disable it through
        # the model's official test option; the pretrained checkpoint and
        # COCO joint definition stay unchanged.
        config_object.model.test_cfg.flip_test = False
        self.model = init_model(
            config_object, checkpoint, device=device)
        self._assert_coco_layout()

    def _assert_coco_layout(self) -> None:
        metadata = self.model.dataset_meta
        names_by_id = metadata.get('keypoint_id2name', {})
        if isinstance(names_by_id, dict):
            names = tuple(
                str(names_by_id[index] if index in names_by_id
                    else names_by_id[str(index)])
                for index in range(NUM_JOINTS)
            )
        else:
            names = tuple(names_by_id)
        if names != COCO_KEYPOINT_NAMES:
            raise RuntimeError(
                'pose model joint order is not the required COCO-17 layout: '
                f'{names}')

    def __call__(
            self, frame_bgr: np.ndarray,
            bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        _synchronize(self.device)
        started = time.perf_counter()
        results = self.inference_topdown(
            self.model, frame_bgr, bboxes=bbox, bbox_format='xyxy')
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        if len(results) != 1:
            raise RuntimeError(
                f'expected one top-down pose result, got {len(results)}')
        instances = results[0].pred_instances
        keypoints = np.asarray(instances.keypoints, dtype=np.float32)
        scores = np.asarray(instances.keypoint_scores, dtype=np.float32)
        if keypoints.shape == (1, NUM_JOINTS, 2):
            keypoints = keypoints[0]
        if scores.shape == (1, NUM_JOINTS):
            scores = scores[0]
        if keypoints.shape != (NUM_JOINTS, 2):
            raise RuntimeError(
                f'MMPose returned keypoints shape {keypoints.shape}, '
                f'expected ({NUM_JOINTS}, 2)')
        if scores.shape != (NUM_JOINTS,):
            raise RuntimeError(
                f'MMPose returned score shape {scores.shape}, '
                f'expected ({NUM_JOINTS},)')
        return keypoints, np.clip(scores, 0, 1), latency_ms


class DominantPersonPosePipeline:
    """Detect and pose-estimate one dominant person in each frame."""

    def __init__(
            self, device: str = 'cuda:0', detector_threshold: float = 0.5,
            pose_config: str | Path | None = None,
            pose_checkpoint: str = MMPOSE_CHECKPOINT):
        self.detector = TorchvisionPersonDetector(
            device=device, score_threshold=detector_threshold)
        self.pose = MMPoseTopDownEstimator(
            device=device, config=pose_config, checkpoint=pose_checkpoint)

    def __call__(self, frame_bgr: np.ndarray, frame_index: int) -> PoseFrame:
        detection = self.detector(frame_bgr)
        if detection.bbox is None:
            return PoseFrame(
                frame_index=frame_index,
                bbox=np.zeros(4, dtype=np.float32),
                detection_score=0.0,
                keypoints=np.zeros((NUM_JOINTS, 2), dtype=np.float32),
                keypoint_scores=np.zeros(NUM_JOINTS, dtype=np.float32),
                detection_ms=detection.latency_ms,
                pose_ms=0.0,
            )
        keypoints, scores, pose_ms = self.pose(frame_bgr, detection.bbox)
        return PoseFrame(
            frame_index=frame_index,
            bbox=detection.bbox,
            detection_score=detection.score,
            keypoints=keypoints,
            keypoint_scores=scores,
            detection_ms=detection.latency_ms,
            pose_ms=pose_ms,
        )


def draw_pose(
        frame_bgr: np.ndarray, pose: PoseFrame,
        keypoint_threshold: float = 0.3) -> np.ndarray:
    """Draw dominant-person bbox and COCO skeleton on a BGR frame."""
    import cv2

    canvas = frame_bgr.copy()
    if not pose.detected:
        return canvas
    x1, y1, x2, y2 = np.rint(pose.bbox).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 2)
    for first, second in COCO_SKELETON:
        if (pose.keypoint_scores[first] >= keypoint_threshold
                and pose.keypoint_scores[second] >= keypoint_threshold):
            point_a = tuple(np.rint(pose.keypoints[first]).astype(int))
            point_b = tuple(np.rint(pose.keypoints[second]).astype(int))
            cv2.line(canvas, point_a, point_b, (0, 210, 255), 2,
                     lineType=cv2.LINE_AA)
    for keypoint, score in zip(pose.keypoints, pose.keypoint_scores):
        if score >= keypoint_threshold:
            point = tuple(np.rint(keypoint).astype(int))
            cv2.circle(canvas, point, 3, (30, 30, 255), -1,
                       lineType=cv2.LINE_AA)
    return canvas

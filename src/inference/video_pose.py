"""Video decoding and dominant-person pose extraction helpers."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from .constants import (
    COCO_KEYPOINT_NAMES,
    DETECTOR_NAME,
    NUM_JOINTS,
    POSE_MODEL_NAME,
)
from .pose_estimator import DominantPersonPosePipeline, PoseFrame


def resolve_device(requested: str) -> str:
    if requested != 'auto':
        return requested
    import torch

    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def extract_video_pose(
        video_path: str | Path, pipeline: DominantPersonPosePipeline,
        progress_interval: int = 100) -> dict:
    """Extract one COCO-17 pose slot for every decoded video frame."""
    import cv2
    import mmpose
    import torch
    import torchvision

    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open video: {video_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            f'invalid video metadata: fps={fps}, size={width}x{height}')

    frames: list[PoseFrame] = []
    frame_index = 0
    extraction_started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(pipeline(frame, frame_index))
        frame_index += 1
        if progress_interval and frame_index % progress_interval == 0:
            print(
                f'pose extraction: {frame_index}/'
                f'{reported_frames if reported_frames > 0 else "?"}',
                flush=True)
    capture.release()
    extraction_wall_time_sec = time.perf_counter() - extraction_started
    if not frames:
        raise RuntimeError(f'video contains no decodable frames: {video_path}')

    keypoint = np.stack([frame.keypoints for frame in frames])[None]
    keypoint_score = np.stack(
        [frame.keypoint_scores for frame in frames])[None]
    bboxes = np.stack([frame.bbox for frame in frames])
    detection_scores = np.asarray(
        [frame.detection_score for frame in frames], dtype=np.float32)
    detection_ms = np.asarray(
        [frame.detection_ms for frame in frames], dtype=np.float64)
    pose_ms = np.asarray(
        [frame.pose_ms for frame in frames], dtype=np.float64)
    detected = detection_scores > 0
    frame_records = [
        {
            'frame_index': int(frame.frame_index),
            'person_id': 0 if frame.detected else -1,
            'person_slot': 0,
            'bbox_xyxy': frame.bbox.tolist(),
            'detection_score': float(frame.detection_score),
            'keypoints': frame.keypoints.tolist(),
            'keypoint_scores': frame.keypoint_scores.tolist(),
        }
        for frame in frames
    ]
    return {
        'format_version': 1,
        'metadata': {
            'source_video': str(video_path),
            'fps': fps,
            'frame_count': len(frames),
            'reported_frame_count': reported_frames,
            'width': width,
            'height': height,
            'layout': 'coco',
            'num_joints': NUM_JOINTS,
            'num_person_slots': 1,
            'keypoint_names': list(COCO_KEYPOINT_NAMES),
            'detector': DETECTOR_NAME,
            'pose_model': POSE_MODEL_NAME,
            'mmpose_version': mmpose.__version__,
            'torch_version': torch.__version__,
            'torchvision_version': torchvision.__version__,
        },
        'keypoint': keypoint.astype(np.float32),
        'keypoint_score': keypoint_score.astype(np.float32),
        'bbox': bboxes.astype(np.float32),
        'detection_score': detection_scores,
        'frames': frame_records,
        'latency': {
            'detection_ms_per_frame': float(np.mean(detection_ms)),
            'pose_ms_per_frame': float(np.mean(pose_ms[detected]))
            if np.any(detected) else 0.0,
            'pose_ms_per_all_frames': float(np.mean(pose_ms)),
            'detected_frames': int(np.sum(detected)),
            'detection_rate': float(np.mean(detected)),
            # Excludes one-time model initialization/download and includes
            # video decoding plus per-frame detection/pose work.
            'pose_extraction_wall_time_sec': extraction_wall_time_sec,
        },
    }


def save_pose_payload(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    with temporary.open('wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return path


def load_pose_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get('format_version') != 1:
        raise ValueError(f'unsupported pose payload: {path}')
    return payload

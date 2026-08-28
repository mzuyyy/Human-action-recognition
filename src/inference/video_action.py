"""Sliding-window action inference, profiling, and video annotation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .action_recognizer import EMASmoother, JointMotionFusion, top_k
from .constants import CLIP_LEN, NTU60_CLASS_NAMES
from .pose_estimator import PoseFrame, draw_pose
from .skeleton_preprocessor import pose_arrays_from_payload, window_slices


def infer_pose_windows(
        payload: dict, model: JointMotionFusion,
        window_frames: int = CLIP_LEN, stride: int = CLIP_LEN // 2,
        ema_momentum: float = 0.8) -> list[dict]:
    """Run raw and EMA-smoothed fixed fusion over overlapping pose windows."""
    keypoints, keypoint_scores, image_shape = pose_arrays_from_payload(payload)
    windows = window_slices(
        keypoints.shape[1], window_frames=window_frames, stride=stride)
    smoother = EMASmoother(ema_momentum)
    records = []
    for index, (start, end) in enumerate(windows):
        output = model(
            keypoints[:, start:end], keypoint_scores[:, start:end],
            image_shape)
        smoothed = smoother.update(output.fusion_scores)
        raw_top1 = top_k(output.fusion_scores, 1)[0]
        smoothed_top1 = top_k(smoothed, 1)[0]
        records.append({
            'window_index': index,
            'start_frame': start,
            'end_frame_exclusive': end,
            'center_frame': (start + end - 1) / 2,
            'raw_top1': raw_top1,
            'smoothed_top1': smoothed_top1,
            'raw_top5': top_k(output.fusion_scores, 5),
            'smoothed_top5': top_k(smoothed, 5),
            'joint_scores': output.joint_scores.tolist(),
            'motion_scores': output.motion_scores.tolist(),
            'fusion_scores': output.fusion_scores.tolist(),
            'smoothed_scores': smoothed.tolist(),
            'model_input_shape': list(output.input_shape),
            'latency_ms': {
                'joint_preprocessing': output.joint_preprocess_ms,
                'motion_preprocessing': output.motion_preprocess_ms,
                'joint_stgcn': output.joint_ms,
                'motion_stgcn': output.motion_ms,
                'fusion': output.fusion_ms,
            },
        })
        print(
            f"window {index + 1}/{len(windows)} [{start}:{end}]: "
            f"raw={raw_top1['action']} ({raw_top1['probability']:.3f}), "
            f"ema={smoothed_top1['action']} "
            f"({smoothed_top1['probability']:.3f})",
            flush=True)
    return records


def build_latency_report(
        payload: dict, windows: list[dict], stride: int,
        action_inference_wall_time_sec: float) -> dict:
    if not windows:
        raise ValueError('latency report requires at least one window')
    stage_keys = (
        'joint_preprocessing', 'motion_preprocessing',
        'joint_stgcn', 'motion_stgcn', 'fusion')
    means = {
        key: float(np.mean([
            window['latency_ms'][key] for window in windows
        ]))
        for key in stage_keys
    }
    pose_latency = payload['latency']
    detection = float(pose_latency['detection_ms_per_frame'])
    pose = float(pose_latency['pose_ms_per_all_frames'])
    skeleton = means['joint_preprocessing'] + means['motion_preprocessing']
    action_per_window = (
        skeleton + means['joint_stgcn'] + means['motion_stgcn']
        + means['fusion'])
    # Action inference occurs once per `stride` frames, so amortize its window
    # cost when presenting a pipeline FPS estimate.
    amortized_action = action_per_window / stride
    total_per_frame = detection + pose + amortized_action
    pose_wall = float(
        pose_latency.get('pose_extraction_wall_time_sec', 0.0))
    measured_wall = pose_wall + action_inference_wall_time_sec
    frame_count = int(payload['metadata']['frame_count'])
    return {
        'units': {
            'detection': 'ms/frame',
            'pose': 'ms/frame (zero on frames without a detection)',
            'skeleton_preprocessing': 'ms/window',
            'joint_stgcn': 'ms/window',
            'motion_stgcn': 'ms/window',
            'fusion': 'ms/window',
        },
        'detection_ms_per_frame': detection,
        'pose_ms_per_frame': pose,
        'skeleton_preprocessing_ms_per_window': skeleton,
        'joint_preprocessing_ms_per_window': means['joint_preprocessing'],
        'motion_preprocessing_ms_per_window': means['motion_preprocessing'],
        'joint_stgcn_ms_per_window': means['joint_stgcn'],
        'motion_stgcn_ms_per_window': means['motion_stgcn'],
        'fusion_ms_per_window': means['fusion'],
        'action_total_ms_per_window': action_per_window,
        'window_stride_frames': stride,
        'amortized_action_ms_per_frame': amortized_action,
        'estimated_total_ms_per_frame': total_per_frame,
        'estimated_pipeline_fps': 1000 / total_per_frame
        if total_per_frame > 0 else None,
        'pose_extraction_wall_time_sec': pose_wall,
        'action_inference_wall_time_sec': action_inference_wall_time_sec,
        'measured_pipeline_wall_time_sec': measured_wall,
        'measured_pipeline_ms_per_frame': 1000 * measured_wall / frame_count,
        'measured_pipeline_fps': frame_count / measured_wall
        if measured_wall > 0 else None,
        'frame_count': frame_count,
        'window_count': len(windows),
        'detected_frames': int(pose_latency['detected_frames']),
        'detection_rate': float(pose_latency['detection_rate']),
    }


def save_json(data: dict | list, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def _window_for_frame(windows: list[dict], frame_index: int) -> dict:
    return min(
        windows,
        key=lambda item: abs(float(item['center_frame']) - frame_index))


def annotate_video(
        video_path: str | Path, output_path: str | Path, payload: dict,
        windows: list[dict], pipeline_fps: float | None,
        show_stream_scores: bool = False,
        keypoint_threshold: float = 0.3) -> Path:
    """Render the extracted skeleton and nearest smoothed window prediction."""
    import cv2

    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot reopen video: {video_path}')
    fps = float(payload['metadata']['fps'])
    width = int(payload['metadata']['width'])
    height = int(payload['metadata']['height'])
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
        (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f'cannot create output video: {output_path}')

    keypoints = np.asarray(payload['keypoint'])[0]
    keypoint_scores = np.asarray(payload['keypoint_score'])[0]
    bboxes = np.asarray(payload['bbox'])
    detection_scores = np.asarray(payload['detection_score'])
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index >= len(keypoints):
            writer.release()
            capture.release()
            raise RuntimeError('decoded annotation frames exceed pose frames')
        pose = PoseFrame(
            frame_index=frame_index,
            bbox=bboxes[frame_index],
            detection_score=float(detection_scores[frame_index]),
            keypoints=keypoints[frame_index],
            keypoint_scores=keypoint_scores[frame_index],
            detection_ms=0.0,
            pose_ms=0.0,
        )
        canvas = draw_pose(
            frame, pose, keypoint_threshold=keypoint_threshold)
        prediction = _window_for_frame(windows, frame_index)
        raw = prediction['raw_top1']
        smooth = prediction['smoothed_top1']
        lines = [
            'Person: dominant slot 0' if pose.detected else 'Person: not detected',
            f"Action: {smooth['action']}",
            f"Fusion confidence: {smooth['probability']:.3f}",
            f"Raw: {raw['action']} ({raw['probability']:.3f})",
        ]
        if pipeline_fps is not None:
            lines.append(f'Pipeline FPS: {pipeline_fps:.2f}')
        if show_stream_scores:
            joint = np.asarray(prediction['joint_scores'])
            motion = np.asarray(prediction['motion_scores'])
            joint_id = int(np.argmax(joint))
            motion_id = int(np.argmax(motion))
            lines.extend((
                f'Joint: {NTU60_CLASS_NAMES[joint_id]} ({joint[joint_id]:.3f})',
                (f'Motion: {NTU60_CLASS_NAMES[motion_id]} '
                 f'({motion[motion_id]:.3f})'),
            ))
        overlay_height = 20 + 29 * len(lines)
        cv2.rectangle(
            canvas, (8, 8), (min(width - 8, 780), overlay_height),
            (15, 15, 15), -1)
        for line_index, line in enumerate(lines):
            cv2.putText(
                canvas, line, (20, 38 + 29 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2,
                lineType=cv2.LINE_AA)
        writer.write(canvas)
        frame_index += 1
    capture.release()
    writer.release()
    if frame_index != len(keypoints):
        raise RuntimeError(
            f'annotated {frame_index} frames but pose has {len(keypoints)}')
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f'annotated output was not created: {output_path}')
    return output_path

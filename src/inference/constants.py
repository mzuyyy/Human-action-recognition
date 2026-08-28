"""Frozen layouts, labels, and model defaults for video inference."""

from scripts import stgcn_evaluation_common as _evaluation

NTU60_CLASS_NAMES = _evaluation.NTU60_CLASS_NAMES

NUM_JOINTS = 17
NUM_CLASSES = 60
NUM_PERSONS = 2
CLIP_LEN = 100
NUM_CLIPS = 1
UNIFORM_SAMPLE_SEED = 255

COCO_KEYPOINT_NAMES = (
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
)

# COCO-17 edges. Each tuple is zero-based and follows COCO ordering above.
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

MMPOSE_VERSION = '1.3.2'
MMPOSE_CONFIG_NAME = (
    'configs/body_2d_keypoint/topdown_heatmap/coco/'
    'td-hm_mobilenetv2_8xb64-210e_coco-256x192.py'
)
MMPOSE_CHECKPOINT = (
    'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/'
    'topdown_heatmap/coco/'
    'td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth'
)
POSE_MODEL_NAME = 'MMPose MobileNetV2 SimpleBaseline 256x192 (COCO)'
DETECTOR_NAME = 'TorchVision Faster R-CNN MobileNetV3 320 FPN (COCO)'

JOINT_CONFIG = 'configs/stgcn_ntu60_xsub_baseline.py'
JOINT_CHECKPOINT = (
    'artifacts/checkpoints/stgcn_joint_ntu60_xsub_best.pth'
)
MOTION_CHECKPOINT = (
    'artifacts/checkpoints/stgcn_joint_motion_ntu60_xsub_best.pth'
)

FUSION_JOINT_WEIGHT = 0.5
FUSION_MOTION_WEIGHT = 0.5

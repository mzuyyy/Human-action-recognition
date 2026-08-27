# ruff: noqa: C408
"""Controlled ST-GCN Joint Motion run: 16 outer x 5 repeats = 80 passes."""

_base_ = './stgcn_ntu60_xsub_40e.py'

work_dir = 'work_dirs/stgcn_ntu60_xsub_joint_motion_80e'

# This is MMAction2's official Joint Motion representation. GenSkeFeat invokes
# ToMotion, which calculates joint[t + 1] - joint[t] along the time axis.
# Every other transform and sampling setting matches the Joint baseline.
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]

# Repeat these references explicitly: overriding a top-level pipeline variable
# does not retroactively replace a pipeline already nested in a base config.
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(times=5, dataset=dict(pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(pipeline=test_pipeline))

custom_hooks = [
    dict(
        type='STGCNEpochMetricsHook',
        repeat_times=5,
        strict_epoch_sequence=True,
        priority='LOWEST')
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=16,
    val_begin=1,
    val_interval=1)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True)
]

load_from = None
resume = False
randomness = dict(seed=42, diff_rank_seed=False, deterministic=False)

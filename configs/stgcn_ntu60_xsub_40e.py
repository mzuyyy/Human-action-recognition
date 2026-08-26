"""ST-GCN Joint baseline: 8 outer epochs x RepeatDataset(5) = 40 passes."""

_base_ = './stgcn_ntu60_xsub_baseline.py'

work_dir = 'work_dirs/stgcn_ntu60_xsub_40e'

custom_imports = dict(
    imports=['hooks.stgcn_epoch_metrics_hook'], allow_failed_imports=False)
custom_hooks = [
    dict(type='STGCNEpochMetricsHook', repeat_times=5, priority='LOWEST')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(times=5))
val_dataloader = dict(
    batch_size=16, num_workers=2, persistent_workers=True)
test_dataloader = dict(num_workers=2, persistent_workers=True)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=8,
    val_begin=1,
    val_interval=1)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=8,
        by_epoch=True,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_best='acc/top1',
        rule='greater',
        save_last=True,
        max_keep_ckpts=-1))

load_from = None
resume = False
randomness = dict(seed=42, diff_rank_seed=False, deterministic=False)

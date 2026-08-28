"""Resume epoch 8 through 16: 16 outer x RepeatDataset(5) = 80 passes."""

_base_ = './stgcn_ntu60_xsub_40e.py'

work_dir = 'work_dirs/stgcn_ntu60_xsub_80e_resume'

custom_hooks = [
    dict(
        type='STGCNEpochMetricsHook',
        repeat_times=5,
        resume_cosine_t_max=16,
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

# The notebook supplies the verified external checkpoint with
# `--resume CHECKPOINT`; resume-first mode is intentional.
load_from = None
resume = True

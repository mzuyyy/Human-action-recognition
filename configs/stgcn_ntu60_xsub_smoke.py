# Kaggle single-GPU SMOKE TEST — inherits everything from the baseline.
# Goal: prove the end-to-end pipeline runs (loss finite, backward OK,
# checkpoint written), NOT accuracy. Run with:
#   python /kaggle/working/mmaction2/tools/train.py configs/stgcn_ntu60_xsub_smoke.py \
#       --work-dir work_dirs/stgcn_smoke_test --seed 42

_base_ = './stgcn_ntu60_xsub_baseline.py'

train_dataloader = dict(
    batch_size=16,          # OOM? -> 8 -> 4 (do not shrink the model)
    num_workers=2,
    persistent_workers=True,
    dataset=dict(times=1))  # RepeatDataset: disable epoch repetition for smoke

val_dataloader = dict(batch_size=16, num_workers=2)
test_dataloader = dict(num_workers=2)

train_cfg = dict(max_epochs=1, val_begin=1, val_interval=1)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=1,
        by_epoch=True,
        convert_to_iter_based=True)
]

# lr=0.1 targets global batch 128; single GPU + 1 epoch -> drop for stability.
optim_wrapper = dict(optimizer=dict(lr=0.01))

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=20))

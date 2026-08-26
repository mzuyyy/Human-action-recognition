# Human Action Recognition — NTU60 skeleton baseline

Skeleton-based human action recognition on **NTU RGB+D 60** (preprocessed 2D
skeletons), **PyTorch + MMAction2**, model **ST-GCN**, split **Cross-Subject (xsub)**.

The smoke pipeline and the first 8 outer epochs are complete. The active run
resumes that ST-GCN Joint checkpoint through outer epoch 16 with
`RepeatDataset(times=5)`, giving 80 effective passes over the NTU60
cross-subject training split.

## Structure

```text
├── configs/
│   ├── stgcn_ntu60_xsub_baseline.py   # flattened copy of the official MMAction2 config (80 epochs)
│   ├── stgcn_ntu60_xsub_smoke.py      # completed single-GPU smoke test
│   ├── stgcn_ntu60_xsub_40e.py        # completed 8 outer x 5 repeats
│   └── stgcn_ntu60_xsub_80e_resume.py # resume to 16 outer x 5 repeats
├── data/skeleton/ntu60_2d.pkl         # NOT committed — downloaded by the notebook (~1.4 GB)
├── scripts/
│   ├── inspect_ntu60.py               # Task 4 — pickle stats, shapes, acceptance checks
│   └── visualize_skeleton.py          # Task 5 — COCO-17 skeletons on white canvas
├── hooks/
│   └── stgcn_epoch_metrics_hook.py     # loss/LR/accuracy/GPU/time per outer epoch
├── notebooks/
│   ├── 01_baseline_stgcn.ipynb        # completed smoke pipeline
│   └── 02_train_stgcn_40e.ipynb       # resume epoch 8 -> 16 (80 effective)
├── artifacts/
│   ├── environment.txt                # filled by notebook Task 1
│   ├── dataset_stats.json             # written by inspect script (--json-out)
│   └── skeleton_samples/              # visualization PNGs
├── work_dirs/stgcn_smoke_test/        # checkpoints + logs (NOT committed)
├── work_dirs/stgcn_ntu60_xsub_40e/    # 40e checkpoints + logs (NOT committed)
├── work_dirs/stgcn_ntu60_xsub_80e_resume/ # resumed checkpoints + logs (NOT committed)
└── requirements.txt
```

## Run on Kaggle (recommended)

1. New Kaggle notebook → **Settings**: Accelerator = `GPU T4 / P100`, Internet = `ON`.
2. Upload/import this repo's files (or clone your fork) so the project sits at
   `/kaggle/working/ntu-action-recognition` — the first notebook cell handles cloning.
3. Run `notebooks/01_baseline_stgcn.ipynb` top-to-bottom:
   - env dump → `artifacts/environment.txt`
   - installs MMEngine and the pinned `mmcv-lite==2.1.0` wheel (ST-GCN does not
     need MMCV's compiled ops, and current Kaggle Python 3.12 images have no
     matching full-MMCV wheel)
   - installs MMAction2 v1.2.0 from its source tree in editable mode (required
     because its regular wheel omits the `localizers/drn` namespace directory)
   - disables MMAction2's unused optional multimodal registry, avoiding its
     incompatible ViNLU import against Kaggle's preinstalled Transformers
   - downloads `ntu60_2d.pkl` from the official OpenMMLab release
     (`https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl`),
     with `/kaggle/input` symlink fallback when Internet is off
   - dataset inspection → acceptance checks must all PASS
   - skeleton visualization → eyeball it; stop if skeletons look broken
   - dataloader batch sanity → NaN/Inf/label-range checks
   - 1-epoch smoke training with GPU monitor → checkpoint in `work_dirs/stgcn_smoke_test`
   - Top-1/Top-5 validation on `xsub_val`
   - report → `artifacts/smoke_report.txt`

OOM during smoke training? Lower `batch_size` in
`configs/stgcn_ntu60_xsub_smoke.py`: 16 → 8 → 4. Never shrink the model.

## Local script usage

Both scripts only need `numpy` (+ `matplotlib` for viz):

```bash
python scripts/inspect_ntu60.py --ann-file data/skeleton/ntu60_2d.pkl --json-out artifacts/dataset_stats.json
python scripts/visualize_skeleton.py --ann-file data/skeleton/ntu60_2d.pkl --out-dir artifacts/skeleton_samples
```

## Resume the baseline through 80 effective epochs

Attach the Kaggle Model containing
`/kaggle/input/models/duymaingoc/resume/pytorch/default/1/best_acc_top1_epoch_8.pth`,
then run `notebooks/02_train_stgcn_40e.ipynb` top-to-bottom. The notebook still
downloads or links `ntu60_2d.pkl`, because true training resume requires the
training and validation datasets. It restores every available MMEngine state,
runs only outer epochs 9–16, and records the resolved config, logs, checkpoints,
and final report under `work_dirs/stgcn_ntu60_xsub_80e_resume/`. If the
published checkpoint omits or has an incompatible optimizer state, the hook
keeps its model/epoch progress but initializes a new optimizer at the correct
mid-schedule LR and records that fallback in `resume_state.json`.

Equivalent command:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python /kaggle/working/mmaction2/tools/train.py configs/stgcn_ntu60_xsub_80e_resume.py \
    --work-dir work_dirs/stgcn_ntu60_xsub_80e_resume --seed 42 \
    --resume /kaggle/input/models/duymaingoc/resume/pytorch/default/1/best_acc_top1_epoch_8.pth
```

(`tools/train.py` comes from the cloned MMAction2 repo; keep the current working
directory at the project root so the config and output paths resolve correctly.)

The resumed config uses batch size 64, `max_epochs=16`, and a cosine schedule
covering all 16 outer epochs. When epoch 8 is loaded, the custom hook expands
the checkpoint's completed 8-epoch cosine state so training does not continue
with a near-zero learning rate. If the resumed run itself is interrupted, the
notebook prefers the checkpoint named by the restored work directory's
`last_checkpoint` marker. The committed config keeps `load_from=None` and
`resume=False`; the notebook supplies the explicit `--resume` checkpoint. It
also sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the training subprocess so
MMEngine 0.10.x can restore this trusted full checkpoint under PyTorch 2.6+.

## Out of scope (this milestone)

MMPose/YOLO/tracking, webcam inference, TensorRT/ONNX export, ST-GCN++,
NTU120, Kinetics400, hyperparameter tuning. Accuracy is irrelevant until the
baseline runs end-to-end.

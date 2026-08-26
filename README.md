# Human Action Recognition — NTU60 skeleton baseline

Skeleton-based human action recognition on **NTU RGB+D 60** (preprocessed 2D
skeletons), **PyTorch + MMAction2**, model **ST-GCN**, split **Cross-Subject (xsub)**.

The smoke pipeline is complete. The active experiment is the ST-GCN Joint
baseline trained for 8 outer epochs with `RepeatDataset(times=5)`, giving 40
effective passes over the NTU60 cross-subject training split.

## Structure

```text
├── configs/
│   ├── stgcn_ntu60_xsub_baseline.py   # flattened copy of the official MMAction2 config (80 epochs)
│   ├── stgcn_ntu60_xsub_smoke.py      # completed single-GPU smoke test
│   └── stgcn_ntu60_xsub_40e.py        # 8 outer x 5 repeats = 40 effective epochs
├── data/skeleton/ntu60_2d.pkl         # NOT committed — downloaded by the notebook (~1.4 GB)
├── scripts/
│   ├── inspect_ntu60.py               # Task 4 — pickle stats, shapes, acceptance checks
│   └── visualize_skeleton.py          # Task 5 — COCO-17 skeletons on white canvas
├── hooks/
│   └── stgcn_epoch_metrics_hook.py     # loss/LR/accuracy/GPU/time per outer epoch
├── notebooks/
│   ├── 01_baseline_stgcn.ipynb        # completed smoke pipeline
│   └── 02_train_stgcn_40e.ipynb       # direct 40-effective-epoch training
├── artifacts/
│   ├── environment.txt                # filled by notebook Task 1
│   ├── dataset_stats.json             # written by inspect script (--json-out)
│   └── skeleton_samples/              # visualization PNGs
├── work_dirs/stgcn_smoke_test/        # checkpoints + logs (NOT committed)
├── work_dirs/stgcn_ntu60_xsub_40e/    # 40e checkpoints + logs (NOT committed)
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

## 40-effective-epoch baseline training

Run `notebooks/02_train_stgcn_40e.ipynb` top-to-bottom. It starts from scratch,
streams the full training log, saves every outer epoch and writes the resolved
config plus final epoch table/report under the dedicated experiment paths.

Equivalent command:

```bash
python /kaggle/working/mmaction2/tools/train.py configs/stgcn_ntu60_xsub_40e.py \
    --work-dir work_dirs/stgcn_ntu60_xsub_40e --seed 42
```

(`tools/train.py` comes from the cloned MMAction2 repo; keep the current working
directory at the project root so the config and output paths resolve correctly.)

After an interrupted session, restore the same work directory and set
`RESUME = True` in the training notebook. This adds MMEngine's `--resume` flag;
the committed config itself remains `load_from=None` and `resume=False` so a new
experiment never consumes the smoke-test checkpoints.

## Out of scope (this milestone)

MMPose/YOLO/tracking, webcam inference, TensorRT/ONNX export, ST-GCN++,
NTU120, Kinetics400, hyperparameter tuning. Accuracy is irrelevant until the
baseline runs end-to-end.

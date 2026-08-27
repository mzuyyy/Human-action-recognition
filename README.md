# Human Action Recognition — NTU60 skeleton baseline

Skeleton-based human action recognition on **NTU RGB+D 60** (preprocessed 2D
skeletons), **PyTorch + MMAction2**, model **ST-GCN**, split **Cross-Subject (xsub)**.

The ST-GCN Joint run is complete through outer epoch 16 with
`RepeatDataset(times=5)`, giving 80 effective passes over the NTU60
cross-subject training split. Its final logged validation result is Top-1
`0.8823` and Top-5 `0.9877`. The frozen Joint baseline is not trained further.
Experiment 2 is a controlled, from-scratch ST-GCN **Joint Motion** run using the
same settings and MMAction2's official temporal-motion feature.

## Structure

```text
├── configs/
│   ├── stgcn_ntu60_xsub_baseline.py   # flattened copy of the official MMAction2 config (80 epochs)
│   ├── stgcn_ntu60_xsub_smoke.py      # completed single-GPU smoke test
│   ├── stgcn_ntu60_xsub_40e.py        # completed 8 outer x 5 repeats
│   ├── stgcn_ntu60_xsub_80e_resume.py # completed resume to 16 x 5
│   └── stgcn_ntu60_xsub_joint_motion_80e.py # controlled experiment 2
├── data/skeleton/ntu60_2d.pkl         # NOT committed — downloaded by the notebook (~1.4 GB)
├── scripts/
│   ├── inspect_ntu60.py               # Task 4 — pickle stats, shapes, acceptance checks
│   ├── visualize_skeleton.py          # Task 5 — COCO-17 skeletons on white canvas
│   ├── stgcn_evaluation_common.py      # NTU60 names + log/checkpoint selection
│   ├── evaluate_stgcn_joint.py         # freeze best + independent full-val inference
│   ├── analyze_stgcn_results.py        # Joint confusion/confidence reports
│   └── analyze_joint_vs_motion.py      # full experiment-2 comparison
├── hooks/
│   └── stgcn_epoch_metrics_hook.py     # loss/LR/accuracy/GPU/time per outer epoch
├── notebooks/
│   ├── 01_baseline_stgcn.ipynb        # completed smoke pipeline
│   ├── 02_train_stgcn_40e.ipynb       # completed resume epoch 8 -> 16
│   ├── 03_evaluate_stgcn_joint.ipynb  # independent Joint evaluation
│   └── 04_train_stgcn_joint_motion_80e.ipynb # train/evaluate/compare
├── artifacts/
│   ├── environment.txt                # filled by notebook Task 1
│   ├── dataset_stats.json             # written by inspect script (--json-out)
│   ├── skeleton_samples/              # visualization PNGs
│   ├── checkpoints/                   # frozen best checkpoint (NOT committed)
│   ├── evaluation/                    # metrics, predictions, labels, scores
│   ├── analysis/                      # confusion/error/curve reports
│   ├── experiments/                   # frozen per-experiment results
│   ├── comparison/                    # Joint-vs-Motion evidence/report
│   └── readme/                        # compact GitHub-ready result files
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

## Independent evaluation and error analysis

Run `notebooks/03_evaluate_stgcn_joint.ipynb` after training completes. In the
same Kaggle session it reads the resume work directory directly. In a new
session, attach one Kaggle Model containing exactly this minimal pair in the
same directory:

```text
epoch_16.pth
epoch_metrics.jsonl
```

Keep the checkpoint filename as `epoch_16.pth`. The notebook recursively finds
one unique matching pair under `/kaggle/input`, links only those two read-only
uploads into `work_dirs/stgcn_ntu60_xsub_80e_resume/`, and downloads or links
`ntu60_2d.pkl` when needed. Other epoch checkpoints, `latest.pth`,
`training_console.log`, and `resume_state.json` are not required.

The evaluation pipeline:

1. parses `epoch_metrics.jsonl`, verifies that uploaded epoch 16 has the highest
   logged validation Top-1, and stops if the actual best checkpoint is missing;
2. copies it to
   `artifacts/checkpoints/stgcn_joint_ntu60_xsub_best.pth` without deleting the
   source;
3. performs fresh inference over all 16,487 `xsub_val` samples and stops if its
   Top-1 differs from the logged result by more than `0.002`;
4. exports per-sample predictions and NumPy arrays, then builds the normalized
   60-class confusion matrix, per-class accuracy, directed confusion pairs,
   confidence analysis, error hypotheses, training curve, and README files.

The analysis script refuses metrics not marked as an accepted independent
evaluation, so a stale prediction cache cannot be used after a discrepancy.
The training curve marks the resume boundary at effective epoch 40 and retains
the documented Top-1 discontinuity from effective epochs 45–80.

Equivalent commands inside the configured Kaggle environment are:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONPATH=/kaggle/working/mmaction2:/kaggle/working/ntu-action-recognition \
python scripts/evaluate_stgcn_joint.py \
    --work-dir work_dirs/stgcn_ntu60_xsub_40e \
    --work-dir work_dirs/stgcn_ntu60_xsub_80e_resume

python scripts/analyze_stgcn_results.py \
    --work-dir work_dirs/stgcn_ntu60_xsub_40e \
    --work-dir work_dirs/stgcn_ntu60_xsub_80e_resume
```

## Experiment 2: Joint Motion

Run `notebooks/04_train_stgcn_joint_motion_80e.ipynb` only after notebook 03
has produced an accepted Joint evaluation with `predictions.csv`, `y_true.npy`,
`y_pred.npy`, and `y_score.npy`. The notebook preserves those files and
`artifacts/checkpoints/stgcn_joint_ntu60_xsub_best.pth` as read-only inputs.

The controlled config changes only every pipeline's `GenSkeFeat` feature from
`['j']` to MMAction2's official `['jm']` representation. It retains ST-GCN,
NTU60 XSub, uniform 100-frame sampling, optimizer/LR, batch size 64, two
workers, seed 42, and validation settings. A clean run uses 16 outer epochs,
`RepeatDataset(times=5)`, and cosine `T_max=16` for 80 effective passes.

Leave `RESUME_CHECKPOINT = None` and `RESUME_METRICS = None` in the notebook for
the initial run. If Kaggle interrupts it, set both variables to the saved full
Joint Motion checkpoint and its matching `epoch_metrics.jsonl`; optimizer and
scheduler state then resume under the unchanged 16-epoch config. The strict
metrics hook requires true outer epochs 1–16 exactly once, preventing the old
off-by-one reporting failure.

After training, the notebook independently evaluates the logged best
checkpoint and creates:

```text
artifacts/checkpoints/stgcn_joint_motion_ntu60_xsub_best.pth
artifacts/experiments/joint_motion/
├── metrics.json
├── predictions.csv
├── y_true.npy
├── y_pred.npy
├── y_score.npy
├── per_class_accuracy.csv
├── confusion_matrix.png
├── high_confidence_errors.csv
└── training_curve.png
artifacts/comparison/
├── joint_vs_joint_motion.csv
├── per_class_joint_vs_motion.csv
├── targeted_class_accuracy.csv
├── targeted_confusions.csv
├── confidence_joint_vs_motion.csv
├── confusion_joint_vs_motion.png
└── joint_vs_joint_motion.md
```

The conclusion and next-experiment recommendation are generated only from the
independent predictions. The notebook does not launch the recommended follow-up.

## Out of scope (this milestone)

Further Joint training, Bone/Bone-Motion training, ST-GCN++, MMPose/YOLO,
RGB/skeleton fusion, webcam inference, TensorRT/ONNX export, NTU120,
Kinetics400, and hyperparameter tuning. None is started by experiment 2.

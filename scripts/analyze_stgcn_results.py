#!/usr/bin/env python3
"""Create NTU60 error-analysis artifacts from frozen validation predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.stgcn_evaluation_common import (
    NTU60_CLASS_NAMES,
    load_training_records,
)

matplotlib.use('Agg')


EXPECTED_RESUMED_TOP1 = {
    9: 0.5125,
    10: 0.8001,
    11: 0.7745,
    12: 0.8593,
    13: 0.8404,
    14: 0.8795,
    15: 0.8818,
    16: 0.8823,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--evaluation-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/evaluation')
    parser.add_argument(
        '--analysis-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/analysis')
    parser.add_argument(
        '--readme-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/readme')
    parser.add_argument(
        '--work-dir', action='append', type=Path, dest='work_dirs')
    return parser.parse_args()


def read_prediction_rows(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    counts = np.zeros((60, 60), dtype=np.int64)
    np.add.at(counts, (y_true, y_pred), 1)
    return counts


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        missing = np.flatnonzero(totals[:, 0] == 0).tolist()
        raise RuntimeError(f'classes absent from validation set: {missing}')
    normalized = counts / totals
    if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError('normalized confusion-matrix rows do not sum to one')
    return normalized


def plot_confusion_matrix(
        matrix: np.ndarray, class_ids: list[int], path: Path,
        title: str, full_matrix: bool) -> None:
    labels = [f'{item}: {NTU60_CLASS_NAMES[item]}' for item in class_ids]
    size = 24 if full_matrix else 15
    fig, axis = plt.subplots(figsize=(size, size - 1))
    image = axis.imshow(matrix, cmap='magma', vmin=0, vmax=1, aspect='auto')
    axis.set_title(title, fontsize=15)
    axis.set_xlabel('Predicted class')
    axis.set_ylabel('Ground-truth class')
    axis.set_xticks(range(len(labels)), labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels)
    tick_size = 4.5 if full_matrix else 7
    axis.tick_params(axis='both', labelsize=tick_size)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label('Fraction of ground-truth class')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        '| Rank | Class ID | Action | Samples | Accuracy |',
        '|---:|---:|---|---:|---:|',
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['class_id']} | {row['action']} | "
            f"{row['samples']} | {float(row['accuracy']):.4f} |")
    return '\n'.join(lines)


def confusion_markdown(rows: list[dict], limit: int = 15) -> str:
    lines = [
        '| Rank | Ground truth | Predicted | Mistakes | GT class % |',
        '|---:|---|---|---:|---:|',
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {row['rank']} | {row['ground_truth_id']}: "
            f"{row['ground_truth_action']} | {row['predicted_id']}: "
            f"{row['predicted_action']} | {row['mistakes']} | "
            f"{100 * float(row['ground_truth_fraction']):.2f}% |")
    return '\n'.join(lines)


def high_confidence_markdown(rows: list[dict]) -> str:
    lines = [
        '| Rank | Sample | Ground truth | Predicted | Confidence |',
        '|---:|---|---|---|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['sample_id']} | "
            f"{row['ground_truth_id']}: {row['ground_truth_action']} | "
            f"{row['predicted_id']}: {row['predicted_action']} | "
            f"{float(row['confidence']):.6f} |")
    return '\n'.join(lines)


def pair_hypothesis(ground_truth_id: int, predicted_id: int) -> str:
    pair = frozenset((ground_truth_id, predicted_id))
    inverse_pairs = {
        frozenset((7, 8)), frozenset((13, 14)), frozenset((15, 16)),
        frozenset((17, 18)), frozenset((19, 20)), frozenset((58, 59)),
    }
    object_dependent = {
        0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        24, 27, 28, 29, 31, 32, 48, 55, 56,
    }
    subtle_upper_body = {
        2, 3, 9, 10, 11, 17, 18, 21, 22, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48,
    }
    semantic_groups = [
        {10, 11, 12, 28, 29},
        {27, 28, 31, 32},
        {34, 35, 36, 37, 40},
        {43, 44, 45, 46, 47},
        set(range(49, 60)),
        {23, 25, 26, 41, 42},
    ]
    reasons: list[str] = []
    if pair in inverse_pairs:
        reasons.append(
            '**action direction ambiguity / insufficient temporal '
            'distinction**: the actions share poses but reverse their temporal '
            'order or direction')
    if ground_truth_id >= 49 and predicted_id >= 49:
        reasons.append(
            '**two-person interaction**: both labels depend on a similar '
            'relational skeleton pattern between two people')
    if ground_truth_id in object_dependent or predicted_id in object_dependent:
        reasons.append(
            '**object-dependent action / hand-object interaction**: a 2D '
            'skeleton does not encode the manipulated object or its appearance')
    if ground_truth_id in subtle_upper_body and predicted_id in subtle_upper_body:
        reasons.append(
            '**similar spatial skeleton / pose-estimation noise**: the '
            'distinction may rely on small hand, head, or upper-body offsets')
    if any({ground_truth_id, predicted_id} <= group for group in semantic_groups):
        reasons.append(
            '**similar temporal motion**: the pair belongs to the same broad '
            'motion family and may differ mainly in local trajectory details')
    if not reasons:
        reasons.append(
            '**similar spatial or temporal skeleton (hypothesis)**: the '
            'aggregate confusion supports overlap in the learned pose/motion '
            'representation, but individual sequences must be inspected before '
            'assigning a more specific cause')
    return '; '.join(reasons) + '.'


def create_error_analysis(path: Path, top_confusions: list[dict]) -> None:
    lines = [
        '# ST-GCN Joint error analysis',
        '',
        ('The items below are directed errors on NTU60 `xsub_val`. '
         'Explanations are hypotheses grounded in the action pair; RGB frames '
         'and per-sample pose quality were not inspected, so they are not '
         'treated as proven causes.'),
        '',
    ]
    for row in top_confusions[:10]:
        lines.extend([
            (f"## {row['rank']}. {row['ground_truth_id']}: "
             f"{row['ground_truth_action']} → {row['predicted_id']}: "
             f"{row['predicted_action']}"),
            '',
            (f"- Evidence: {row['mistakes']} mistakes, "
             f"{100 * float(row['ground_truth_fraction']):.2f}% of the "
             'ground-truth class.'),
            f"- Hypothesis: {pair_hypothesis(int(row['ground_truth_id']), int(row['predicted_id']))}",
            '',
        ])
    path.write_text('\n'.join(lines))


def build_training_curve(
        records: dict[int, dict], path: Path) -> tuple[dict[int, dict], list[str]]:
    notes: list[str] = []
    if 8 not in records:
        records[8] = {
            'outer_epoch': 8,
            'effective_epoch': 40,
            'val_acc_top1': 0.8775,
            'val_acc_top5': 0.9866,
            'source': 'documented pre-resume validation output',
        }
        notes.append(
            'Effective epoch 40 was added from the documented epoch-8 '
            'validation output because its local metric log was unavailable.')

    for epoch, expected_top1 in EXPECTED_RESUMED_TOP1.items():
        if epoch not in records or records[epoch].get('val_acc_top1') is None:
            records.setdefault(epoch, {
                'outer_epoch': epoch, 'effective_epoch': epoch * 5})
            records[epoch]['val_acc_top1'] = expected_top1
            records[epoch]['source'] = 'documented resumed sequence'
            notes.append(
                f'Effective epoch {epoch * 5} Top-1 was added from the '
                'documented resumed sequence because its local log was missing.')
        elif not math.isclose(
                float(records[epoch]['val_acc_top1']), expected_top1,
                abs_tol=5e-5):
            raise RuntimeError(
                f'logged Top-1 for effective epoch {epoch * 5} is '
                f"{records[epoch]['val_acc_top1']}, expected documented "
                f'{expected_top1}')

    ordered = [records[key] for key in sorted(records)]
    fig, (accuracy_axis, loss_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]})
    for key, label, color in (
            ('val_acc_top1', 'Validation Top-1', '#1565c0'),
            ('val_acc_top5', 'Validation Top-5', '#2e7d32')):
        points = [
            (item['effective_epoch'], item[key]) for item in ordered
            if item.get(key) is not None
        ]
        if points:
            x, y = zip(*points)
            accuracy_axis.plot(x, y, marker='o', label=label, color=color)
    accuracy_axis.set_ylabel('Accuracy')
    accuracy_axis.set_ylim(0, 1.02)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(loc='lower right')

    loss_points = [
        (item['effective_epoch'], item['train_loss']) for item in ordered
        if item.get('train_loss') is not None
    ]
    if loss_points:
        x, y = zip(*loss_points)
        loss_axis.plot(x, y, marker='o', color='#c62828', label='Training loss')
        loss_axis.legend(loc='upper right')
    loss_axis.set_xlabel('Effective epoch')
    loss_axis.set_ylabel('Train loss')
    loss_axis.grid(alpha=0.25)

    for axis in (accuracy_axis, loss_axis):
        axis.axvline(40, color='black', linestyle='--', linewidth=1.4)
    accuracy_axis.annotate(
        'Resume boundary', xy=(40, 1.0), xytext=(42, 0.94),
        arrowprops={'arrowstyle': '->', 'color': 'black'})
    accuracy_axis.set_title(
        'ST-GCN Joint training curve (resume discontinuity retained)')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return dict(sorted(records.items())), notes


def main() -> None:
    args = parse_args()
    work_dirs = args.work_dirs or [
        PROJECT_ROOT / 'work_dirs/stgcn_ntu60_xsub_40e',
        PROJECT_ROOT / 'work_dirs/stgcn_ntu60_xsub_80e_resume',
    ]
    required = {
        'metrics': args.evaluation_dir / 'baseline_metrics.json',
        'selection': args.evaluation_dir / 'best_checkpoint.json',
        'predictions': args.evaluation_dir / 'predictions.csv',
        'y_true': args.evaluation_dir / 'y_true.npy',
        'y_pred': args.evaluation_dir / 'y_pred.npy',
        'y_score': args.evaluation_dir / 'y_score.npy',
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'independent evaluation outputs missing: ' + ', '.join(missing))

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.readme_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(required['metrics'].read_text())
    selection = json.loads(required['selection'].read_text())
    if metrics.get('evaluation_status') != 'accepted':
        raise RuntimeError(
            'baseline_metrics.json is not from an accepted independent '
            'evaluation; refusing to analyze cached or discrepant predictions')
    prediction_rows = read_prediction_rows(required['predictions'])
    y_true = np.load(required['y_true'])
    y_pred = np.load(required['y_pred'])
    y_score = np.load(required['y_score'])
    if not (len(y_true) == len(y_pred) == len(y_score) == len(prediction_rows)):
        raise RuntimeError('prediction CSV and arrays have inconsistent lengths')
    if y_score.shape != (len(y_true), 60):
        raise RuntimeError(f'unexpected y_score shape: {y_score.shape}')

    counts = confusion_counts(y_true, y_pred)
    normalized = normalize_rows(counts)
    plot_confusion_matrix(
        normalized, list(range(60)),
        args.analysis_dir / 'confusion_matrix.png',
        'NTU60 XSub validation — normalized confusion matrix', True)
    error_counts = counts.sum(axis=1) - np.diag(counts)
    top20_ids = sorted(
        np.argsort(-error_counts, kind='stable')[:20].tolist())
    subset = normalized[np.ix_(top20_ids, top20_ids)]
    plot_confusion_matrix(
        subset, top20_ids,
        args.analysis_dir / 'confusion_matrix_top20.png',
        'NTU60 XSub — 20 ground-truth classes with most errors', False)

    per_class_rows = []
    for class_id in range(60):
        samples = int(counts[class_id].sum())
        correct = int(counts[class_id, class_id])
        per_class_rows.append({
            'class_id': class_id,
            'action': NTU60_CLASS_NAMES[class_id],
            'samples': samples,
            'correct': correct,
            'incorrect': samples - correct,
            'accuracy': correct / samples,
        })
    write_csv(
        args.analysis_dir / 'per_class_accuracy.csv', per_class_rows,
        ['class_id', 'action', 'samples', 'correct', 'incorrect', 'accuracy'])
    easiest = sorted(
        per_class_rows,
        key=lambda item: (-item['accuracy'], -item['samples'], item['class_id']))[:10]
    hardest = sorted(
        per_class_rows,
        key=lambda item: (item['accuracy'], -item['samples'], item['class_id']))[:10]

    confusion_rows = []
    for ground_truth_id in range(60):
        total = int(counts[ground_truth_id].sum())
        for predicted_id in range(60):
            if ground_truth_id == predicted_id or counts[
                    ground_truth_id, predicted_id] == 0:
                continue
            mistakes = int(counts[ground_truth_id, predicted_id])
            confusion_rows.append({
                'ground_truth_id': ground_truth_id,
                'ground_truth_action': NTU60_CLASS_NAMES[ground_truth_id],
                'predicted_id': predicted_id,
                'predicted_action': NTU60_CLASS_NAMES[predicted_id],
                'mistakes': mistakes,
                'ground_truth_samples': total,
                'ground_truth_fraction': mistakes / total,
            })
    confusion_rows.sort(
        key=lambda item: (
            -item['mistakes'], -item['ground_truth_fraction'],
            item['ground_truth_id'], item['predicted_id']))
    top_confusions = []
    for rank, row in enumerate(confusion_rows[:15], start=1):
        top_confusions.append({'rank': rank, **row})
    write_csv(
        args.analysis_dir / 'top_confusions.csv', top_confusions,
        ['rank', 'ground_truth_id', 'ground_truth_action', 'predicted_id',
         'predicted_action', 'mistakes', 'ground_truth_samples',
         'ground_truth_fraction'])

    confidence = y_score[np.arange(len(y_pred)), y_pred]
    correct_mask = y_true == y_pred
    correct_confidence = confidence[correct_mask]
    incorrect_confidence = confidence[~correct_mask]
    confidence_stats = {
        'mean_confidence_correct': float(np.mean(correct_confidence)),
        'mean_confidence_incorrect': float(np.mean(incorrect_confidence)),
        'median_confidence_correct': float(np.median(correct_confidence)),
        'median_confidence_incorrect': float(np.median(incorrect_confidence)),
        'correct_predictions': int(correct_mask.sum()),
        'incorrect_predictions': int((~correct_mask).sum()),
    }
    (args.analysis_dir / 'confidence_analysis.json').write_text(
        json.dumps(confidence_stats, indent=2))
    wrong_indices = np.flatnonzero(~correct_mask)
    wrong_indices = wrong_indices[
        np.argsort(-confidence[wrong_indices], kind='stable')[:10]]
    high_confidence_rows = []
    for rank, index in enumerate(wrong_indices, start=1):
        row = prediction_rows[int(index)]
        high_confidence_rows.append({
            'rank': rank,
            'sample_id': row['sample_id'],
            'ground_truth_id': int(y_true[index]),
            'ground_truth_action': NTU60_CLASS_NAMES[y_true[index]],
            'predicted_id': int(y_pred[index]),
            'predicted_action': NTU60_CLASS_NAMES[y_pred[index]],
            'confidence': float(confidence[index]),
            'top5_classes': row['top5_classes'],
            'top5_scores': row['top5_scores'],
        })
    write_csv(
        args.analysis_dir / 'high_confidence_errors.csv',
        high_confidence_rows,
        ['rank', 'sample_id', 'ground_truth_id', 'ground_truth_action',
         'predicted_id', 'predicted_action', 'confidence', 'top5_classes',
         'top5_scores'])
    create_error_analysis(
        args.analysis_dir / 'error_analysis.md', top_confusions)

    records = load_training_records(work_dirs)
    records, curve_notes = build_training_curve(
        records, args.analysis_dir / 'training_curve.png')
    sequence = '\n'.join(
        f"- Effective {epoch * 5}: Top-1 {EXPECTED_RESUMED_TOP1[epoch]:.4f}"
        for epoch in sorted(EXPECTED_RESUMED_TOP1))
    resume_state_paths = [
        Path(work_dir) / 'resume_state.json' for work_dir in work_dirs
        if (Path(work_dir) / 'resume_state.json').is_file()
    ]
    resume_state_note = 'No local resume_state.json was available.'
    if resume_state_paths:
        state = json.loads(resume_state_paths[-1].read_text())
        resume_state_note = (
            'Recorded resume state: optimizer status = '
            f"`{state.get('optimizer_status', 'unknown')}`, cosine T_max = "
            f"`{state.get('old_cosine_t_max', 'unknown')}` → "
            f"`{state.get('new_cosine_t_max', 'unknown')}` steps."
        )

    hardest_table = markdown_table(hardest)
    easiest_table = markdown_table(easiest)
    confusions_table = confusion_markdown(top_confusions)
    high_confidence_table = high_confidence_markdown(high_confidence_rows)
    observations = f"""- Independent validation reproduced Top-1 `{metrics['top1']:.4f}`, Top-5 `{metrics['top5']:.4f}`, and mean-class accuracy `{metrics['mean1']:.4f}`.
- A resume boundary is marked at effective epoch 40. The observed post-resume Top-1 sequence is retained without smoothing:
{sequence}
- {resume_state_note}
- The highest-confidence mistakes are saved separately for sample-level inspection.
"""
    if curve_notes:
        observations += '- Curve provenance notes:\n' + '\n'.join(
            f'  - {note}' for note in curve_notes) + '\n'

    summary = f"""# ST-GCN Joint NTU60 XSub experiment summary

EXPERIMENT
----------
Model: ST-GCN
Dataset: NTU RGB+D 60 2D skeleton
Protocol: Cross-Subject (XSub)
Input representation: Joint
Number of classes: 60
Train samples: {metrics.get('train_samples', 40091)}
Validation samples: {metrics['validation_samples']}
Effective epochs: 80

BEST RESULT
-----------
Best epoch: {selection['best_epoch']} (effective {selection['best_epoch'] * 5})
Top-1: {metrics['top1']:.4f}
Top-5: {metrics['top5']:.4f}
Mean-class accuracy: {metrics['mean1']:.4f}

HARDEST CLASSES
---------------
{hardest_table}

EASIEST CLASSES
---------------
{easiest_table}

TOP CONFUSIONS
--------------
{confusions_table}

CONFIDENCE ANALYSIS
-------------------
- Mean confidence, correct: {confidence_stats['mean_confidence_correct']:.6f}
- Mean confidence, incorrect: {confidence_stats['mean_confidence_incorrect']:.6f}
- Median confidence, correct: {confidence_stats['median_confidence_correct']:.6f}
- Median confidence, incorrect: {confidence_stats['median_confidence_incorrect']:.6f}

### 10 highest-confidence wrong predictions

{high_confidence_table}

OBSERVATIONS
------------
{observations}
LIMITATIONS
-----------
- The model sees 2D skeleton joints, not RGB appearance or object identity.
- Explanations of confusion pairs are hypotheses until individual RGB/pose sequences are inspected.
- Results cover NTU60 XSub validation only; they do not establish cross-view or external-dataset generalization.
- The interrupted/resumed optimization path differs from an uninterrupted 80-effective-epoch run and is shown explicitly in the training curve.

NEXT EXPERIMENT
---------------
After reviewing the high-confidence Joint errors, run an otherwise matched ST-GCN Bone-input baseline as the next controlled ablation. It has not been started by this pipeline.
"""
    (args.analysis_dir / 'experiment_summary.md').write_text(summary)

    baseline_summary = f"""# Baseline result

| Model | Input | Dataset | Protocol | Top-1 | Top-5 |
|-------|-------|---------|----------|-------|-------|
| ST-GCN | Joint | NTU60 | XSub | {metrics['top1']:.4f} | {metrics['top5']:.4f} |

- Best outer epoch: {selection['best_epoch']}
- Mean-class accuracy: {metrics['mean1']:.4f}
- Validation samples: {metrics['validation_samples']}
- Evaluation source: independently generated `xsub_val` predictions from the frozen checkpoint.
"""
    (args.readme_dir / 'baseline_summary.md').write_text(baseline_summary)
    for source_name, destination_name in (
            ('training_curve.png', 'training_curve.png'),
            ('confusion_matrix.png', 'confusion_matrix.png'),
            ('top_confusions.csv', 'top_confusions.csv'),
            ('per_class_accuracy.csv', 'per_class_accuracy.csv')):
        shutil.copy2(
            args.analysis_dir / source_name,
            args.readme_dir / destination_name)

    print('\nTop 10 easiest classes\n')
    print(easiest_table)
    print('\nTop 10 hardest classes\n')
    print(hardest_table)
    print('\nTop 15 directed confusion pairs\n')
    print(confusions_table)
    print('\nConfidence analysis\n')
    print(
        'Mean confidence correct:',
        f"{confidence_stats['mean_confidence_correct']:.6f}")
    print(
        'Mean confidence incorrect:',
        f"{confidence_stats['mean_confidence_incorrect']:.6f}")
    print('\n10 highest-confidence wrong predictions\n')
    print(high_confidence_table)
    print('\nAnalysis artifacts written to:', args.analysis_dir)


if __name__ == '__main__':
    main()

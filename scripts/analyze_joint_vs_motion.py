#!/usr/bin/env python3
"""Compare the frozen ST-GCN Joint and controlled Joint Motion experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scripts.analyze_stgcn_results import (
    confusion_counts,
    normalize_rows,
    plot_confusion_matrix,
    write_csv,
)
from scripts.stgcn_evaluation_common import NTU60_CLASS_NAMES

TARGET_CLASS_IDS = (9, 33, 36, 3, 43, 30, 31, 11, 29)
TARGET_PAIRS = (
    (9, 33),   # clapping -> rub two hands together
    (36, 3),   # wipe face -> brushing hair
    (36, 43),  # wipe face -> touch head
    (30, 31),  # pointing -> taking a selfie
    (11, 29),  # writing -> typing on a keyboard
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--baseline', type=Path,
        default=PROJECT_ROOT / 'artifacts/experiments/joint/baseline.json')
    parser.add_argument(
        '--joint-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/evaluation')
    parser.add_argument(
        '--motion-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/experiments/joint_motion')
    parser.add_argument(
        '--comparison-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/comparison')
    parser.add_argument(
        '--work-dir', type=Path,
        default=PROJECT_ROOT /
        'work_dirs/stgcn_ntu60_xsub_joint_motion_80e')
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('required experiment artifacts missing: ' +
                                ', '.join(missing))


def load_prediction_bundle(directory: Path, metrics_name: str) -> dict:
    paths = {
        'metrics': directory / metrics_name,
        'predictions': directory / 'predictions.csv',
        'y_true': directory / 'y_true.npy',
        'y_pred': directory / 'y_pred.npy',
        'y_score': directory / 'y_score.npy',
    }
    require_files(list(paths.values()))
    bundle = {
        'metrics': json.loads(paths['metrics'].read_text()),
        'rows': read_rows(paths['predictions']),
        'y_true': np.load(paths['y_true']),
        'y_pred': np.load(paths['y_pred']),
        'y_score': np.load(paths['y_score']),
    }
    lengths = {
        len(bundle['rows']), len(bundle['y_true']), len(bundle['y_pred']),
        len(bundle['y_score']),
    }
    if len(lengths) != 1:
        raise RuntimeError(f'inconsistent prediction lengths in {directory}')
    if bundle['y_score'].shape != (len(bundle['y_true']), 60):
        raise RuntimeError(
            f"invalid score shape in {directory}: {bundle['y_score'].shape}")
    for index, row in enumerate(bundle['rows']):
        if (int(row['ground_truth_id']) != int(bundle['y_true'][index]) or
                int(row['predicted_id']) != int(bundle['y_pred'][index])):
            raise RuntimeError(
                f'prediction CSV/array mismatch in {directory} at {index}')
    return bundle


def measured_metrics(bundle: dict) -> tuple[float, float]:
    y_true = bundle['y_true']
    y_pred = bundle['y_pred']
    ranking = np.argsort(-bundle['y_score'], axis=1, kind='stable')
    top1 = float(np.mean(y_pred == y_true))
    top5 = float(np.mean(np.any(ranking[:, :5] == y_true[:, None], axis=1)))
    return top1, top5


def class_rows(
        joint_counts: np.ndarray, motion_counts: np.ndarray) -> list[dict]:
    rows = []
    for class_id, action in enumerate(NTU60_CLASS_NAMES):
        joint_total = int(joint_counts[class_id].sum())
        motion_total = int(motion_counts[class_id].sum())
        if joint_total != motion_total or joint_total == 0:
            raise RuntimeError(
                f'class {class_id} has incompatible validation counts: '
                f'{joint_total} vs {motion_total}')
        joint_accuracy = joint_counts[class_id, class_id] / joint_total
        motion_accuracy = motion_counts[class_id, class_id] / motion_total
        rows.append({
            'class_id': class_id,
            'action': action,
            'joint_accuracy': float(joint_accuracy),
            'motion_accuracy': float(motion_accuracy),
            'delta': float(motion_accuracy - joint_accuracy),
            'num_samples': joint_total,
        })
    return rows


def class_markdown(rows: list[dict]) -> str:
    lines = [
        '| Rank | Class | Joint accuracy | Motion accuracy | Delta |',
        '|---:|---|---:|---:|---:|',
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['class_id']}: {row['action']} | "
            f"{row['joint_accuracy']:.4f} | {row['motion_accuracy']:.4f} | "
            f"{row['delta']:+.4f} |")
    return '\n'.join(lines)


def targeted_class_markdown(rows: list[dict]) -> str:
    lines = [
        '| Class | Joint accuracy | Motion accuracy | Delta | Samples |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['class_id']}: {row['action']} | "
            f"{row['joint_accuracy']:.4f} | {row['motion_accuracy']:.4f} | "
            f"{row['delta']:+.4f} | {row['num_samples']} |")
    return '\n'.join(lines)


def targeted_confusion_rows(
        joint_counts: np.ndarray, motion_counts: np.ndarray) -> list[dict]:
    rows = []
    for ground_truth, predicted in TARGET_PAIRS:
        joint_errors = int(joint_counts[ground_truth, predicted])
        motion_errors = int(motion_counts[ground_truth, predicted])
        rows.append({
            'ground_truth': (
                f'{ground_truth}: {NTU60_CLASS_NAMES[ground_truth]}'),
            'predicted': f'{predicted}: {NTU60_CLASS_NAMES[predicted]}',
            'joint_error_count': joint_errors,
            'motion_error_count': motion_errors,
            'difference': motion_errors - joint_errors,
        })
    return rows


def targeted_confusion_markdown(rows: list[dict]) -> str:
    lines = [
        '| Ground truth → predicted | Joint errors | Motion errors | Delta |',
        '|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['ground_truth']} → {row['predicted']} | "
            f"{row['joint_error_count']} | {row['motion_error_count']} | "
            f"{int(row['difference']):+d} |")
    return '\n'.join(lines)


def expected_calibration_error(
        y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray,
        bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    correct = y_true == y_pred
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = ((confidence >= lower) &
                (confidence < upper if index < bins - 1
                 else confidence <= upper))
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) -
                float(np.mean(correct[mask])))
    return result


def confidence_summary(bundle: dict) -> dict:
    y_true, y_pred, y_score = (
        bundle['y_true'], bundle['y_pred'], bundle['y_score'])
    confidence = y_score[np.arange(len(y_pred)), y_pred]
    correct = y_true == y_pred
    if not np.any(correct) or not np.any(~correct):
        raise RuntimeError('confidence analysis requires correct and wrong cases')
    return {
        'mean_confidence_correct': float(np.mean(confidence[correct])),
        'mean_confidence_incorrect': float(np.mean(confidence[~correct])),
        'median_confidence_correct': float(np.median(confidence[correct])),
        'median_confidence_incorrect': float(np.median(confidence[~correct])),
        'ece_15bin': expected_calibration_error(
            y_true, y_pred, confidence, bins=15),
        'confidence': confidence,
        'correct': correct,
    }


def write_high_confidence_errors(path: Path, bundle: dict, summary: dict) -> None:
    confidence = summary['confidence']
    wrong = np.flatnonzero(~summary['correct'])
    wrong = wrong[np.argsort(-confidence[wrong], kind='stable')[:10]]
    rows = []
    for rank, index in enumerate(wrong, start=1):
        source = bundle['rows'][int(index)]
        ground_truth = int(bundle['y_true'][index])
        predicted = int(bundle['y_pred'][index])
        rows.append({
            'rank': rank,
            'sample_id': source['sample_id'],
            'ground_truth_id': ground_truth,
            'ground_truth_name': NTU60_CLASS_NAMES[ground_truth],
            'predicted_id': predicted,
            'predicted_name': NTU60_CLASS_NAMES[predicted],
            'confidence': float(confidence[index]),
        })
    write_csv(
        path, rows,
        ['rank', 'sample_id', 'ground_truth_id', 'ground_truth_name',
         'predicted_id', 'predicted_name', 'confidence'])


def plot_confusion_comparison(
        joint: np.ndarray, motion: np.ndarray, class_ids: list[int],
        path: Path) -> None:
    labels = [f'{item}: {NTU60_CLASS_NAMES[item]}' for item in class_ids]
    joint_subset = joint[np.ix_(class_ids, class_ids)]
    motion_subset = motion[np.ix_(class_ids, class_ids)]
    difference = motion_subset - joint_subset
    limit = max(float(np.max(np.abs(difference))), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    panels = (
        (joint_subset, 'Joint', 'magma', 0, 1),
        (motion_subset, 'Joint Motion', 'magma', 0, 1),
        (difference, 'Motion − Joint', 'coolwarm', -limit, limit),
    )
    for axis, (matrix, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(
            matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axis.set_title(title)
        axis.set_xlabel('Predicted class')
        axis.set_ylabel('Ground-truth class')
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.suptitle('Normalized confusion comparison — most error-prone classes')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def load_complete_history(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Joint Motion epoch log missing: {path}')
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    counts = Counter(int(row['outer_epoch']) for row in rows)
    expected = set(range(1, 17))
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise RuntimeError(
            'Joint Motion epoch log must contain each true outer epoch 1..16 '
            f'exactly once; observed counts: {dict(sorted(counts.items()))}')
    required = (
        'effective_epoch', 'train_loss', 'val_acc_top1', 'val_acc_top5',
        'learning_rate', 'gpu_memory_mb', 'epoch_wall_time_sec')
    ordered = sorted(rows, key=lambda row: int(row['outer_epoch']))
    for row in ordered:
        epoch = int(row['outer_epoch'])
        if int(row['effective_epoch']) != epoch * 5:
            raise RuntimeError(f'off-by-one effective epoch in record: {row}')
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise RuntimeError(
                f'epoch {epoch} has missing metrics: {missing}')
    return ordered


def plot_training_curve(history: list[dict], path: Path) -> dict:
    effective = np.asarray([row['effective_epoch'] for row in history])
    loss = np.asarray([row['train_loss'] for row in history], dtype=float)
    top1 = np.asarray([row['val_acc_top1'] for row in history], dtype=float)
    top5 = np.asarray([row['val_acc_top5'] for row in history], dtype=float)
    learning_rate = np.asarray(
        [row['learning_rate'] for row in history], dtype=float)
    expected_lr = .05 * (1 + np.cos(np.pi * effective / 80))
    restarts = np.flatnonzero(np.diff(learning_rate) > 1e-10).tolist()
    maximum_deviation = float(np.max(np.abs(learning_rate - expected_lr)))
    status = 'ok' if not restarts and maximum_deviation <= 1e-3 else 'unexpected'

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(effective, loss, marker='o', color='#c62828')
    axes[0].set_ylabel('Training loss')
    axes[1].plot(effective, top1, marker='o', label='Validation Top-1')
    axes[1].plot(effective, top5, marker='o', label='Validation Top-5')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[2].plot(effective, learning_rate, marker='o', label='Logged LR')
    axes[2].plot(
        effective, expected_lr, linestyle='--', label='Expected cosine LR')
    axes[2].set_ylabel('Learning rate')
    axes[2].set_xlabel('Effective epoch')
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle('ST-GCN Joint Motion — clean 16 × 5 training run')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return {
        'status': status,
        'lr_restart_after_outer_epochs': [item + 1 for item in restarts],
        'maximum_absolute_deviation_from_expected_cosine': maximum_deviation,
        'expected_initial_lr': .1,
        't_max_outer_epochs': 16,
    }


def history_markdown(history: list[dict]) -> str:
    lines = [
        '| Outer | Effective | Loss | Top-1 | Top-5 | LR |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for row in history:
        lines.append(
            f"| {row['outer_epoch']} | {row['effective_epoch']} | "
            f"{row['train_loss']:.6f} | {row['val_acc_top1']:.4f} | "
            f"{row['val_acc_top5']:.4f} | {row['learning_rate']:.9f} |")
    return '\n'.join(lines)


def choose_recommendation(targeted: list[dict]) -> str:
    dynamic_joint = sum(row['joint_error_count'] for row in targeted[:3])
    dynamic_motion = sum(row['motion_error_count'] for row in targeted[:3])
    object_joint = sum(row['joint_error_count'] for row in targeted[3:])
    object_motion = sum(row['motion_error_count'] for row in targeted[3:])
    if dynamic_motion < dynamic_joint:
        return 'Joint + Motion fusion'
    if object_motion >= object_joint:
        return 'RGB/skeleton fusion'
    return 'Bone'


def main() -> None:
    args = parse_args()
    require_files([args.baseline])
    baseline = json.loads(args.baseline.read_text())
    joint = load_prediction_bundle(args.joint_dir, 'baseline_metrics.json')
    motion = load_prediction_bundle(args.motion_dir, 'metrics.json')
    require_files([args.motion_dir / 'best_checkpoint.json'])
    selection = json.loads(
        (args.motion_dir / 'best_checkpoint.json').read_text())
    if motion['metrics'].get('evaluation_status') != 'accepted':
        raise RuntimeError('Joint Motion independent evaluation is not accepted')
    if joint['metrics'].get('evaluation_status') != 'accepted':
        raise RuntimeError('Joint independent evaluation is not accepted')
    if not np.array_equal(joint['y_true'], motion['y_true']):
        raise RuntimeError(
            'Joint and Joint Motion validation samples/ordering differ')
    joint_ids = [row['sample_id'] for row in joint['rows']]
    motion_ids = [row['sample_id'] for row in motion['rows']]
    if joint_ids != motion_ids:
        raise RuntimeError(
            'Joint and Joint Motion validation sample IDs/order differ')

    joint_measured_top1, joint_measured_top5 = measured_metrics(joint)
    motion_top1, motion_top5 = measured_metrics(motion)
    if (abs(joint_measured_top1 - float(baseline['top1'])) > .002 or
            abs(joint_measured_top5 - float(baseline['top5'])) > .002):
        raise RuntimeError(
            'Joint predictions do not reproduce baseline Top-1/Top-5')
    if abs(motion_top1 - float(motion['metrics']['top1'])) > 1e-9:
        raise RuntimeError('Joint Motion arrays disagree with metrics.json')
    if abs(motion_top5 - float(motion['metrics']['top5'])) > 1e-9:
        raise RuntimeError(
            'Joint Motion Top-5 array result disagrees with metrics.json')

    args.motion_dir.mkdir(parents=True, exist_ok=True)
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    joint_counts = confusion_counts(joint['y_true'], joint['y_pred'])
    motion_counts = confusion_counts(motion['y_true'], motion['y_pred'])
    joint_normalized = normalize_rows(joint_counts)
    motion_normalized = normalize_rows(motion_counts)
    plot_confusion_matrix(
        motion_normalized, list(range(60)),
        args.motion_dir / 'confusion_matrix.png',
        'NTU60 XSub — ST-GCN Joint Motion normalized confusion matrix', True)

    error_weight = (
        joint_counts.sum(axis=1) - np.diag(joint_counts) +
        motion_counts.sum(axis=1) - np.diag(motion_counts))
    confused_ids = sorted(
        np.argsort(-error_weight, kind='stable')[:20].tolist())
    plot_confusion_comparison(
        joint_normalized, motion_normalized, confused_ids,
        args.comparison_dir / 'confusion_joint_vs_motion.png')

    all_class_rows = class_rows(joint_counts, motion_counts)
    write_csv(
        args.comparison_dir / 'per_class_joint_vs_motion.csv', all_class_rows,
        ['class_id', 'action', 'joint_accuracy', 'motion_accuracy', 'delta',
         'num_samples'])
    motion_class_rows = [{
        'class_id': row['class_id'], 'action': row['action'],
        'num_samples': row['num_samples'],
        'correct': int(motion_counts[row['class_id'], row['class_id']]),
        'incorrect': int(row['num_samples'] -
                         motion_counts[row['class_id'], row['class_id']]),
        'accuracy': row['motion_accuracy'],
    } for row in all_class_rows]
    write_csv(
        args.motion_dir / 'per_class_accuracy.csv', motion_class_rows,
        ['class_id', 'action', 'num_samples', 'correct', 'incorrect',
         'accuracy'])
    improved = sorted(
        all_class_rows,
        key=lambda row: (-row['delta'], -row['num_samples'], row['class_id']))[:10]
    degraded = sorted(
        all_class_rows,
        key=lambda row: (row['delta'], -row['num_samples'], row['class_id']))[:10]
    targeted_classes = [all_class_rows[index] for index in TARGET_CLASS_IDS]
    write_csv(
        args.comparison_dir / 'targeted_class_accuracy.csv', targeted_classes,
        ['class_id', 'action', 'joint_accuracy', 'motion_accuracy', 'delta',
         'num_samples'])
    targeted = targeted_confusion_rows(joint_counts, motion_counts)
    write_csv(
        args.comparison_dir / 'targeted_confusions.csv', targeted,
        ['ground_truth', 'predicted', 'joint_error_count',
         'motion_error_count', 'difference'])

    delta_top1 = motion_top1 - float(baseline['top1'])
    delta_top5 = motion_top5 - float(baseline['top5'])
    global_rows = [
        {
            'model': 'ST-GCN', 'input': 'Joint',
            'top1': baseline['top1'], 'top5': baseline['top5'],
            'delta_top1_vs_joint': 0.0, 'delta_top5_vs_joint': 0.0,
        },
        {
            'model': 'ST-GCN', 'input': 'Joint Motion',
            'top1': motion_top1, 'top5': motion_top5,
            'delta_top1_vs_joint': delta_top1,
            'delta_top5_vs_joint': delta_top5,
        },
    ]
    write_csv(
        args.comparison_dir / 'joint_vs_joint_motion.csv', global_rows,
        ['model', 'input', 'top1', 'top5', 'delta_top1_vs_joint',
         'delta_top5_vs_joint'])

    joint_confidence = confidence_summary(joint)
    motion_confidence = confidence_summary(motion)
    write_high_confidence_errors(
        args.motion_dir / 'high_confidence_errors.csv',
        motion, motion_confidence)
    confidence_rows = []
    for representation, accuracy, summary in (
            ('Joint', float(baseline['top1']), joint_confidence),
            ('Joint Motion', motion_top1, motion_confidence)):
        confidence_rows.append({
            'input': representation,
            'top1': accuracy,
            'mean_confidence_correct': summary['mean_confidence_correct'],
            'mean_confidence_incorrect': summary['mean_confidence_incorrect'],
            'median_confidence_correct': summary['median_confidence_correct'],
            'median_confidence_incorrect': summary[
                'median_confidence_incorrect'],
            'ece_15bin': summary['ece_15bin'],
        })
    write_csv(
        args.comparison_dir / 'confidence_joint_vs_motion.csv',
        confidence_rows,
        ['input', 'top1', 'mean_confidence_correct',
         'mean_confidence_incorrect', 'median_confidence_correct',
         'median_confidence_incorrect', 'ece_15bin'])

    history = load_complete_history(args.work_dir / 'epoch_metrics.jsonl')
    lr_check = plot_training_curve(
        history, args.motion_dir / 'training_curve.png')
    (args.motion_dir / 'lr_schedule_check.json').write_text(
        json.dumps(lr_check, indent=2))
    improved_table = class_markdown(improved)
    degraded_table = class_markdown(degraded)
    target_class_table = targeted_class_markdown(targeted_classes)
    target_confusion_table = targeted_confusion_markdown(targeted)
    schedule_statement = (
        'The logged LR follows the expected uninterrupted 16-epoch cosine '
        'schedule.' if lr_check['status'] == 'ok' else
        'WARNING: the logged LR contains a restart or deviates unexpectedly '
        'from the configured 16-epoch cosine schedule.')

    dynamic_joint = sum(row['joint_error_count'] for row in targeted[:3])
    dynamic_motion = sum(row['motion_error_count'] for row in targeted[:3])
    object_joint = sum(row['joint_error_count'] for row in targeted[3:])
    object_motion = sum(row['motion_error_count'] for row in targeted[3:])
    ece_delta = motion_confidence['ece_15bin'] - joint_confidence['ece_15bin']
    if abs(ece_delta) < .005:
        calibration_statement = (
            '15-bin ECE changed by less than 0.005, so the measured change is '
            'primarily classification behavior rather than clear calibration '
            'change.')
    elif ece_delta < 0:
        calibration_statement = (
            '15-bin ECE decreased, providing evidence of improved calibration '
            'as well as any accuracy change.')
    else:
        calibration_statement = (
            '15-bin ECE increased, so Joint Motion is less calibrated even if '
            'its classification accuracy changed favorably.')
    recommendation = choose_recommendation(targeted)
    conclusion = f"""# Joint vs Joint Motion

## 1. Did Joint Motion improve overall accuracy?

{'Yes' if delta_top1 > 0 else 'No'}.

- Joint Top-1: `{float(baseline['top1']):.4f}`
- Joint Motion Top-1: `{motion_top1:.4f}`
- ΔTop-1: `{delta_top1:+.4f}`
- ΔTop-5: `{delta_top5:+.4f}`

## 2. Which classes benefited most?

{improved_table}

## 3. Which classes became worse?

{degraded_table}

## 4. Did Motion reduce errors between similar dynamic actions?

Across the three targeted dynamic directions (clapping → rubbing hands and
the two wipe-face errors), errors changed from `{dynamic_joint}` to
`{dynamic_motion}`. Negative pair deltas mean fewer errors with Motion.

{target_confusion_table}

Targeted per-class evidence:

{target_class_table}

## 5. Did Motion fix object-dependent actions?

For pointing → selfie and writing → typing, the combined directed error count
changed from `{object_joint}` to `{object_motion}`. These measured counts—not
the expected hypothesis—determine whether motion helped. Skeleton motion still
contains no explicit object appearance or scene context.

## Confidence comparison

- Joint mean confidence (correct / incorrect):
  `{joint_confidence['mean_confidence_correct']:.4f}` /
  `{joint_confidence['mean_confidence_incorrect']:.4f}`
- Joint Motion mean confidence (correct / incorrect):
  `{motion_confidence['mean_confidence_correct']:.4f}` /
  `{motion_confidence['mean_confidence_incorrect']:.4f}`
- Joint / Motion 15-bin ECE: `{joint_confidence['ece_15bin']:.4f}` /
  `{motion_confidence['ece_15bin']:.4f}`
- Interpretation: {calibration_statement}

## Training schedule

{schedule_statement}

{history_markdown(history)}

## 6. What should the next experiment be?

**{recommendation}**. This recommendation is selected from the requested
candidate list using the measured targeted confusion changes. No next
experiment has been started.
"""
    (args.comparison_dir / 'joint_vs_joint_motion.md').write_text(conclusion)

    print('\nJOINT BASELINE\n--------------')
    print(f"Top-1: {float(baseline['top1']):.4f}")
    print(f"Top-5: {float(baseline['top5']):.4f}")
    print('\nJOINT MOTION\n------------')
    print(f"Best epoch: {selection['best_epoch']}")
    print(f'Top-1: {motion_top1:.4f}')
    print(f'Top-5: {motion_top5:.4f}')
    print('\nGLOBAL DIFFERENCE\n-----------------')
    print(f'ΔTop-1: {delta_top1:+.4f}')
    print(f'ΔTop-5: {delta_top5:+.4f}')
    print('\nMOST IMPROVED CLASSES\n---------------------')
    print(improved_table)
    print('\nMOST DEGRADED CLASSES\n---------------------')
    print(degraded_table)
    print('\nTARGETED CONFUSIONS\n-------------------')
    for row in targeted:
        print(
            f"{row['ground_truth']} → {row['predicted']}: "
            f"Joint={row['joint_error_count']}, "
            f"Motion={row['motion_error_count']}, "
            f"Δ={int(row['difference']):+d}")
    print('\nCONCLUSION\n----------')
    print('Joint Motion improved overall Top-1.' if delta_top1 > 0 else
          'Joint Motion did not improve overall Top-1.')
    print('\nRECOMMENDED NEXT EXPERIMENT\n---------------------------')
    print(recommendation)


if __name__ == '__main__':
    main()

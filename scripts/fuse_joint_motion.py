#!/usr/bin/env python3
"""Evaluate fixed 50/50 score fusion of ST-GCN Joint and Joint Motion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np

matplotlib.use('Agg')

from scripts.analyze_stgcn_results import (
    confusion_counts,
    normalize_rows,
    plot_confusion_matrix,
    write_csv,
)
from scripts.stgcn_evaluation_common import NTU60_CLASS_NAMES

NUM_SAMPLES = 16487
NUM_CLASSES = 60
JOINT_BASELINE_TOP1 = 0.8823
JOINT_BASELINE_TOP5 = 0.9877
EXPECTED_MOTION_TOP1 = 0.8825
EXPECTED_MOTION_TOP5 = 0.9874
ALPHA_JOINT = 0.5
ALPHA_MOTION = 0.5
# The published values have four decimal places. A complete prediction bundle
# must round back to those values; a looser tolerance could silently accept a
# different checkpoint/run.
METRIC_TOLERANCE = 0.0001

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
        '--joint-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/evaluation')
    parser.add_argument(
        '--motion-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/experiments/joint_motion')
    parser.add_argument(
        '--output-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/fusion')
    return parser.parse_args()


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is required to verify/repair sample alignment')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'sample_id', 'ground_truth_id'}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f'{path} is missing CSV columns: {sorted(missing)}')
    return rows


def load_bundle(directory: Path, name: str) -> dict:
    paths = {
        'y_true': directory / 'y_true.npy',
        'y_score': directory / 'y_score.npy',
        'predictions': directory / 'predictions.csv',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'{name} prediction bundle is incomplete: {missing}')
    y_true = np.asarray(np.load(paths['y_true']), dtype=np.int64)
    y_score = np.asarray(np.load(paths['y_score']), dtype=np.float64)
    rows = read_prediction_rows(paths['predictions'])
    if y_true.shape != (NUM_SAMPLES,):
        raise RuntimeError(
            f'{name} y_true shape must be ({NUM_SAMPLES},), got {y_true.shape}')
    if y_score.shape != (NUM_SAMPLES, NUM_CLASSES):
        raise RuntimeError(
            f'{name} score shape must be ({NUM_SAMPLES}, {NUM_CLASSES}), '
            f'got {y_score.shape}')
    if len(rows) != NUM_SAMPLES:
        raise RuntimeError(
            f'{name} predictions.csv has {len(rows)} rows, expected '
            f'{NUM_SAMPLES}')
    if not np.all((0 <= y_true) & (y_true < NUM_CLASSES)):
        raise RuntimeError(f'{name} labels fall outside 0..59')
    if not np.all(np.isfinite(y_score)):
        raise RuntimeError(f'{name} scores contain NaN or infinity')
    if np.any(y_score < -1e-7):
        raise RuntimeError(
            f'{name} y_score is not the exported probability array')
    row_sums = y_score.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise RuntimeError(
            f'{name} score rows do not sum to one; do not mix logits and '
            'probabilities in 50/50 fusion')
    sample_ids = [row['sample_id'] for row in rows]
    if len(set(sample_ids)) != NUM_SAMPLES:
        raise RuntimeError(f'{name} sample_id values are not unique')
    csv_labels = np.asarray(
        [int(row['ground_truth_id']) for row in rows], dtype=np.int64)
    if not np.array_equal(csv_labels, y_true):
        raise RuntimeError(f'{name} predictions.csv and y_true.npy disagree')
    if 'predicted_id' in rows[0]:
        csv_predictions = np.asarray(
            [int(row['predicted_id']) for row in rows], dtype=np.int64)
        score_predictions = np.argmax(y_score, axis=1)
        if not np.array_equal(csv_predictions, score_predictions):
            raise RuntimeError(
                f'{name} predictions.csv and y_score.npy disagree')
    return {
        'name': name,
        'y_true': y_true,
        'y_score': y_score,
        'rows': rows,
        'sample_ids': sample_ids,
    }


def align_motion_to_joint(joint: dict, motion: dict) -> tuple[dict, dict]:
    if joint['sample_ids'] == motion['sample_ids']:
        if not np.array_equal(joint['y_true'], motion['y_true']):
            raise RuntimeError(
                'sample IDs match but Joint and Motion labels differ')
        return motion, {
            'method': 'already_aligned_by_sample_id',
            'reordered': False,
            'labels_identical': True,
        }

    joint_ids = set(joint['sample_ids'])
    motion_ids = set(motion['sample_ids'])
    if joint_ids != motion_ids:
        missing = sorted(joint_ids - motion_ids)[:10]
        extra = sorted(motion_ids - joint_ids)[:10]
        raise RuntimeError(
            'Joint/Motion sample sets differ; '
            f'missing from Motion={missing}, extra in Motion={extra}')
    motion_index = {
        sample_id: index
        for index, sample_id in enumerate(motion['sample_ids'])
    }
    order = np.asarray(
        [motion_index[sample_id] for sample_id in joint['sample_ids']],
        dtype=np.int64)
    aligned = {
        **motion,
        'y_true': motion['y_true'][order],
        'y_score': motion['y_score'][order],
        'rows': [motion['rows'][index] for index in order],
        'sample_ids': list(joint['sample_ids']),
    }
    if not np.array_equal(joint['y_true'], aligned['y_true']):
        raise RuntimeError(
            'Joint/Motion labels still differ after sample_id alignment')
    return aligned, {
        'method': 'motion_reordered_by_sample_id',
        'reordered': True,
        'labels_identical': True,
    }


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    ranking = np.argsort(-y_score, axis=1, kind='stable')
    y_pred = ranking[:, 0].astype(np.int64)
    top1 = float(np.mean(y_pred == y_true))
    top5 = float(np.mean(np.any(ranking[:, :5] == y_true[:, None], axis=1)))
    class_accuracy = []
    for class_id in range(NUM_CLASSES):
        mask = y_true == class_id
        if not np.any(mask):
            raise RuntimeError(f'xsub_val has no samples for class {class_id}')
        class_accuracy.append(float(np.mean(y_pred[mask] == class_id)))
    return {
        'top1': top1,
        'top5': top5,
        'mean1': float(np.mean(class_accuracy)),
        'y_pred': y_pred,
        'ranking': ranking,
        'class_accuracy': np.asarray(class_accuracy, dtype=np.float64),
    }


def verify_source_metrics(joint: dict, motion: dict) -> None:
    checks = (
        ('Joint Top-1', joint['top1'], JOINT_BASELINE_TOP1),
        ('Joint Top-5', joint['top5'], JOINT_BASELINE_TOP5),
        ('Motion Top-1', motion['top1'], EXPECTED_MOTION_TOP1),
        ('Motion Top-5', motion['top5'], EXPECTED_MOTION_TOP5),
    )
    failures = [
        f'{name}: measured={measured:.6f}, expected={expected:.6f}'
        for name, measured, expected in checks
        if abs(measured - expected) > METRIC_TOLERANCE
    ]
    if failures:
        raise RuntimeError(
            'source scores do not reproduce the stated experiments:\n' +
            '\n'.join(failures))


def build_per_class_rows(
        y_true: np.ndarray, joint: dict, motion: dict,
        fusion: dict) -> list[dict]:
    rows = []
    for class_id, action in enumerate(NTU60_CLASS_NAMES):
        samples = int(np.sum(y_true == class_id))
        rows.append({
            'class_id': class_id,
            'action': action,
            'num_samples': samples,
            'joint_accuracy': float(joint['class_accuracy'][class_id]),
            'motion_accuracy': float(motion['class_accuracy'][class_id]),
            'fusion_accuracy': float(fusion['class_accuracy'][class_id]),
            'fusion_vs_joint_delta': float(
                fusion['class_accuracy'][class_id] -
                joint['class_accuracy'][class_id]),
        })
    return rows


def targeted_confusion_rows(
        y_true: np.ndarray, joint_pred: np.ndarray,
        motion_pred: np.ndarray, fusion_pred: np.ndarray) -> list[dict]:
    rows = []
    for ground_truth, predicted in TARGET_PAIRS:
        mask = y_true == ground_truth
        rows.append({
            'ground_truth_id': ground_truth,
            'ground_truth': NTU60_CLASS_NAMES[ground_truth],
            'predicted_id': predicted,
            'predicted': NTU60_CLASS_NAMES[predicted],
            'joint_errors': int(np.sum(mask & (joint_pred == predicted))),
            'motion_errors': int(np.sum(mask & (motion_pred == predicted))),
            'fusion_errors': int(np.sum(mask & (fusion_pred == predicted))),
        })
    return rows


def disagreement_analysis(
        y_true: np.ndarray, joint_pred: np.ndarray,
        motion_pred: np.ndarray, fusion_pred: np.ndarray) -> dict:
    same = joint_pred == motion_pred
    different = ~same
    different_count = int(np.sum(different))
    if different_count == 0:
        raise RuntimeError('Joint and Motion never disagree; check input bundles')
    joint_correct = joint_pred == y_true
    motion_correct = motion_pred == y_true
    fusion_correct = fusion_pred == y_true

    def summarize(mask: np.ndarray) -> dict:
        count = int(np.sum(mask))
        return {
            'count': count,
            'fraction_of_all_samples': count / NUM_SAMPLES,
            'fraction_of_disagreements': count / different_count,
        }

    return {
        'num_samples': NUM_SAMPLES,
        'same_prediction': {
            'count': int(np.sum(same)),
            'fraction': float(np.mean(same)),
        },
        'different_prediction': {
            'count': different_count,
            'fraction': float(np.mean(different)),
        },
        'among_disagreements': {
            'joint_correct_motion_wrong': summarize(
                different & joint_correct & ~motion_correct),
            'motion_correct_joint_wrong': summarize(
                different & motion_correct & ~joint_correct),
            'both_wrong': summarize(
                different & ~joint_correct & ~motion_correct),
            'fusion_correct': summarize(different & fusion_correct),
        },
    }


def write_predictions(
        path: Path, sample_ids: list[str], y_true: np.ndarray,
        fusion_score: np.ndarray, fusion: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        fields = [
            'sample_id', 'ground_truth_id', 'ground_truth_name',
            'predicted_id', 'predicted_name', 'confidence', 'correct',
            'top5_classes', 'top5_names', 'top5_scores',
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            predicted = int(fusion['y_pred'][index])
            top5 = fusion['ranking'][index, :5]
            writer.writerow({
                'sample_id': sample_id,
                'ground_truth_id': int(y_true[index]),
                'ground_truth_name': NTU60_CLASS_NAMES[y_true[index]],
                'predicted_id': predicted,
                'predicted_name': NTU60_CLASS_NAMES[predicted],
                'confidence': float(fusion_score[index, predicted]),
                'correct': bool(predicted == y_true[index]),
                'top5_classes': json.dumps(top5.tolist()),
                'top5_names': json.dumps(
                    [NTU60_CLASS_NAMES[item] for item in top5]),
                'top5_scores': json.dumps(
                    [float(fusion_score[index, item]) for item in top5]),
            })


def class_table(rows: list[dict]) -> str:
    lines = [
        '| Rank | Class | Joint | Motion | Fusion | Δ Fusion−Joint |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['class_id']}: {row['action']} | "
            f"{row['joint_accuracy']:.4f} | {row['motion_accuracy']:.4f} | "
            f"{row['fusion_accuracy']:.4f} | "
            f"{row['fusion_vs_joint_delta']:+.4f} |")
    return '\n'.join(lines)


def targeted_table(rows: list[dict]) -> str:
    lines = [
        '| Ground truth → predicted | Joint | Motion | Fusion |',
        '|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['ground_truth_id']}: {row['ground_truth']} → "
            f"{row['predicted_id']}: {row['predicted']} | "
            f"{row['joint_errors']} | {row['motion_errors']} | "
            f"{row['fusion_errors']} |")
    return '\n'.join(lines)


def percent(value: float) -> str:
    return f'{100 * value:.2f}%'


def main() -> None:
    args = parse_args()
    if not np.isclose(ALPHA_JOINT, 0.5) or not np.isclose(ALPHA_MOTION, 0.5):
        raise RuntimeError('this milestone permits only fixed 0.5/0.5 fusion')
    joint_bundle = load_bundle(args.joint_dir, 'Joint')
    motion_bundle = load_bundle(args.motion_dir, 'Joint Motion')
    motion_bundle, alignment = align_motion_to_joint(
        joint_bundle, motion_bundle)
    y_true = joint_bundle['y_true']
    if not np.array_equal(y_true, motion_bundle['y_true']):
        raise RuntimeError('Joint and aligned Motion y_true arrays differ')

    joint = calculate_metrics(y_true, joint_bundle['y_score'])
    motion = calculate_metrics(y_true, motion_bundle['y_score'])
    verify_source_metrics(joint, motion)
    fusion_score = (
        ALPHA_JOINT * joint_bundle['y_score'] +
        ALPHA_MOTION * motion_bundle['y_score'])
    if not np.allclose(fusion_score.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError('fused probability rows do not sum to one')
    fusion = calculate_metrics(y_true, fusion_score)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    per_class = build_per_class_rows(y_true, joint, motion, fusion)
    write_csv(
        output / 'per_class_accuracy.csv', per_class,
        ['class_id', 'action', 'num_samples', 'joint_accuracy',
         'motion_accuracy', 'fusion_accuracy', 'fusion_vs_joint_delta'])
    targeted = targeted_confusion_rows(
        y_true, joint['y_pred'], motion['y_pred'], fusion['y_pred'])
    write_csv(
        output / 'targeted_confusions.csv', targeted,
        ['ground_truth_id', 'ground_truth', 'predicted_id', 'predicted',
         'joint_errors', 'motion_errors', 'fusion_errors'])
    disagreement = disagreement_analysis(
        y_true, joint['y_pred'], motion['y_pred'], fusion['y_pred'])
    (output / 'disagreement_analysis.json').write_text(
        json.dumps(disagreement, indent=2))
    write_predictions(
        output / 'predictions.csv', joint_bundle['sample_ids'], y_true,
        fusion_score, fusion)
    plot_confusion_matrix(
        normalize_rows(confusion_counts(y_true, fusion['y_pred'])),
        list(range(NUM_CLASSES)), output / 'confusion_matrix.png',
        'NTU60 XSub — ST-GCN Joint + Joint Motion 50/50 fusion', True)

    delta_top1 = fusion['top1'] - joint['top1']
    delta_top5 = fusion['top5'] - joint['top5']
    metrics = {
        'model': 'ST-GCN score-level late fusion',
        'dataset': 'NTU60',
        'protocol': 'xsub',
        'num_samples': NUM_SAMPLES,
        'num_classes': NUM_CLASSES,
        'weights': {'joint': ALPHA_JOINT, 'joint_motion': ALPHA_MOTION},
        'alpha_tuned': False,
        'alignment': alignment,
        'joint': {
            'top1': joint['top1'], 'top5': joint['top5'],
            'mean1': joint['mean1'],
        },
        'joint_motion': {
            'top1': motion['top1'], 'top5': motion['top5'],
            'mean1': motion['mean1'],
        },
        'fusion': {
            'top1': fusion['top1'], 'top5': fusion['top5'],
            'mean1': fusion['mean1'],
        },
        'delta_vs_joint_baseline': {
            'top1': delta_top1, 'top5': delta_top5,
        },
        'frozen_joint_reference': {
            'top1': JOINT_BASELINE_TOP1,
            'top5': JOINT_BASELINE_TOP5,
        },
    }
    (output / 'joint_motion_fusion_metrics.json').write_text(
        json.dumps(metrics, indent=2))

    improved = sorted(
        per_class,
        key=lambda row: (-row['fusion_vs_joint_delta'], row['class_id']))[:10]
    degraded = sorted(
        per_class,
        key=lambda row: (row['fusion_vs_joint_delta'], row['class_id']))[:10]
    disagreement_values = disagreement['among_disagreements']
    if delta_top1 > 0:
        conclusion = (
            'Fixed 50/50 late fusion improves Top-1 over Joint, supporting '
            'complementarity without validation-set alpha tuning.')
    else:
        conclusion = (
            'Fixed 50/50 late fusion does not improve Top-1 over Joint; the '
            'measured disagreement is not converted into a global gain.')
    report = f"""# Joint + Joint Motion fixed 50/50 late fusion

No alpha search or neural-network training was performed.

| Representation | Top-1 | Top-5 | Mean-class accuracy |
|---|---:|---:|---:|
| Joint | {joint['top1']:.4f} | {joint['top5']:.4f} | {joint['mean1']:.4f} |
| Joint Motion | {motion['top1']:.4f} | {motion['top5']:.4f} | {motion['mean1']:.4f} |
| Joint + Motion (0.5/0.5) | {fusion['top1']:.4f} | {fusion['top5']:.4f} | {fusion['mean1']:.4f} |

## Delta versus frozen Joint baseline

- ΔTop-1: `{delta_top1:+.4f}`
- ΔTop-5: `{delta_top5:+.4f}`

## Model disagreement

- Same prediction: `{percent(disagreement['same_prediction']['fraction'])}`
- Different prediction: `{percent(disagreement['different_prediction']['fraction'])}`
- Joint correct / Motion wrong among disagreements:
  `{percent(disagreement_values['joint_correct_motion_wrong']['fraction_of_disagreements'])}`
- Motion correct / Joint wrong among disagreements:
  `{percent(disagreement_values['motion_correct_joint_wrong']['fraction_of_disagreements'])}`
- Both wrong among disagreements:
  `{percent(disagreement_values['both_wrong']['fraction_of_disagreements'])}`
- Fusion correct among disagreements:
  `{percent(disagreement_values['fusion_correct']['fraction_of_disagreements'])}`

## 10 classes most improved by fusion

{class_table(improved)}

## 10 classes most degraded by fusion

{class_table(degraded)}

## Targeted confusions

{targeted_table(targeted)}

## Conclusion

{conclusion}
"""
    (output / 'report.md').write_text(report)

    print('\nJOINT\n-----')
    print(f"Top-1: {joint['top1']:.4f}")
    print(f"Top-5: {joint['top5']:.4f}")
    print('\nJOINT MOTION\n------------')
    print(f"Top-1: {motion['top1']:.4f}")
    print(f"Top-5: {motion['top5']:.4f}")
    print('\n50/50 FUSION\n------------')
    print(f"Top-1: {fusion['top1']:.4f}")
    print(f"Top-5: {fusion['top5']:.4f}")
    print(f"Mean accuracy: {fusion['mean1']:.4f}")
    print('\nDELTA VS JOINT\n--------------')
    print(f'Top-1: {delta_top1:+.4f}')
    print(f'Top-5: {delta_top5:+.4f}')
    print('\nMODEL DISAGREEMENT\n------------------')
    print('Same prediction:', percent(
        disagreement['same_prediction']['fraction']))
    print('Different prediction:', percent(
        disagreement['different_prediction']['fraction']))
    print('\nWhen predictions differ:')
    print('Joint correct only:', percent(
        disagreement_values[
            'joint_correct_motion_wrong']['fraction_of_disagreements']))
    print('Motion correct only:', percent(
        disagreement_values[
            'motion_correct_joint_wrong']['fraction_of_disagreements']))
    print('Both wrong:', percent(
        disagreement_values['both_wrong']['fraction_of_disagreements']))
    print('Fusion correct:', percent(
        disagreement_values['fusion_correct']['fraction_of_disagreements']))
    print('\n10 CLASSES MOST IMPROVED\n------------------------')
    print(class_table(improved))
    print('\n10 CLASSES MOST DEGRADED\n------------------------')
    print(class_table(degraded))
    print('\nTARGETED CONFUSIONS\n-------------------')
    print(targeted_table(targeted))
    print('\nCONCLUSION\n----------')
    print(conclusion)


if __name__ == '__main__':
    main()

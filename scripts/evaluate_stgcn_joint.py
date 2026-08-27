#!/usr/bin/env python3
"""Freeze and independently evaluate an ST-GCN skeleton checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

# MMEngine 0.10.x calls torch.load without an explicit weights_only argument.
# These experiment checkpoints are trusted, locally produced MMEngine files.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

from scripts.stgcn_evaluation_common import (
    NTU60_CLASS_NAMES,
    checkpoint_epoch,
    class_name_mapping,
    load_training_records,
    select_best_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'configs/stgcn_ntu60_xsub_80e_resume.py')
    parser.add_argument(
        '--ann-file', type=Path,
        default=PROJECT_ROOT / 'data/skeleton/ntu60_2d.pkl')
    parser.add_argument(
        '--work-dir', action='append', type=Path, dest='work_dirs',
        help='Training workdir to search; may be supplied more than once.')
    parser.add_argument(
        '--frozen-checkpoint', type=Path,
        default=PROJECT_ROOT / 'artifacts/checkpoints/'
        'stgcn_joint_ntu60_xsub_best.pth')
    parser.add_argument(
        '--evaluation-dir', type=Path,
        default=PROJECT_ROOT / 'artifacts/evaluation')
    parser.add_argument(
        '--input-representation', choices=('joint', 'joint_motion'),
        default='joint')
    parser.add_argument(
        '--metrics-name', default='baseline_metrics.json',
        help='Metrics filename inside --evaluation-dir.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--top1-tolerance', type=float, default=0.002,
        help='Maximum absolute independent-vs-training Top-1 difference.')
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def probabilities_from_score(score: np.ndarray) -> tuple[np.ndarray, str]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.shape != (60,):
        raise RuntimeError(f'expected 60 model scores, got {score.shape}')
    if not np.all(np.isfinite(score)):
        raise RuntimeError('model scores contain NaN or infinity')
    if (np.all(score >= -1e-7)
            and np.isclose(score.sum(), 1.0, atol=1e-4)):
        probability = np.clip(score, 0, None)
        probability /= probability.sum()
        return probability.astype(np.float32), 'probability'
    shifted = score - score.max()
    probability = np.exp(shifted)
    probability /= probability.sum()
    return probability.astype(np.float32), 'logit_softmax'


def build_validation_loader(config: Config, ann_file: Path, seed: int):
    dataloader_cfg = copy.deepcopy(config.val_dataloader)
    dataloader_cfg.dataset.ann_file = str(ann_file)
    dataloader_cfg.dataset.split = 'xsub_val'
    dataloader_cfg.dataset.test_mode = True
    dataloader_cfg.sampler.shuffle = False

    # MMEngine BaseDataset serializes annotations by default and then clears
    # dataset.data_list. Carry the stable dataset index and identifier through
    # PackActionInputs instead of trying to read that cleared list later.
    pack_transforms = [
        transform for transform in dataloader_cfg.dataset.pipeline
        if transform.get('type') == 'PackActionInputs'
    ]
    if len(pack_transforms) != 1:
        raise RuntimeError(
            'validation pipeline must contain exactly one PackActionInputs')
    default_meta_keys = ('img_shape', 'img_key', 'video_id', 'timestamp')
    existing_meta_keys = tuple(
        pack_transforms[0].get('meta_keys', default_meta_keys))
    pack_transforms[0]['meta_keys'] = tuple(dict.fromkeys((
        *existing_meta_keys, 'sample_idx', 'frame_dir', 'filename')))

    return Runner.build_dataloader(
        dataloader_cfg, seed=seed, diff_rank_seed=False)


def infer(model, dataloader, device: torch.device):
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    y_true: list[int] = []
    scores: list[np.ndarray] = []
    sample_ids: list[str] = []
    score_kinds: set[str] = set()
    sample_index = 0

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch_index, data_batch in enumerate(dataloader, start=1):
            outputs = model.test_step(data_batch)
            for output in outputs:
                probability, score_kind = probabilities_from_score(
                    output.pred_score.detach().cpu().numpy())
                label = int(output.gt_label.item())
                if sample_index >= dataset_size:
                    raise RuntimeError('model returned more samples than dataset')

                metadata = output.metainfo
                if 'sample_idx' not in metadata:
                    raise RuntimeError(
                        'prediction has no sample_idx metadata; check the '
                        'validation PackActionInputs configuration')
                data_index = int(metadata['sample_idx'])
                if data_index != sample_index:
                    raise RuntimeError(
                        'dataloader order mismatch: expected dataset index '
                        f'{sample_index}, received {data_index}')
                identifier = metadata.get(
                    'frame_dir', metadata.get('filename'))
                if identifier is None:
                    raise RuntimeError(
                        f'prediction {data_index} has no frame_dir/filename '
                        'metadata')
                y_true.append(label)
                scores.append(probability)
                sample_ids.append(str(identifier))
                score_kinds.add(score_kind)
                sample_index += 1
            if batch_index % 100 == 0 or batch_index == len(dataloader):
                print(
                    f'inference: {batch_index}/{len(dataloader)} batches, '
                    f'{sample_index}/{dataset_size} samples', flush=True)

    if sample_index != dataset_size:
        raise RuntimeError(
            f'inference produced {sample_index} predictions for '
            f'{dataset_size} samples')
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError('validation sample identifiers are not unique')
    return (
        np.asarray(y_true, dtype=np.int64),
        np.stack(scores).astype(np.float32),
        sample_ids,
        sorted(score_kinds),
    )


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if y_score.shape != (len(y_true), 60):
        raise RuntimeError(
            f'expected score shape ({len(y_true)}, 60), got {y_score.shape}')
    if not np.all((0 <= y_true) & (y_true < 60)):
        raise RuntimeError('validation labels fall outside the range 0..59')
    if not np.all(np.isfinite(y_score)):
        raise RuntimeError('validation probabilities contain NaN or infinity')
    if not np.allclose(y_score.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError('validation probability rows do not sum to one')
    ranking = np.argsort(-y_score, axis=1, kind='stable')
    y_pred = ranking[:, 0]
    top1 = float(np.mean(y_pred == y_true))
    top5 = float(np.mean(np.any(ranking[:, :5] == y_true[:, None], axis=1)))
    class_accuracy = []
    for class_id in range(60):
        mask = y_true == class_id
        if not np.any(mask):
            raise RuntimeError(f'xsub_val contains no samples for class {class_id}')
        class_accuracy.append(float(np.mean(y_pred[mask] == class_id)))
    return {
        'top1': top1,
        'top5': top5,
        'mean1': float(np.mean(class_accuracy)),
        'y_pred': y_pred.astype(np.int64),
        'ranking': ranking,
    }


def write_predictions(
        path: Path, sample_ids: list[str], y_true: np.ndarray,
        y_pred: np.ndarray, y_score: np.ndarray, ranking: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        fieldnames = [
            'sample_id', 'ground_truth_id', 'ground_truth_name',
            'predicted_id', 'predicted_name', 'confidence', 'correct',
            'top5_classes', 'top5_names', 'top5_scores',
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            top5 = ranking[index, :5]
            writer.writerow({
                'sample_id': sample_id,
                'ground_truth_id': int(y_true[index]),
                'ground_truth_name': NTU60_CLASS_NAMES[y_true[index]],
                'predicted_id': int(y_pred[index]),
                'predicted_name': NTU60_CLASS_NAMES[y_pred[index]],
                'confidence': f'{float(y_score[index, y_pred[index]]):.9f}',
                'correct': bool(y_pred[index] == y_true[index]),
                'top5_classes': json.dumps(top5.tolist()),
                'top5_names': json.dumps(
                    [NTU60_CLASS_NAMES[item] for item in top5]),
                'top5_scores': json.dumps(
                    [round(float(y_score[index, item]), 9) for item in top5]),
            })


def main() -> None:
    args = parse_args()
    if Path(args.metrics_name).name != args.metrics_name:
        raise ValueError('--metrics-name must be a filename, not a path')
    work_dirs = args.work_dirs or [
        PROJECT_ROOT / 'work_dirs/stgcn_ntu60_xsub_40e',
        PROJECT_ROOT / 'work_dirs/stgcn_ntu60_xsub_80e_resume',
    ]
    missing = [
        str(path) for path in (args.config, args.ann_file) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError('required input files missing: ' + ', '.join(missing))

    records = load_training_records(work_dirs)
    best_source, best_record = select_best_checkpoint(work_dirs, records)
    best_epoch = checkpoint_epoch(best_source)
    logged_top1 = float(best_record['val_acc_top1'])
    logged_top5 = float(best_record['val_acc_top5'])
    print('Best checkpoint:', best_source)
    print('Best epoch:', best_epoch)
    print('Best Top-1:', f'{logged_top1:.4f}')
    print('Best Top-5:', f'{logged_top5:.4f}')

    atomic_copy(best_source, args.frozen_checkpoint)
    source_hash = sha256(best_source)
    frozen_hash = sha256(args.frozen_checkpoint)
    if source_hash != frozen_hash:
        raise RuntimeError('frozen checkpoint hash differs from source')

    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    # Never let a failed re-evaluation leave usable-looking predictions from a
    # previous run. The independently computed outputs below are the only
    # accepted inputs to the analysis stage.
    for stale_name in (
            args.metrics_name, 'evaluation_discrepancy.json',
            'predictions.csv', 'y_true.npy', 'y_pred.npy', 'y_score.npy'):
        (args.evaluation_dir / stale_name).unlink(missing_ok=True)

    selection = {
        'source_checkpoint': str(best_source.resolve()),
        'frozen_checkpoint': str(args.frozen_checkpoint.resolve()),
        'sha256': frozen_hash,
        'best_epoch': best_epoch,
        'logged_top1': logged_top1,
        'logged_top5': logged_top5,
        'metric_source': best_record.get('source'),
    }
    (args.evaluation_dir / 'best_checkpoint.json').write_text(
        json.dumps(selection, indent=2))
    (args.evaluation_dir / 'class_names.json').write_text(
        json.dumps(class_name_mapping(), indent=2, ensure_ascii=False))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    register_all_modules(init_default_scope=True)
    config = Config.fromfile(str(args.config))
    dataloader = build_validation_loader(config, args.ann_file, args.seed)
    model = MODELS.build(config.model)
    load_checkpoint(model, str(args.frozen_checkpoint), map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true, y_score, sample_ids, score_kinds = infer(
        model, dataloader, device)
    calculated = compute_metrics(y_true, y_score)
    y_pred = calculated.pop('y_pred')
    ranking = calculated.pop('ranking')

    discrepancy = abs(calculated['top1'] - logged_top1)
    accepted = discrepancy <= args.top1_tolerance
    metrics = {
        'model': 'ST-GCN',
        'dataset': 'NTU60',
        'protocol': 'xsub',
        'input': args.input_representation,
        'num_classes': 60,
        'train_samples': 40091,
        'validation_samples': len(y_true),
        'top1': calculated['top1'],
        'top5': calculated['top5'],
        'mean1': calculated['mean1'],
        'best_epoch': best_epoch,
        'checkpoint': str(args.frozen_checkpoint.resolve()),
        'logged_top1': logged_top1,
        'logged_top5': logged_top5,
        'top1_absolute_difference': discrepancy,
        'acceptance_tolerance': args.top1_tolerance,
        'evaluation_status': 'accepted' if accepted else 'failed',
        'score_representation': score_kinds,
    }
    metrics_path = args.evaluation_dir / args.metrics_name
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if not accepted:
        discrepancy_path = args.evaluation_dir / 'evaluation_discrepancy.json'
        discrepancy_path.write_text(json.dumps({
            **metrics,
            'status': 'failed',
        }, indent=2))
        raise RuntimeError(
            'independent Top-1 differs from the logged best by '
            f'{discrepancy:.6f}, exceeding tolerance '
            f'{args.top1_tolerance:.6f}; analysis stopped')

    np.save(args.evaluation_dir / 'y_true.npy', y_true)
    np.save(args.evaluation_dir / 'y_pred.npy', y_pred)
    np.save(args.evaluation_dir / 'y_score.npy', y_score)
    write_predictions(
        args.evaluation_dir / 'predictions.csv', sample_ids, y_true,
        y_pred, y_score, ranking)
    print('Independent evaluation accepted; predictions exported.')


if __name__ == '__main__':
    main()

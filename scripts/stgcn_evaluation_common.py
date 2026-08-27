"""Shared constants and log/checkpoint helpers for ST-GCN evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

NTU60_CLASS_NAMES = (
    'drink water',
    'eat meal/snack',
    'brushing teeth',
    'brushing hair',
    'drop',
    'pickup',
    'throw',
    'sitting down',
    'standing up (from sitting position)',
    'clapping',
    'reading',
    'writing',
    'tear up paper',
    'wear jacket',
    'take off jacket',
    'wear a shoe',
    'take off a shoe',
    'wear on glasses',
    'take off glasses',
    'put on a hat/cap',
    'take off a hat/cap',
    'cheer up',
    'hand waving',
    'kicking something',
    'reach into pocket',
    'hopping (one foot jumping)',
    'jump up',
    'make a phone call/answer phone',
    'playing with phone/tablet',
    'typing on a keyboard',
    'pointing to something with finger',
    'taking a selfie',
    'check time (from watch)',
    'rub two hands together',
    'nod head/bow',
    'shake head',
    'wipe face',
    'salute',
    'put palms together',
    'cross hands in front (say stop)',
    'sneeze/cough',
    'staggering',
    'falling',
    'touch head (headache)',
    'touch chest (stomachache/heart pain)',
    'touch back (backache)',
    'touch neck (neckache)',
    'nausea or vomiting condition',
    'use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person',
    'kicking other person',
    'pushing other person',
    'pat on back of other person',
    'point finger at the other person',
    'hugging other person',
    'giving something to other person',
    "touch other person's pocket",
    'handshaking',
    'walking towards each other',
    'walking apart from each other',
)

CUSTOM_METRIC_RE = re.compile(
    r'Outer epoch\s+(?P<epoch>\d+)\s+/\s+effective epoch\s+'
    r'(?P<effective>\d+):\s+loss=(?P<loss>[-+\d.eE]+),\s+'
    r'acc/top1=(?P<top1>[-+\d.eE]+),\s+'
    r'acc/top5=(?P<top5>[-+\d.eE]+),\s+'
    r'lr=(?P<lr>[^,]+),\s+GPU=(?P<gpu>[-+\d.eE]+)\s+MiB,\s+'
    r'wall=(?P<wall>[-+\d.eE]+)s')

STANDARD_VAL_RE = re.compile(
    r'Epoch\(val\)\s+\[(?P<epoch>\d+)\].*?'
    r'acc/top1:\s*(?P<top1>[-+\d.eE]+)\s+'
    r'acc/top5:\s*(?P<top5>[-+\d.eE]+)\s+'
    r'acc/mean1:\s*(?P<mean1>[-+\d.eE]+)')

CHECKPOINT_EPOCH_RE = re.compile(r'epoch_(\d+)\.pth$')


def class_name_mapping() -> dict[str, str]:
    """Return the canonical zero-based NTU60 label mapping."""
    if len(NTU60_CLASS_NAMES) != 60:
        raise RuntimeError('NTU60 class mapping must contain exactly 60 names')
    return {str(index): name for index, name in enumerate(NTU60_CLASS_NAMES)}


def checkpoint_epoch(path: Path) -> int | None:
    """Extract a one-based outer epoch from an MMEngine checkpoint name."""
    match = CHECKPOINT_EPOCH_RE.search(path.name)
    return int(match.group(1)) if match else None


def _maximum_epoch_for_directory(path: Path) -> int | None:
    checkpoint_epochs = [
        epoch for checkpoint in path.glob('*.pth')
        if (epoch := checkpoint_epoch(checkpoint)) is not None
    ]
    if checkpoint_epochs:
        # The actual files take precedence over a historical workdir name. A
        # resumed run may have been written back into a directory ending 40e.
        return max(checkpoint_epochs)
    name = path.name.lower()
    if '80e' in name or 'resume' in name:
        return 16
    if '40e' in name:
        return 8
    return None


def _merge_record(records: dict[int, dict], epoch: int, update: dict) -> None:
    record = records.setdefault(
        epoch, {'outer_epoch': epoch, 'effective_epoch': epoch * 5})
    for key, value in update.items():
        if value is not None:
            record[key] = value


def _parse_jsonl(path: Path, maximum_epoch: int | None) -> dict[int, dict]:
    raw = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    raw_epochs = {int(item['outer_epoch']) for item in raw}
    legacy_shift = int(
        maximum_epoch == 8 and 1 not in raw_epochs
        and set(range(2, 10)).issubset(raw_epochs))
    records: dict[int, dict] = {}
    for item in raw:
        epoch = int(item['outer_epoch']) - legacy_shift
        if maximum_epoch is not None and not 1 <= epoch <= maximum_epoch:
            continue
        normalized = dict(item)
        normalized['outer_epoch'] = epoch
        normalized['effective_epoch'] = epoch * 5
        normalized['source'] = str(path)
        records[epoch] = normalized
    return records


def _parse_text_log(path: Path, maximum_epoch: int | None) -> dict[int, dict]:
    text = path.read_text(errors='replace')
    records: dict[int, dict] = {}
    custom_matches = list(CUSTOM_METRIC_RE.finditer(text))
    raw_custom_epochs = {int(match.group('epoch')) for match in custom_matches}
    legacy_shift = int(
        maximum_epoch == 8 and 1 not in raw_custom_epochs
        and set(range(2, 10)).issubset(raw_custom_epochs))

    for match in custom_matches:
        epoch = int(match.group('epoch')) - legacy_shift
        if maximum_epoch is not None and not 1 <= epoch <= maximum_epoch:
            continue
        _merge_record(records, epoch, {
            'effective_epoch': epoch * 5,
            'train_loss': float(match.group('loss')),
            'val_acc_top1': float(match.group('top1')),
            'val_acc_top5': float(match.group('top5')),
            'learning_rate': (
                None if match.group('lr').strip() == 'None'
                else float(match.group('lr'))),
            'gpu_memory_mb': float(match.group('gpu')),
            'epoch_wall_time_sec': float(match.group('wall')),
            'source': str(path),
        })

    for match in STANDARD_VAL_RE.finditer(text):
        epoch = int(match.group('epoch'))
        if maximum_epoch is not None and not 1 <= epoch <= maximum_epoch:
            continue
        _merge_record(records, epoch, {
            'val_acc_top1': float(match.group('top1')),
            'val_acc_top5': float(match.group('top5')),
            'val_acc_mean1': float(match.group('mean1')),
            'source': str(path),
        })
    return records


def load_training_records(work_dirs: Iterable[Path]) -> dict[int, dict]:
    """Merge per-epoch JSONL and console/MMEngine logs from all workdirs."""
    records: dict[int, dict] = {}
    for work_dir in work_dirs:
        work_dir = Path(work_dir)
        if not work_dir.exists():
            continue
        maximum_epoch = _maximum_epoch_for_directory(work_dir)
        jsonl = work_dir / 'epoch_metrics.jsonl'
        if jsonl.is_file():
            for epoch, item in _parse_jsonl(jsonl, maximum_epoch).items():
                _merge_record(records, epoch, item)

        log_paths = sorted({
            *work_dir.rglob('*.log'),
            *work_dir.rglob('*.txt'),
        })
        for log_path in log_paths:
            for epoch, item in _parse_text_log(
                    log_path, maximum_epoch).items():
                _merge_record(records, epoch, item)
    return dict(sorted(records.items()))


def find_checkpoint_candidates(work_dirs: Iterable[Path]) -> list[Path]:
    """Return unique regular/best checkpoints without following latest links."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for work_dir in work_dirs:
        work_dir = Path(work_dir)
        if not work_dir.exists():
            continue
        for path in sorted(work_dir.glob('*.pth')):
            if (not path.is_file() or path.name == 'latest.pth'
                    or checkpoint_epoch(path) is None):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    return candidates


def select_best_checkpoint(
        work_dirs: Iterable[Path], records: dict[int, dict]) -> tuple[Path, dict]:
    """Select the available checkpoint whose logged validation Top-1 is best."""
    candidates = find_checkpoint_candidates(work_dirs)
    ranked = []
    for path in candidates:
        epoch = checkpoint_epoch(path)
        record = records.get(epoch or -1, {})
        top1 = record.get('val_acc_top1')
        if top1 is not None:
            ranked.append((float(top1), epoch, path, record))
    if not ranked:
        found = ', '.join(str(path) for path in candidates) or 'none'
        metric_epochs = ', '.join(map(str, records)) or 'none'
        raise RuntimeError(
            'No checkpoint could be matched to a logged validation metric. '
            f'Checkpoints: {found}. Metric epochs: {metric_epochs}.')

    # Prefer an explicitly named best checkpoint when duplicate files represent
    # the same winning epoch and metric.
    ranked.sort(
        key=lambda item: (
            item[0], item[1] or -1,
            int(item[2].name.startswith('best_'))),
        reverse=True)
    _, _, path, record = ranked[0]
    return path, record

#!/usr/bin/env python3
"""Disable unused MMPose registries that require MMDetection/full MMCV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--report', type=Path,
        default=Path('artifacts/inference/mmpose_lite_patch.json'))
    return parser.parse_args()


def patch_file(
        path: Path, replacements: tuple[tuple[str, str], ...]) -> bool:
    original = path.read_text()
    updated = original
    for old, new in replacements:
        if old in updated:
            updated = updated.replace(old, new)
        elif new not in updated:
            raise RuntimeError(
                f'MMPose source differs from pinned v1.3.2 at {path}; '
                f'missing {old!r}')
    if updated == original:
        return False
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    temporary.write_text(updated)
    temporary.replace(path)
    return True


def main() -> None:
    args = parse_args()
    import mmpose

    if mmpose.__version__ != '1.3.2':
        raise RuntimeError(
            f'this compatibility patch only supports MMPose 1.3.2, got '
            f'{mmpose.__version__}')
    package = Path(mmpose.__file__).resolve().parent
    heads = package / 'models/heads/__init__.py'
    hybrid = package / 'models/heads/hybrid_heads/__init__.py'
    if not heads.is_file() or not hybrid.is_file():
        raise FileNotFoundError(
            f'cannot find pinned MMPose head registries under {package}')

    changed = []
    if patch_file(heads, (
        (
            'from .hybrid_heads import DEKRHead, RTMOHead, VisPredictHead',
            'from .hybrid_heads import DEKRHead, VisPredictHead',
        ),
        (
            'from .transformer_heads import EDPoseHead',
            '# EDPose is disabled: it requires compiled mmcv.ops.',
        ),
        (", 'EDPoseHead'", ''),
        (", 'RTMOHead'", ''),
    )):
        changed.append(str(heads))
    if patch_file(hybrid, (
        (
            'from .rtmo_head import RTMOHead',
            '# RTMO is disabled: it requires MMDetection.',
        ),
        (", 'RTMOHead'", ''),
    )):
        changed.append(str(hybrid))

    # This import is the acceptance check. It imports the complete remaining
    # registry, not merely the light-weight package __init__.
    from mmpose.apis import inference_topdown, init_model  # noqa: F401
    from mmpose.models import MobileNetV2  # noqa: F401

    report = {
        'mmpose_version': mmpose.__version__,
        'package': str(package),
        'status': 'pass',
        'changed_files': changed,
        'disabled_unused_registries': [
            {
                'name': 'EDPoseHead',
                'reason': 'requires compiled mmcv.ops',
            },
            {
                'name': 'RTMOHead',
                'reason': 'requires MMDetection',
            },
        ],
        'selected_pose_model_affected': False,
        'selected_pose_model': 'MobileNetV2 + HeatmapHead',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

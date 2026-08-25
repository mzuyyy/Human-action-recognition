#!/usr/bin/env python
"""Task 4 — Inspect ntu60_2d.pkl before training.

Usage:
    python scripts/inspect_ntu60.py --ann-file data/skeleton/ntu60_2d.pkl \
        [--json-out artifacts/dataset_stats.json]

Exit code 0 = all acceptance checks pass, 1 = at least one failed.
"""
import argparse
import pickle
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ann-file', default='data/skeleton/ntu60_2d.pkl')
    # ponytail: NaN scan is strided over this many samples, not the full set;
    # raise to len(annotations) for an exhaustive scan if ever needed.
    ap.add_argument('--nan-scan-samples', type=int, default=4000)
    ap.add_argument('--json-out', default=None,
                    help='optional path to dump summary stats as JSON')
    args = ap.parse_args()

    with open(args.ann_file, 'rb') as f:
        data = pickle.load(f)

    print('type(data):', type(data))
    print('data.keys():', list(data.keys()))
    assert {'split', 'annotations'} <= set(data.keys()), \
        'expected top-level keys: split, annotations'

    splits, anns = data['split'], data['annotations']
    assert anns, 'annotations is empty'
    print('\n== SPLIT OVERVIEW ==')
    print('number of annotations:', len(anns))
    for name, ids in splits.items():
        print(f'  {name}: {len(ids)} samples')

    s = anns[0]
    kp, ks = s['keypoint'], s['keypoint_score']
    print('\n== FIRST SAMPLE ==')
    print('keys:', list(s.keys()))
    print('frame_dir:', s['frame_dir'])
    print('total_frames:', s['total_frames'])
    print('label:', s['label'])
    print('img_shape:', s['img_shape'])
    print('keypoint.shape:', kp.shape, '(M x T x V x C)')
    print('keypoint_score.shape:', ks.shape, '(M x T x V)')

    frames = np.array([a['total_frames'] for a in anns])
    labels = np.array([a['label'] for a in anns])
    uniq = np.unique(labels)
    print('\n== DATASET-WIDE STATS ==')
    print(f'frames  min/max/mean: {frames.min()} / {frames.max()} '
          f'/ {frames.mean():.2f}')
    print(f'label   min/max: {labels.min()} / {labels.max()}')
    print('unique labels:', uniq.size)

    # shape sanity on every sample (cheap), NaN on a strided subsample
    bad_shape = []
    bad_metadata = []
    for i, a in enumerate(anns):
        k = a['keypoint']
        score = a['keypoint_score']
        if (k.ndim != 4 or score.ndim != 3 or k.shape[2:] != (17, 2)
                or k.shape[:3] != score.shape
                or k.shape[1] != a['total_frames']):
            bad_shape.append(i)
        if (a['total_frames'] < 1 or len(a['img_shape']) != 2
                or any(x < 1 for x in a['img_shape'])):
            bad_metadata.append(i)
    annotation_ids = {a['frame_dir'] for a in anns}
    missing_split_ids = {
        name: len(set(ids) - annotation_ids) for name, ids in splits.items()
    }
    step = max(1, len(anns) // max(1, args.nan_scan_samples))
    scanned = anns[::step]
    n_nonfinite = sum(int(not np.isfinite(a['keypoint']).all()
                          or not np.isfinite(a['keypoint_score']).all())
                      for a in scanned)

    checks = {
        'dataset loads without error': True,
        'action labels are exactly 0..59': np.array_equal(uniq, np.arange(60)),
        f'keypoint tensors valid (M,T,V=17,C=2) — {len(bad_shape)} malformed':
            not bad_shape,
        f'frame count and image shape valid — {len(bad_metadata)} malformed':
            not bad_metadata,
        'frame_dir identifiers are unique': len(annotation_ids) == len(anns),
        'every split ID has an annotation': not any(missing_split_ids.values()),
        'split xsub_train exists and is nonempty':
            'xsub_train' in splits and len(splits['xsub_train']) > 0,
        'split xsub_val exists and is nonempty':
            'xsub_val' in splits and len(splits['xsub_val']) > 0,
        f'no NaN/Inf in {len(scanned)} scanned samples ({n_nonfinite} bad)':
            n_nonfinite == 0,
    }
    print('\n== ACCEPTANCE CHECKS ==')
    ok = True
    for name, passed in checks.items():
        print(('PASS' if passed else 'FAIL'), '-', name)
        ok &= passed

    if args.json_out:
        import json
        from pathlib import Path
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump({
                'ann_file': args.ann_file,
                'num_annotations': len(anns),
                'num_classes': int(uniq.size),
                'splits': {k: len(v) for k, v in splits.items()},
                'frames_min': int(frames.min()),
                'frames_max': int(frames.max()),
                'frames_mean': float(frames.mean()),
                'label_min': int(labels.min()),
                'label_max': int(labels.max()),
                'example_keypoint_shape': list(kp.shape),
                'example_keypoint_score_shape': list(ks.shape),
            }, f, indent=2)
        print('\nstats written to', args.json_out)

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

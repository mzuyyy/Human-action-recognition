#!/usr/bin/env python
"""Task 5 — Visualize COCO-17 skeletons from ntu60_2d.pkl (no RGB video needed).

Draws joints + bone connections on a white canvas so we can verify coordinates,
joint indexing and temporal motion sanity BEFORE training.

Usage:
    python scripts/visualize_skeleton.py --ann-file data/skeleton/ntu60_2d.pkl \
        --out-dir artifacts/skeleton_samples [--num-samples 3] [--frames-per-sample 5]
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# COCO-17 skeleton pairs (mmaction2/mmpose COCO layout)
COCO_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]


def draw_person(ax, kp: np.ndarray, score: np.ndarray, thr: float) -> None:
    """kp: V x C, score: V — draw one person's skeleton."""
    ok = score > thr
    ax.scatter(kp[ok, 0], kp[ok, 1], s=18, c='#d62728', zorder=3)
    for a, b in COCO_PAIRS:
        if ok[a] and ok[b]:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]],
                    '-', c='#1f77b4', lw=2, zorder=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ann-file', default='data/skeleton/ntu60_2d.pkl')
    ap.add_argument('--split', default='xsub_train')
    ap.add_argument('--out-dir', default='artifacts/skeleton_samples')
    ap.add_argument('--num-samples', type=int, default=3)
    ap.add_argument('--frames-per-sample', type=int, default=5)
    ap.add_argument('--score-thr', type=float, default=0.3,
                    help='skip keypoints below this confidence')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ann_file, 'rb') as f:
        data = pickle.load(f)
    anns = data['annotations']

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(anns), size=args.num_samples, replace=False)

    k = args.frames_per_sample
    for idx in picks:
        s = anns[idx]
        kp = s['keypoint']      # M x T x V x C
        ks = s['keypoint_score']  # M x T x V
        h, w = s['img_shape'][0], s['img_shape'][1]
        frame_ids = np.linspace(0, s['total_frames'] - 1, k).round().astype(int)

        fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.4))
        for c, fi in enumerate(frame_ids):
            ax = axes[c]
            ax.set_facecolor('white')
            for m in range(kp.shape[0]):  # every detected person
                draw_person(ax, kp[m, fi], ks[m, fi], args.score_thr)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)          # image coords: y grows downward
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'frame {fi}', fontsize=9)
        fig.suptitle(
            f"{s['frame_dir']} | label={s['label']} | "
            f"total_frames={s['total_frames']} | img_shape={tuple(s['img_shape'])}",
            fontsize=10)
        fig.tight_layout()
        sample_path = out_dir / f"{str(s['frame_dir']).replace('/', '_')}.png"
        fig.savefig(sample_path, dpi=110, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print('saved', sample_path)

    # combined contact sheet from the per-sample figures
    fig, axes_all = plt.subplots(args.num_samples, k,
                                 figsize=(3 * k, 3 * args.num_samples))
    axes_all = np.atleast_2d(axes_all)
    for row, idx in enumerate(picks):
        s = anns[idx]
        kp, ks = s['keypoint'], s['keypoint_score']
        h, w = s['img_shape'][0], s['img_shape'][1]
        frame_ids = np.linspace(0, s['total_frames'] - 1, k).round().astype(int)
        for c, fi in enumerate(frame_ids):
            ax = axes_all[row][c]
            for m in range(kp.shape[0]):
                draw_person(ax, kp[m, fi], ks[m, fi], args.score_thr)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect('equal')
            ax.axis('off')
    fig.suptitle(f'{args.num_samples} samples x {k} frames '
                 f'(split={args.split}, seed={args.seed})', fontsize=11)
    sheet_path = out_dir / 'skeleton_samples_overview.png'
    fig.savefig(sheet_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', sheet_path)


if __name__ == '__main__':
    main()

# Training-compatible skeleton preprocessing contract

This document freezes the exact validation-time input contract used by the
successful ST-GCN Joint and Joint Motion experiments. The online video path
must satisfy this contract before either action model is called.

## Resolved validation pipelines

Joint (`configs/stgcn_ntu60_xsub_baseline.py`):

```text
PreNormalize2D
GenSkeFeat(dataset='coco', feats=['j'])
UniformSampleFrames(clip_len=100, num_clips=1, test_mode=True, seed=255)
PoseDecode
FormatGCNInput(num_person=2, mode='zero')
PackActionInputs
```

Joint Motion (`configs/stgcn_ntu60_xsub_joint_motion_80e.py`) is identical
except for `GenSkeFeat(dataset='coco', feats=['jm'])`.

There is no `Resize`, `PoseCompact`, crop, coordinate interpolation, or
`PoseNormalize` transform in either validation pipeline.

## Spatial contract

- Graph/layout: COCO.
- Joint count: 17.
- Raw coordinate order: `(x, y)` in source-image pixels.
- Raw arrays before transforms:
  - `keypoint`: `(M, T_raw, 17, 2)`
  - `keypoint_score`: `(M, T_raw, 17)`
- `M` is the available person-slot count. The first video implementation uses
  one dominant person (`M=1`).
- `PreNormalize2D` uses the sample's actual `img_shape=(height, width)`:

```text
x_norm = (x - width / 2) / (width / 2)
y_norm = (y - height / 2) / (height / 2)
```

- `GenSkeFeat` appends keypoint confidence as channel 3 for COCO 2D input.
  Joint features therefore have shape `(M, T_raw, 17, 3)` and channels
  `(x_norm, y_norm, confidence)`.
- `FormatGCNInput(num_person=2, mode='zero')` keeps the first two people,
  truncates any additional people, or appends all-zero person slots. It does
  not duplicate a single person because the configured mode is `zero`.

The formulas and person padding above are taken from the pinned MMAction2
v1.2.0 implementations of
[`PreNormalize2D`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/datasets/transforms/pose_transforms.py#L648-L681),
[`GenSkeFeat`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/datasets/transforms/pose_transforms.py#L855-L904), and
[`FormatGCNInput`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/datasets/transforms/formatting.py#L356-L410).

## Temporal contract

- Validation uses one deterministic clip of 100 sampled frames.
- `UniformSampleFrames` uses test seed 255 and the exact v1.2.0 branch logic:
  - shorter than 100: produce 100 consecutive indices and wrap with modulo;
  - 100 through 199 frames: deterministically drop `T_raw - 100` positions;
  - at least 200 frames: split the sequence into 100 bins and choose one
    deterministic position from each bin.
- `PoseDecode` then indexes both feature coordinates and confidence with those
  100 frame indices. It performs selection only; it does not interpolate.
- Long-video inference uses overlapping raw windows. Each window is passed
  through this same official sampler. The default raw window is 100 frames and
  the default stride is 50 frames, both derived from the inspected `clip_len`.

The sampling and decoding behavior comes from the pinned
[`UniformSampleFrames`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/datasets/transforms/pose_transforms.py#L913-L1059)
and [`PoseDecode`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/datasets/transforms/pose_transforms.py#L1125-L1177) source.

## Joint Motion contract

`GenSkeFeat(feats=['jm'])` invokes MMAction2's official `ToMotion` **before**
temporal sampling:

```text
motion[:, 0:T_raw-1, :, 0:2] = diff(joint_xy, axis=time)
motion[:, 0:T_raw-1, :, 2] =
    (joint_score[:, 0:T_raw-1] + joint_score[:, 1:T_raw]) / 2
motion[:, T_raw-1] = 0
```

Thus the motion confidence channel is an adjacent-frame average, not a
confidence difference. See the pinned
[`ToMotion`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/datasets/transforms/pose_transforms.py#L757-L805)
implementation.

## Final tensor shape

After `FormatGCNInput`, one sample is:

```text
(num_clips, num_person, clip_len, num_joints, channels)
= (1, 2, 100, 17, 3)
```

`PackActionInputs` converts it to a tensor. The data preprocessor stacks the
batch, so the tensor entering `RecognizerGCN` is:

```text
(batch, num_clips, num_person, clip_len, num_joints, channels)
= (1, 1, 2, 100, 17, 3)
```

This agrees with the v1.2.0
[`RecognizerGCN`](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/models/recognizers/recognizer_gcn.py#L13-L35)
input contract. Both streams have the same shape.

## Online compatibility rule

The inference implementation passes raw MMPose COCO-17 coordinates,
confidence, `total_frames`, and the original image shape into the official
MMAction2 transforms listed above. Shape assertions require the final
`(1, 2, 100, 17, 3)` sample before adding the batch dimension. A layout or
shape mismatch is a hard error; inference must not continue with reordered or
non-COCO keypoints.

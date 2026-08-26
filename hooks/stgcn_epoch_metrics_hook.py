"""Per-outer-epoch metrics for the NTU60 ST-GCN baseline."""

import json
import time
from pathlib import Path

import torch
from mmengine.hooks import Hook

from mmaction.registry import HOOKS


@HOOKS.register_module()
class STGCNEpochMetricsHook(Hook):
    """Record one complete JSON row after each train/validation outer epoch."""

    def __init__(self, repeat_times: int = 5) -> None:
        self.repeat_times = repeat_times
        self._epoch_started = 0.0
        self._loss_sum = 0.0
        self._loss_count = 0
        self._last_lr = None

    def before_train_epoch(self, runner) -> None:
        self._epoch_started = time.perf_counter()
        self._loss_sum = 0.0
        self._loss_count = 0
        self._last_lr = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs=None) -> None:
        if isinstance(outputs, dict) and outputs.get('loss') is not None:
            loss = outputs['loss']
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().item()
            self._loss_sum += float(loss)
            self._loss_count += 1

        lr_groups = runner.optim_wrapper.get_lr()
        if lr_groups:
            self._last_lr = float(next(iter(lr_groups.values()))[0])

    def after_val_epoch(self, runner, metrics=None) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        else:
            gpu_memory_mb = 0.0

        epoch = runner.epoch + 1
        metrics = metrics or {}
        record = {
            'outer_epoch': epoch,
            'effective_epoch': epoch * self.repeat_times,
            'learning_rate': self._last_lr,
            'train_loss': (
                self._loss_sum / self._loss_count
                if self._loss_count else None),
            'val_acc_top1': float(metrics['acc/top1']),
            'val_acc_top5': float(metrics['acc/top5']),
            'gpu_memory_mb': gpu_memory_mb,
            'epoch_wall_time_sec': time.perf_counter() - self._epoch_started,
        }

        if runner.rank == 0:
            work_dir = Path(runner.work_dir)
            with (work_dir / 'epoch_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')

            regular_checkpoint = work_dir / f'epoch_{epoch}.pth'
            latest_checkpoint = work_dir / 'latest.pth'
            if regular_checkpoint.is_file():
                temporary_link = work_dir / '.latest.pth.tmp'
                temporary_link.unlink(missing_ok=True)
                temporary_link.symlink_to(regular_checkpoint.name)
                temporary_link.replace(latest_checkpoint)

        runner.logger.info(
            'Outer epoch %d / effective epoch %d: loss=%.6f, '
            'acc/top1=%.4f, acc/top5=%.4f, lr=%s, GPU=%.0f MiB, wall=%.1fs',
            record['outer_epoch'], record['effective_epoch'],
            record['train_loss'], record['val_acc_top1'],
            record['val_acc_top5'], record['learning_rate'],
            record['gpu_memory_mb'], record['epoch_wall_time_sec'])

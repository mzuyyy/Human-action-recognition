"""Per-outer-epoch metrics for the NTU60 ST-GCN baseline."""

import json
import math
import time
from pathlib import Path

import torch
from mmengine.hooks import Hook

from mmaction.registry import HOOKS


@HOOKS.register_module()
class STGCNEpochMetricsHook(Hook):
    """Record one complete JSON row after each train/validation outer epoch."""

    def __init__(self, repeat_times: int = 5,
                 resume_cosine_t_max: int | None = None) -> None:
        self.repeat_times = repeat_times
        self.resume_cosine_t_max = resume_cosine_t_max
        self._epoch_started = 0.0
        self._loss_sum = 0.0
        self._loss_count = 0
        self._last_lr = None

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Extend a completed cosine schedule when resuming for more epochs."""
        if self.resume_cosine_t_max is None:
            return

        meta = checkpoint.get('meta', {})
        checkpoint_epoch = int(meta.get('epoch', 0))
        checkpoint_iter = int(meta.get('iter', 0))
        if checkpoint_epoch < 1 or checkpoint_iter < 1:
            raise RuntimeError('resume checkpoint has no valid epoch/iter metadata')
        if checkpoint_epoch > self.resume_cosine_t_max:
            raise RuntimeError(
                'resume checkpoint is beyond the configured final epoch')

        scheduler_states = checkpoint.get('param_schedulers', [])
        cosine_states = [
            state for state in scheduler_states if 'T_max' in state
        ] if isinstance(scheduler_states, list) else []
        if len(cosine_states) != 1:
            raise RuntimeError(
                'expected exactly one cosine scheduler in resume checkpoint')

        epoch_length = checkpoint_iter / checkpoint_epoch
        target_t_max = int(round(self.resume_cosine_t_max * epoch_length))
        state = cosine_states[0]
        current_step = int(state['last_step'])
        if current_step > target_t_max:
            raise RuntimeError('resume scheduler step exceeds its new T_max')
        base_values = [float(value) for value in state['base_values']]

        resumed_lrs = []
        for base_value in base_values:
            eta_min = state.get('eta_min')
            if eta_min is None:
                eta_min = base_value * float(state['eta_min_ratio'])
            lr = eta_min + 0.5 * (base_value - eta_min) * (
                1 + math.cos(math.pi * current_step / target_t_max))
            resumed_lrs.append(lr)

        optimizer_state = checkpoint.get('optimizer', {})
        param_groups = optimizer_state.get('param_groups', [])
        if len(param_groups) != len(resumed_lrs):
            raise RuntimeError('optimizer and cosine scheduler groups differ')
        for group, lr in zip(param_groups, resumed_lrs):
            group['lr'] = lr

        old_t_max = state['T_max']
        state['T_max'] = target_t_max
        if state.get('end') == old_t_max:
            state['end'] = target_t_max
        state['_last_value'] = resumed_lrs
        runner.logger.info(
            'Extended resumed cosine schedule T_max %s -> %s steps; LR=%s',
            old_t_max, target_t_max, resumed_lrs)

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

        # EpochBasedTrainLoop increments its internal epoch before validation,
        # so runner.epoch is already one-based inside after_val_epoch.
        epoch = runner.epoch
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

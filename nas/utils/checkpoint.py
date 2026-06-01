"""
断点保存与恢复模块

实现健壮的checkpoint保存与恢复机制，确保中断后可无损续训
"""

import os
import glob
import torch
import logging

_logger = logging.getLogger('nas_checkpoint')


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, lr_scheduler,
                    model_ema=None, loss_scaler=None, best_metric=None,
                    is_best=False, tag='checkpoint'):
    """
    保存训练断点

    Args:
        checkpoint_dir: 断点保存目录
        epoch: 当前epoch
        model: 模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        model_ema: EMA模型
        loss_scaler: 混合精度loss scaler
        best_metric: 当前最优指标
        is_best: 是否为最优模型
        tag: 断点文件名标识
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'best_metric': best_metric,
    }

    if model_ema is not None:
        checkpoint['state_dict_ema'] = model_ema.state_dict()

    if loss_scaler is not None:
        checkpoint['loss_scaler'] = loss_scaler.state_dict()

    # 保存最新断点
    latest_path = os.path.join(checkpoint_dir, f'{tag}_latest.pth')
    torch.save(checkpoint, latest_path)

    # 保存当前epoch断点
    epoch_path = os.path.join(checkpoint_dir, f'{tag}_epoch_{epoch:04d}.pth')
    torch.save(checkpoint, epoch_path)

    # 保存最优断点
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'{tag}_best.pth')
        torch.save(checkpoint, best_path)
        _logger.info(f"Saved best checkpoint to {best_path}")

    _logger.info(f"Saved checkpoint to {epoch_path}")
    return epoch_path


def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None,
                    model_ema=None, loss_scaler=None, strict=True):
    """
    加载训练断点

    Args:
        checkpoint_path: 断点文件路径
        model: 模型
        optimizer: 优化器（可选）
        lr_scheduler: 学习率调度器（可选）
        model_ema: EMA模型（可选）
        loss_scaler: 混合精度loss scaler（可选）
        strict: 是否严格匹配state_dict

    Returns:
        dict: 包含epoch, best_metric等信息的字典
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    _logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型权重
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        _logger.info("Loaded optimizer state")

    # 加载学习率调度器状态
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        _logger.info("Loaded lr_scheduler state")

    # 加载EMA模型状态
    if model_ema is not None and 'state_dict_ema' in checkpoint:
        model_ema.load_state_dict(checkpoint['state_dict_ema'])
        _logger.info("Loaded EMA model state")
        if hasattr(model_ema, 'sync_stale_buffers'):
            synced_buffers = model_ema.sync_stale_buffers(model)
            if synced_buffers:
                _logger.warning(
                    "Synchronized %d stale EMA buffers from model state; "
                    "the checkpoint likely came from an older EMA implementation.",
                    synced_buffers)

    # 加载loss scaler状态
    if loss_scaler is not None and 'loss_scaler' in checkpoint:
        loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        _logger.info("Loaded loss scaler state")

    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', None)

    _logger.info(f"Loaded checkpoint from epoch {epoch}, best_metric={best_metric}")
    return {
        'epoch': epoch,
        'best_metric': best_metric,
    }


def resume_training(checkpoint_dir, model, optimizer, lr_scheduler=None,
                    model_ema=None, loss_scaler=None, tag='checkpoint'):
    """
    自动恢复训练：查找最新的断点文件并加载

    Args:
        checkpoint_dir: 断点保存目录
        model: 模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        model_ema: EMA模型
        loss_scaler: 混合精度loss scaler
        tag: 断点文件名标识

    Returns:
        tuple: (start_epoch, best_metric) 或 (0, None) 如果没有找到断点
    """
    latest_path = os.path.join(checkpoint_dir, f'{tag}_latest.pth')

    if os.path.isfile(latest_path):
        result = load_checkpoint(
            latest_path, model, optimizer, lr_scheduler,
            model_ema, loss_scaler, strict=True
        )
        return result['epoch'] + 1, result['best_metric']

    # 如果没有latest，查找最新的epoch断点
    pattern = os.path.join(checkpoint_dir, f'{tag}_epoch_*.pth')
    epoch_files = sorted(glob.glob(pattern))
    if epoch_files:
        latest_epoch_file = epoch_files[-1]
        result = load_checkpoint(
            latest_epoch_file, model, optimizer, lr_scheduler,
            model_ema, loss_scaler, strict=True
        )
        return result['epoch'] + 1, result['best_metric']

    _logger.info("No checkpoint found, starting from scratch")
    return 0, None


def cleanup_old_checkpoints(checkpoint_dir, keep_num=3, tag='checkpoint'):
    """
    清理旧的断点文件，只保留最新的keep_num个

    Args:
        checkpoint_dir: 断点保存目录
        keep_num: 保留的断点数量
        tag: 断点文件名标识
    """
    pattern = os.path.join(checkpoint_dir, f'{tag}_epoch_*.pth')
    epoch_files = sorted(glob.glob(pattern))

    if len(epoch_files) > keep_num:
        for old_file in epoch_files[:-keep_num]:
            try:
                os.remove(old_file)
                _logger.info(f"Removed old checkpoint: {old_file}")
            except OSError:
                pass

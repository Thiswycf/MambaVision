"""
CUDA_VISIBLE_DEVICES=3 python nas/train_supernet.py -c nas/configs/supernet_tiny.yaml

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 nas/train_supernet.py -c nas/configs/supernet_tiny.yaml

CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 nas/train_supernet.py -c nas/configs/supernet_tiny.yaml

超网训练脚本

基于Sandwich Rule的SPOS（Single Path One-Shot）训练流程
每次迭代同时训练：
    - 1个最大子网（全Attention）
    - 1个中等子网（全Mamba）
    - 1个最小子网（全CNN）
    - K-3个随机采样子网

所有子网共享超网参数，统一反向传播更新。
"""

import argparse
import json
import os
import sys
import time
import yaml
import logging
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

# 添加项目路径
sys.path.insert(0, '/public/zhanghaojie/MambaVision')
sys.path.insert(0, '/public/zhanghaojie/MambaVision/mambavision')

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup
from timm.models import safe_model_name
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2
from timm.utils import ApexScaler, NativeScaler

from nas.supernet import MambaVisionSuperNet
from nas.search_space import get_sandwich_subnet_genotypes, TOTAL_LAYERS
from nas.utils import (
    SuperNetTensorboardLogger,
    save_checkpoint, load_checkpoint, resume_training, cleanup_old_checkpoints,
    ModelEMA, AverageMeter, accuracy
)

_logger = logging.getLogger('train_supernet')
# torch.autograd.set_detect_anomaly(True)


def setup_default_logging():
    """设置默认日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='MambaVision SuperNet Training (SPOS + Sandwich Rule)')

    # 配置文件
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    # 数据集参数
    parser.add_argument('--data_dir', metavar='DIR', default='/home/lqz25zhj/data/ImageNet1k',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')

    # 模型参数
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes')
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                        help='Image patch size (default: 224)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                        metavar='N', help='Input image center crop percent')

    # 超网特定参数
    parser.add_argument('--supernet-k', type=int, default=5, metavar='K',
                        help='Number of subnets to train per iteration (Sandwich Rule, default: 5)')
    parser.add_argument('--supernet-dim', type=int, default=80,
                        help='SuperNet base dim (default: 80 for Tiny)')
    parser.add_argument('--supernet-in-dim', type=int, default=32,
                        help='SuperNet input dim (default: 32 for Tiny)')
    parser.add_argument('--supernet-depths', type=int, nargs='+', default=[1, 3, 8, 4],
                        help='SuperNet depths per stage')
    parser.add_argument('--supernet-num-heads', type=int, nargs='+', default=[2, 4, 8, 16],
                        help='SuperNet num heads per stage')
    parser.add_argument('--supernet-window-size', type=int, nargs='+', default=[8, 8, 14, 7],
                        help='SuperNet window size per stage')

    # 训练超参数
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='Validation batch size override')
    parser.add_argument('--limit-train-batches', type=int, default=0,
                        help='Limit train batches per epoch for budgeted pilot runs; 0 means full epoch')
    parser.add_argument('--limit-val-batches', type=int, default=0,
                        help='Limit validation batches for budgeted pilot runs; 0 means full validation')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon')
    parser.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',
                        help='Clip gradient norm (default: 5.0)')

    # 学习率调度参数
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                        help='lower lr bound')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR (default: 10)')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate')

    # 正则化参数
    parser.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1 for supernet)')
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # 数据增强参数
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                        help='AutoAugment policy')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count')

    # Mixup/Cutmix参数
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params')

    # EMA参数
    parser.add_argument('--model-ema', action='store_true', default=True,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average')

    # 分布式训练参数
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='Enable synchronized BatchNorm')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats')

    # 混合精度训练
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use Native AMP for mixed precision training')

    # 日志和输出参数
    parser.add_argument('--log-per-epoch', type=int, default=10, metavar='N',
                        help='how many times to log per epoch (default: 10)')
    parser.add_argument('--output', default='/public/zhanghaojie/MambaVision/nas/weights', type=str,
                        help='path to output folder for weights')
    parser.add_argument('--checkpoint-dir', default='/public/zhanghaojie/MambaVision/nas/checkpoints', type=str,
                        help='path to checkpoint directory')
    parser.add_argument('--log-dir', default='/public/zhanghaojie/MambaVision/nas/logs', type=str,
                        help='path to tensorboard log directory')
    parser.add_argument('--tag', default='supernet', type=str,
                        help='experiment tag')
    parser.add_argument('--metrics-json', default='', type=str,
                        help='Optional path to write parseable epoch metrics as JSON')
    parser.add_argument('--checkpoint-hist', type=int, default=3, metavar='N',
                        help='number of checkpoints to keep')

    # 断点续训
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint')
    parser.add_argument('--resume-weights-only', action='store_true', default=False,
                        help='Load only model weights from --resume and start a fresh optimizer/scheduler')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                        help='Auto resume from latest checkpoint in checkpoint-dir')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')
    parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                        help='how many training processes to use')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--mesa',  type=float, default=0.0,
                        help='use memory efficient sharpness optimization, enabled if >0.0')
    parser.add_argument('--mesa-start-ratio',  type=float, default=0.25,
                        help='when to start MESA, ratio to total training time, def 0.25')

    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # Apply YAML as defaults, then parse the full command line so explicit CLI
    # arguments override config values for budgeted pilot runs.
    args = parser.parse_args()
    return args

kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()

def kdloss(y, teacher_scores):
    T = 3
    p = torch.nn.functional.log_softmax(y/T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores/T, dim=1)
    l_kl = 50.0*kl_loss(p, q)
    return l_kl


def create_dataloaders(args):
    """创建训练和验证数据加载器"""
    data_config = resolve_data_config(vars(args), verbose=args.local_rank == 0)

    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size)

    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        batch_size=args.batch_size)

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if not args.no_prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=not args.no_prefetcher,
        no_aug=False,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    return loader_train, loader_eval, mixup_fn, data_config


def train_one_epoch(
    epoch, model, loader, optimizer, loss_fn, args,
    lr_scheduler=None, model_ema=None, mixup_fn=None,
    amp_autocast=None, loss_scaler=None, log_writer=None
):
    """
    基于Sandwich Rule的单epoch训练

    每次迭代训练K个子网，所有子网共享参数，梯度按K归一化后统一更新
    """
    if amp_autocast is None:
        from contextlib import suppress
        amp_autocast = suppress

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    # 为每个子网维护loss计量器
    subnet_loss_meters = {}

    model.train()

    end = time.time()
    max_batches = int(getattr(args, 'limit_train_batches', 0) or 0)
    effective_len = min(len(loader), max_batches) if max_batches else len(loader)
    last_idx = effective_len - 1
    num_updates = epoch * effective_len

    for batch_idx, (input, target) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        data_time_m.update(time.time() - end)

        if not args.no_prefetcher:
            # prefetcher已经将数据放到GPU
            pass
        else:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # 获取Sandwich Rule子网
        subnets = get_sandwich_subnet_genotypes(k=args.supernet_k)

        subnet_losses = {}
        total_loss_value = 0.0
        optimizer.zero_grad()
        scaler = getattr(loss_scaler, '_scaler', None) if loss_scaler is not None else None

        # 依次前向传播每个子网
        for tag_prefix, genotype in subnets:
            with amp_autocast():
                output = model(input, genotype)
                loss = loss_fn(output, target)

                if args.mesa > 0.0 and model_ema is not None:
                    if epoch / args.epochs > args.mesa_start_ratio:
                        with torch.no_grad():
                            ema_output = model_ema.module(input).data.detach()
                        kd = kdloss(output, ema_output)
                        loss += args.mesa * kd

            raw_loss = loss.detach()
            scaled_loss = loss / args.supernet_k
            subnet_losses[tag_prefix] = raw_loss
            total_loss_value += raw_loss.item()

            # 更新计量器
            if tag_prefix not in subnet_loss_meters:
                subnet_loss_meters[tag_prefix] = AverageMeter()
            subnet_loss_meters[tag_prefix].update(raw_loss.item(), input.size(0))

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if scaler is not None:
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        # 更新EMA
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        # 学习率调度
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates)

        # 日志记录（基于log_per_epoch）
        log_per_epoch = max(1, args.log_per_epoch)
        steps_per_log = len(loader) / log_per_epoch
        current_log_idx = int(batch_idx / steps_per_log)
        next_log_idx = int((batch_idx + 1) / steps_per_log)
        log_triggered = False
        if batch_idx == last_idx:
            log_triggered = True
        else:
            # 检查是否跨越了log边界
            if current_log_idx < next_log_idx or (batch_idx == 0 and log_per_epoch > 0):
                log_triggered = True

        if log_triggered:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                progress = 100. * batch_idx / max(1, last_idx)
                log_parts = [
                    f'Train: {epoch} [{batch_idx:>4d}/{effective_len} ({progress:>3.0f}%)]'
                ]
                for tag_prefix, loss_val in subnet_losses.items():
                    log_parts.append(f'{tag_prefix}_loss: {loss_val.item():.4f}')
                log_parts.append(f'Total: {total_loss_value:.4f}')
                log_parts.append(f'LR: {lr:.3e}')
                _logger.info('  '.join(log_parts))

                # TensorBoard记录
                if log_writer is not None:
                    current_step = epoch * len(loader) + batch_idx
                    for tag_prefix, loss_val in subnet_losses.items():
                        log_writer.update(
                            tag_prefix=tag_prefix,
                            step=current_step,
                            train_loss=loss_val.item()
                        )
                    log_writer.update(tag_prefix='global', step=current_step, lr=lr)
                    log_writer.update(tag_prefix='epoch', step=current_step, epoch=epoch + batch_idx / len(loader))
                    log_writer.flush()

        end = time.time()

    # 返回各子网的平均loss
    result = OrderedDict()
    for tag_prefix, meter in subnet_loss_meters.items():
        result[f'{tag_prefix}_loss'] = meter.avg
    return result


@torch.no_grad()
def validate(model, loader, loss_fn, args, amp_autocast=None, log_suffix='', log_writer=None, epoch=0, genotype=None):
    """验证模型

    Args:
        genotype: 如果提供，则验证指定子网；否则验证中等子网
    """
    if amp_autocast is None:
        from contextlib import suppress
        amp_autocast = suppress

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    max_batches = int(getattr(args, 'limit_val_batches', 0) or 0)
    effective_len = min(len(loader), max_batches) if max_batches else len(loader)
    last_idx = effective_len - 1

    # 默认使用中等子网进行验证
    if genotype is None:
        genotype = 'M' * TOTAL_LAYERS

    for batch_idx, (input, target) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        last_batch = batch_idx == last_idx

        if not args.no_prefetcher:
            pass
        else:
            input = input.cuda()
            target = target.cuda()

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input, genotype)

        if isinstance(output, (tuple, list)):
            output = output[0]

        loss = loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = torch.tensor([loss.item()], device='cuda')
            acc1 = torch.tensor([acc1.item()], device='cuda')
            acc5 = torch.tensor([acc5.item()], device='cuda')
            torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(acc1, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(acc5, op=torch.distributed.ReduceOp.SUM)
            reduced_loss /= args.world_size
            acc1 /= args.world_size
            acc5 /= args.world_size
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))
        top1_m.update(acc1.item(), input.size(0))
        top5_m.update(acc5.item(), input.size(0))
        batch_time_m.update(time.time() - end)

        # 日志记录（基于log_per_epoch）
        log_per_epoch = max(1, args.log_per_epoch)
        steps_per_log = len(loader) / log_per_epoch
        current_log_idx = int(batch_idx / steps_per_log)
        next_log_idx = int((batch_idx + 1) / steps_per_log)
        log_triggered = False
        if last_batch:
            log_triggered = True
        else:
            # 检查是否跨越了log边界
            if current_log_idx < next_log_idx or (batch_idx == 0 and log_per_epoch > 0):
                log_triggered = True

        if args.local_rank == 0 and log_triggered:
            display_total = max(0, effective_len - 1)
            log_name = 'Test' + log_suffix
            _logger.info(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                    log_name, batch_idx, display_total, batch_time=batch_time_m,
                    loss=losses_m, top1=top1_m, top5=top5_m))

        end = time.time()

    metrics = OrderedDict([
        ('loss', losses_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg)
    ])

    # TensorBoard记录
    if log_writer is not None and args.local_rank == 0:
        suffix = log_suffix.strip().replace(' ', '_').lower() or 'default'
        log_writer.update(tag_prefix=suffix, step=epoch, test_loss=metrics['loss'])
        log_writer.update(tag_prefix=suffix, step=epoch, test_acc1=metrics['top1'])
        log_writer.update(tag_prefix=suffix, step=epoch, test_acc5=metrics['top5'])
        log_writer.flush()

    return metrics


@torch.no_grad()
def validate_multiple_subnets(model, loader, loss_fn, args, amp_autocast=None, log_writer=None, epoch=0):
    """验证多个子网（max, mid, min），分别记录指标"""
    from nas.search_space import get_max_subnet_genotype, get_mid_subnet_genotype, get_min_subnet_genotype, sample_subnet_genotype

    assert args.supernet_k >= 3, "Sandwich Rule要求K >= 3"
    max_g = get_max_subnet_genotype()
    mid_g = get_mid_subnet_genotype()
    min_g = get_min_subnet_genotype()
    random_g = sample_subnet_genotype(num=1, exclude=[max_g, mid_g, min_g])[0]

    subnets = [
        ('max', max_g),
        ('mid', mid_g),
        ('min', min_g),
        ('random_0', random_g),
    ]

    all_metrics = {}
    for tag_prefix, genotype in subnets:
        _logger.info(f'Validating subnet: {tag_prefix}')
        metrics = validate(
            model, loader, loss_fn, args,
            amp_autocast=amp_autocast,
            log_suffix=f' {tag_prefix}',
            log_writer=log_writer,
            epoch=epoch,
            genotype=genotype
        )
        all_metrics[tag_prefix] = metrics
        _logger.info(f'{tag_prefix} subnet - Acc@1: {metrics["top1"]:.4f}, Loss: {metrics["loss"]:.4f}')

    # 返回`所有指标均值`作为主评估指标
    return {k: sum(m[k] for m in all_metrics.values()) / len(all_metrics) for k in all_metrics['mid']}


def main():
    setup_default_logging()
    args = parse_args()

    # 分布式训练设置
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = f'cuda:{args.local_rank}'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(f'Training in distributed mode: rank={args.rank}, world_size={args.world_size}')
    else:
        _logger.info('Training with a single process')

    # 设置随机种子
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed(args.seed + args.rank)

    # 混合精度设置
    amp_autocast = None
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP')

    # 创建模型
    _logger.info(f'Creating SuperNet model: dim={args.supernet_dim}, depths={args.supernet_depths}')
    model = MambaVisionSuperNet(
        depths=args.supernet_depths,
        num_heads=args.supernet_num_heads,
        window_size=args.supernet_window_size,
        dim=args.supernet_dim,
        in_dim=args.supernet_in_dim,
        drop_path_rate=args.drop_path,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
    )

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # SyncBN
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 创建优化器
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # 创建学习率调度器
    # 只传递create_scheduler_v2需要的参数
    scheduler_kwargs = {
        'sched': args.sched,
        'num_epochs': args.epochs,
        'decay_epochs': getattr(args, 'decay_epochs', 90),
        'decay_milestones': getattr(args, 'decay_milestones', (90, 180, 270)),
        'cooldown_epochs': getattr(args, 'cooldown_epochs', 0),
        'patience_epochs': getattr(args, 'patience_epochs', 10),
        'decay_rate': args.decay_rate,
        'min_lr': args.min_lr,
        'warmup_lr': args.warmup_lr,
        'warmup_epochs': args.warmup_epochs,
        'warmup_prefix': getattr(args, 'warmup_prefix', False),
        'noise': getattr(args, 'noise', None),
        'noise_pct': getattr(args, 'noise_pct', 0.67),
        'noise_std': getattr(args, 'noise_std', 1.0),
        'noise_seed': getattr(args, 'noise_seed', 42),
        'cycle_mul': getattr(args, 'cycle_mul', 1.0),
        'cycle_decay': getattr(args, 'cycle_decay', 0.1),
        'cycle_limit': getattr(args, 'cycle_limit', 1),
        'k_decay': getattr(args, 'k_decay', 1.0),
        'plateau_mode': getattr(args, 'plateau_mode', 'max'),
        'step_on_epochs': getattr(args, 'step_on_epochs', True),
        'updates_per_epoch': getattr(args, 'updates_per_epoch', 0),
    }
    lr_scheduler, num_epochs = create_scheduler_v2(optimizer, **scheduler_kwargs)

    # 创建EMA模型
    model_ema = None
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
        _logger.info(f'Using Model EMA with decay={args.model_ema_decay}')

    # 断点续训
    start_epoch = 0
    best_metric = None
    if args.resume:
        resume_optimizer = None if args.resume_weights_only else optimizer
        resume_scheduler = None if args.resume_weights_only else lr_scheduler
        resume_loss_scaler = None if args.resume_weights_only else loss_scaler
        result = load_checkpoint(
            args.resume, model, resume_optimizer, resume_scheduler,
            model_ema, resume_loss_scaler, strict=True
        )
        if args.resume_weights_only:
            start_epoch = 0
            best_metric = None
        else:
            start_epoch = result['epoch'] + 1
            best_metric = result['best_metric']
    elif args.auto_resume:
        start_epoch, best_metric = resume_training(
            args.checkpoint_dir, model, optimizer, lr_scheduler,
            model_ema, loss_scaler, tag=args.tag
        )

    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # 分布式包装
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)

    # 创建数据加载器
    loader_train, loader_eval, mixup_fn, data_config = create_dataloaders(args)

    # 损失函数
    mixup_active = args.mixup > 0 or args.cutmix > 0.
    if mixup_active:
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # TensorBoard日志
    log_writer = None
    if args.rank == 0:
        log_dir = os.path.join(args.log_dir, f'{args.tag}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SuperNetTensorboardLogger(log_dir=log_dir)
        _logger.info(f'TensorBoard logs: {log_dir}')

    # 输出目录
    if args.rank == 0:
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 训练循环
    eval_metric = 'top1'
    best_epoch = None
    history = []

    _logger.info(f'Starting training from epoch {start_epoch}, total epochs {num_epochs}')

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            # 训练
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, model_ema=model_ema, mixup_fn=mixup_fn,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, log_writer=log_writer
            )

            # 同步BN统计量
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                # 简化处理：不实现distribute_bn
                pass

            # 验证（使用原始模型）- 验证max/mid/min三个子网
            eval_metrics = validate_multiple_subnets(
                model.module if args.distributed else model,
                loader_eval, validate_loss_fn, args,
                amp_autocast=amp_autocast, log_writer=log_writer, epoch=epoch
            )

            # EMA验证 - 同样验证max/mid/min三个子网
            if model_ema is not None:
                ema_eval_metrics = validate_multiple_subnets(
                    model_ema.module, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast, log_writer=log_writer, epoch=epoch
                )
                # 使用EMA指标作为最终评估
                eval_metrics = ema_eval_metrics

            # 学习率调度
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            # 保存断点
            if args.rank == 0:
                is_best = False
                if best_metric is None or eval_metrics[eval_metric] > best_metric:
                    best_metric = eval_metrics[eval_metric]
                    best_epoch = epoch
                    is_best = True

                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    epoch=epoch,
                    model=model.module if args.distributed else model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    model_ema=model_ema,
                    loss_scaler=loss_scaler,
                    best_metric=best_metric,
                    is_best=is_best,
                    tag=args.tag
                )

                # 保存最优EMA权重
                if is_best and model_ema is not None:
                    ema_weight_path = os.path.join(args.output, f'{args.tag}_best_ema.pth')
                    torch.save(model_ema.module.state_dict(), ema_weight_path)
                    _logger.info(f'Saved best EMA weights to {ema_weight_path}')

                cleanup_old_checkpoints(args.checkpoint_dir, keep_num=args.checkpoint_hist, tag=args.tag)

                _logger.info(
                    f'Epoch {epoch}: test_acc1={eval_metrics["top1"]:.4f}, '
                    f'test_loss={eval_metrics["loss"]:.4f}, best_acc1={best_metric:.4f} (epoch {best_epoch})'
                )

                history.append(OrderedDict([
                    ('epoch', epoch),
                    ('train', train_metrics),
                    ('eval', eval_metrics),
                    ('best_metric', best_metric),
                    ('best_epoch', best_epoch),
                ]))
                if args.metrics_json:
                    os.makedirs(os.path.dirname(args.metrics_json) or '.', exist_ok=True)
                    with open(args.metrics_json, 'w') as f:
                        json.dump(OrderedDict([
                            ('tag', args.tag),
                            ('config', args.config),
                            ('supernet_dim', args.supernet_dim),
                            ('supernet_in_dim', args.supernet_in_dim),
                            ('supernet_num_heads', args.supernet_num_heads),
                            ('limit_train_batches', getattr(args, 'limit_train_batches', 0)),
                            ('limit_val_batches', getattr(args, 'limit_val_batches', 0)),
                            ('history', history),
                        ]), f, indent=2)

                # 记录各子网训练loss
                if log_writer is not None:
                    for k, v in train_metrics.items():
                        tag_prefix = k.replace('_loss', '')
                        log_writer.update(tag_prefix=tag_prefix, step=epoch, train_loss_epoch=v)
                    log_writer.flush()

    except KeyboardInterrupt:
        _logger.info('Training interrupted by user')
    except Exception as e:
        _logger.error(f'Training error: {e}', exc_info=True)
        raise

    # 保存最终断点
    if args.rank == 0:
        final_path = os.path.join(args.checkpoint_dir, f'{args.tag}_final.pth')
        torch.save({
            'epoch': num_epochs - 1,
            'state_dict': (model.module if args.distributed else model).state_dict(),
            'state_dict_ema': model_ema.module.state_dict() if model_ema is not None else None,
            'best_metric': best_metric,
            'best_epoch': best_epoch,
        }, final_path)
        _logger.info(f'Saved final checkpoint to {final_path}')

    if best_metric is not None:
        _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')


if __name__ == '__main__':
    main()

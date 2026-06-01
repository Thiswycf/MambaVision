"""
Evaluate trained MambaVision NAS supernet subnets.

Example:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 conda run -n mambavision \
  python nas/evaluate_supernet.py -c nas/configs/supernet_tiny.yaml \
  --checkpoint nas/checkpoints/supernet_tiny/supernet_tiny_best.pth \
  --batch-size 16 --workers 4 --amp
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn as nn
import yaml
from timm.data import create_dataset, create_loader, resolve_data_config

sys.path.insert(0, "/public/zhanghaojie/MambaVision")
sys.path.insert(0, "/public/zhanghaojie/MambaVision/mambavision")

from nas.search_space import (  # noqa: E402
    TOTAL_LAYERS,
    compute_genotype_throughput,
    get_max_subnet_genotype,
    get_mid_subnet_genotype,
    get_min_subnet_genotype,
    validate_genotype,
)
from nas.supernet import MambaVisionSuperNet  # noqa: E402
from nas.utils import AverageMeter, accuracy  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MambaVision SuperNet subnets")
    parser.add_argument("-c", "--config", default="nas/configs/supernet_tiny.yaml")
    parser.add_argument("--checkpoint", default="nas/checkpoints/supernet_tiny/supernet_tiny_best.pth")
    parser.add_argument("--use-ema", action="store_true", help="load state_dict_ema when present")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--throughput-iters", type=int, default=100)
    parser.add_argument("--throughput-warmup", type=int, default=10)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--crop-pct", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--limit-batches", type=int, default=0, help="debug only; 0 means full val set")
    parser.add_argument("--output-json", default="nas/eval_supernet_tiny.json")
    parser.add_argument("--output-csv", default="nas/eval_supernet_tiny.csv")
    parser.add_argument(
        "--genotype",
        action="append",
        default=[],
        metavar="NAME:GENE",
        help="extra subnet genotype; can be used multiple times",
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if args.data_dir is None:
        args.data_dir = cfg.get("data_dir", "/home/lqz25zhj/data/ImageNet1k")
    if args.dataset is None:
        args.dataset = cfg.get("dataset", "")
    if args.val_split is None:
        args.val_split = cfg.get("val_split", "val")
    if args.crop_pct is None:
        args.crop_pct = cfg.get("crop_pct", 1.0)
    args.img_size = cfg.get("img_size", 224)
    args.input_size = cfg.get("input_size", [3, args.img_size, args.img_size])
    args.num_classes = cfg.get("num_classes", 1000)
    args.supernet_dim = cfg.get("supernet_dim", 80)
    args.supernet_in_dim = cfg.get("supernet_in_dim", 32)
    args.supernet_depths = cfg.get("supernet_depths", [1, 3, 8, 4])
    args.supernet_num_heads = cfg.get("supernet_num_heads", [2, 4, 8, 16])
    args.supernet_window_size = cfg.get("supernet_window_size", [8, 8, 14, 7])
    args.drop_path = cfg.get("drop_path", 0.1)
    args.drop_rate = cfg.get("drop_rate", 0.0)
    args.attn_drop_rate = cfg.get("attn_drop_rate", 0.0)
    return args


def default_genotypes(extra):
    result = OrderedDict(
        [
            ("max_all_attention", get_max_subnet_genotype()),
            ("mid_all_mamba", get_mid_subnet_genotype()),
            ("min_all_conv", get_min_subnet_genotype()),
            ("mambavision_tiny_original", "CCCCMMMMAAAAMMAA"),
            ("conv_stem_mamba_tail", "CCCCMMMMMMMMMMMM"),
            ("conv_stem_attention_tail", "CCCCAAAAAAAAAAAA"),
            ("original_stage3_attention", "CCCCMMMMAAAAAAAA"),
            ("original_stage2_mamba", "CCCCMMMMMMMMMMAA"),
        ]
    )
    for item in extra:
        if ":" not in item:
            raise ValueError(f"--genotype must be NAME:GENE, got {item}")
        name, gene = item.split(":", 1)
        result[name] = gene
    for name, gene in result.items():
        if not validate_genotype(gene):
            raise ValueError(f"Invalid genotype for {name}: {gene}; expected {TOTAL_LAYERS} chars from C/M/A")
    return result


def count_active_params(model, genotype):
    shared_modules = [
        model.patch_embed,
        model.norm,
        model.avgpool,
        model.head,
    ]
    seen = set()

    def add_module(module):
        total = 0
        for param in module.parameters():
            if id(param) not in seen:
                total += param.numel()
                seen.add(id(param))
        return total

    total = sum(add_module(module) for module in shared_modules)
    cursor = 0
    for stage in model.stages:
        if stage.downsample is not None:
            total += add_module(stage.downsample)
        for block in stage.blocks:
            op = genotype[cursor]
            cursor += 1
            if op == "C":
                total += add_module(block.op_c)
            elif op == "M":
                total += add_module(block.norm_m)
                total += add_module(block.op_m)
                total += add_module(block.norm2)
                total += add_module(block.mlp)
            elif op == "A":
                total += add_module(block.norm_a)
                total += add_module(block.op_a)
                total += add_module(block.norm2)
                total += add_module(block.mlp)
            if block.gamma_1 is not None:
                for param in (block.gamma_1, block.gamma_2):
                    if id(param) not in seen:
                        total += param.numel()
                        seen.add(id(param))
    return total / 1e6

from thop import profile

def count_flops(model, genotype, shape):
    class ModelWrapper(nn.Module):
        def __init__(self, model, genotype):
            super().__init__()
            self.model = model
            self.genotype = genotype
        def forward(self, x):
            return self.model(x, self.genotype)
    wrapped_model = ModelWrapper(model, genotype)
    flops, _ = profile(wrapped_model, inputs=(torch.randn(shape).cuda(),), verbose=False)
    return flops / 1e9


def build_loader(args):
    args.local_rank = 0
    data_config = resolve_data_config(vars(args), verbose=True)
    dataset = create_dataset(args.dataset, root=args.data_dir, split=args.val_split, is_training=False)
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=True,
    )
    return loader, data_config


def load_model(args):
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    key = "state_dict_ema" if args.use_ema and isinstance(checkpoint.get("state_dict_ema"), dict) else "state_dict"
    state_dict = checkpoint[key] if key in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model, checkpoint, key


@torch.no_grad()
def evaluate_one(model, loader, loss_fn, genotype, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    image_time = AverageMeter()
    autocast = torch.cuda.amp.autocast if args.amp else nullcontext

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    end = start
    total_images = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with autocast():
            outputs = model(inputs, genotype)
            loss = loss_fn(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        torch.cuda.synchronize()
        elapsed = time.time() - end
        bs = inputs.size(0)
        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)
        batch_time.update(elapsed)
        image_time.update(elapsed / bs, bs)
        total_images += bs
        end = time.time()

        if args.log_interval and (batch_idx + 1) % args.log_interval == 0:
            print(
                f"  [{batch_idx + 1:04d}/{len(loader)}] "
                f"loss={losses.avg:.4f} top1={top1.avg:.4f} top5={top5.avg:.4f} "
                f"img/s={total_images / (time.time() - start):.2f}",
                flush=True,
            )
        if args.limit_batches and (batch_idx + 1) >= args.limit_batches:
            break

    return OrderedDict(
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg,
    )


def main():
    args = parse_args()
    genotypes = default_genotypes(args.genotype)
    loader, data_config = build_loader(args)
    model, checkpoint, state_key = load_model(args)
    loss_fn = nn.CrossEntropyLoss().cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    metadata = OrderedDict(
        checkpoint=args.checkpoint,
        state_key=state_key,
        checkpoint_epoch=checkpoint.get("epoch"),
        checkpoint_best_metric=checkpoint.get("best_metric"),
        checkpoint_best_epoch=checkpoint.get("best_epoch"),
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=args.amp,
        channels_last=args.channels_last,
        throughput_batch_size=args.throughput_batch_size,
        throughput_iters=args.throughput_iters,
        throughput_warmup=args.throughput_warmup,
        data_config=data_config,
        total_supernet_params=total_params,
    )
    print(json.dumps(metadata, indent=2, default=str), flush=True)

    rows = []
    for name, genotype in genotypes.items():
        print(f"\nEvaluating {name}: {genotype}", flush=True)
        throughput_metrics = compute_genotype_throughput(
            model,
            genotype,
            batch_size=args.throughput_batch_size,
            input_resolution=tuple(args.input_size),
            num_iterations=args.throughput_iters,
            num_warmup=args.throughput_warmup,
            use_amp=args.amp,
            channels_last=args.channels_last,
        )
        metrics = evaluate_one(model, loader, loss_fn, genotype, args)
        row = OrderedDict(
            name=name,
            genotype=genotype,
            active_params=count_active_params(model, genotype),
            FLOPs=count_flops(model, genotype, shape=(args.batch_size, *args.input_size)),
            throughput_img_per_sec=throughput_metrics['throughput'],
        )
        row.update(metrics)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    output = OrderedDict(metadata=metadata, results=rows)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()

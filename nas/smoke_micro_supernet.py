"""
Forward and throughput smoke test for a MambaVision NAS supernet config.

The script does not require a checkpoint. It verifies that a width-scaled
supernet can instantiate and run several fixed genotypes before GPU-hours are
spent on training.
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext

import torch
import yaml

sys.path.insert(0, "/public/zhanghaojie/MambaVision")

from nas.search_space import TOTAL_LAYERS, validate_genotype  # noqa: E402
from nas.supernet import MambaVisionSuperNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test a MambaVision SuperNet config")
    parser.add_argument("-c", "--config", default="nas/configs/supernet_micro_0p5.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="refine-logs/micro_0p5_smoke.json")
    parser.add_argument(
        "--genotype",
        action="append",
        default=[],
        metavar="NAME:GENE",
        help="extra genotype to test; can be repeated",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def default_genotypes(extra):
    genotypes = OrderedDict(
        [
            ("all_attention", "A" * TOTAL_LAYERS),
            ("all_mamba", "M" * TOTAL_LAYERS),
            ("all_conv", "C" * TOTAL_LAYERS),
            ("mambavision_original", "CCCCMMMMAAAAMMAA"),
            ("mid_m_late_a", "CCCCMMMMAAAAAAAA"),
            ("stage1_m_late_a", "CMMMAAAAAAAAAAAA"),
        ]
    )
    for item in extra:
        if ":" not in item:
            raise ValueError(f"--genotype must be NAME:GENE, got {item}")
        name, gene = item.split(":", 1)
        genotypes[name] = gene
    for name, gene in genotypes.items():
        if not validate_genotype(gene):
            raise ValueError(f"Invalid genotype for {name}: {gene}")
    return genotypes


def count_active_params(model, genotype):
    seen = set()

    def add_module(module):
        total = 0
        for param in module.parameters():
            if id(param) not in seen:
                total += param.numel()
                seen.add(id(param))
        return total

    total = 0
    for module in (model.patch_embed, model.norm, model.avgpool, model.head):
        total += add_module(module)

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
    return total


def build_model(cfg):
    return MambaVisionSuperNet(
        depths=cfg.get("supernet_depths", [1, 3, 8, 4]),
        num_heads=cfg.get("supernet_num_heads", [1, 2, 4, 8]),
        window_size=cfg.get("supernet_window_size", [8, 8, 14, 7]),
        dim=int(cfg.get("supernet_dim", 40)),
        in_dim=int(cfg.get("supernet_in_dim", 16)),
        drop_path_rate=float(cfg.get("drop_path", 0.1)),
        num_classes=int(cfg.get("num_classes", 1000)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
        attn_drop_rate=float(cfg.get("attn_drop_rate", 0.0)),
    )


@torch.no_grad()
def run_one(model, inputs, genotype, args, device):
    autocast = torch.cuda.amp.autocast if args.amp and device.type == "cuda" else nullcontext
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with autocast():
        output = model(inputs, genotype)
    if tuple(output.shape) != (inputs.shape[0], int(model.num_classes)):
        raise RuntimeError(f"Unexpected output shape for {genotype}: {tuple(output.shape)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(args.warmup):
        with autocast():
            model(inputs, genotype)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.iters):
        with autocast():
            model(inputs, genotype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = elapsed / max(1, args.iters)
    return OrderedDict(
        output_shape=list(output.shape),
        active_params_m=count_active_params(model, genotype) / 1e6,
        avg_time_sec=avg_time,
        throughput_img_per_sec=inputs.shape[0] / avg_time if avg_time > 0 else None,
        peak_memory_mb=torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else None,
    )


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    genotypes = default_genotypes(args.genotype)

    model = build_model(cfg).to(device).eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    input_size = list(cfg.get("input_size", [3, 224, 224]))
    inputs = torch.randn(args.batch_size, *input_size, device=device)
    if args.channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)

    rows = []
    for name, genotype in genotypes.items():
        print(f"Smoke {name}: {genotype}", flush=True)
        metrics = run_one(model, inputs, genotype, args, device)
        row = OrderedDict(name=name, genotype=genotype)
        row.update(metrics)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    output = OrderedDict(
        metadata=OrderedDict(
            config=args.config,
            batch_size=args.batch_size,
            iters=args.iters,
            warmup=args.warmup,
            device=str(device),
            amp=args.amp,
            channels_last=args.channels_last,
            total_supernet_params_m=sum(p.numel() for p in model.parameters()) / 1e6,
            input_size=input_size,
        ),
        results=rows,
    )
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

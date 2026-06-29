"""
Analyze model-relative sequence pressure for MambaVision NAS.

This script is the zero-GPU first step for the SCH-NAS pilot. It computes
stage-level token/channel ratios for multiple width scales and summarizes the
existing 1.0x cache so that later 0.5x experiments have fixed, reproducible
templates instead of ad hoc genotype choices.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, OrderedDict

import yaml

sys.path.insert(0, "/public/zhanghaojie/MambaVision")

from nas.search_space import OPS, STAGE_DEPTHS, STAGE_INDICES, TOTAL_LAYERS, validate_genotype  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SCH-NAS sequence pressure and cache templates")
    parser.add_argument("--cache", default="nas/logs/evolution/global_evaluation_cache.json")
    parser.add_argument("--config", default="nas/configs/supernet_tiny.yaml")
    parser.add_argument("--scales", default="1.0,0.75,0.5,0.375")
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--cost-objective", choices=["throughput", "flops", "params"], default="throughput")
    parser.add_argument("--out", default="refine-logs/sequence_pressure_0p5.json")
    parser.add_argument("--out-csv", default="refine-logs/sequence_pressure_stage_profiles.csv")
    parser.add_argument("--out-md", default="refine-logs/sequence_pressure_summary.md")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_cache(path):
    with open(path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        rows = list(raw.values())
    else:
        rows = raw
    normalized = []
    for row in rows:
        genotype = row.get("genotype")
        if not genotype or not validate_genotype(genotype):
            continue
        item = {"genotype": genotype}
        for key in ("top1", "top5", "loss", "flops", "params", "throughput"):
            value = row.get(key)
            item[key] = None if value is None else float(value)
        normalized.append(item)
    return normalized


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def stage_dims(base_dim, scale):
    return [max(8, int(round(base_dim * scale)) * (2 ** stage)) for stage in range(len(STAGE_DEPTHS))]


def stage_profiles(cfg, scales):
    img_size = int(cfg.get("img_size", 224))
    base_dim = int(cfg.get("supernet_dim", 80))
    window_sizes = list(cfg.get("supernet_window_size", [8, 8, 14, 7]))
    depths = list(cfg.get("supernet_depths", STAGE_DEPTHS))
    profiles = []

    for scale in scales:
        dims = stage_dims(base_dim, scale)
        for stage, depth in enumerate(depths):
            resolution = img_size // (4 * (2 ** stage))
            tokens = resolution * resolution
            dim = dims[stage]
            window_size = int(window_sizes[stage])
            window_tokens = min(tokens, window_size * window_size)
            token_dim_ratio = tokens / dim
            window_token_dim_ratio = window_tokens / dim
            spi = token_dim_ratio if stage < 2 else window_token_dim_ratio
            profiles.append(
                OrderedDict(
                    scale=scale,
                    stage=stage,
                    depth=depth,
                    resolution=resolution,
                    tokens=tokens,
                    dim=dim,
                    window_size=window_size,
                    window_tokens=window_tokens,
                    token_dim_ratio=token_dim_ratio,
                    window_token_dim_ratio=window_token_dim_ratio,
                    spi=spi,
                )
            )
    return profiles


def dominates(a, b, cost_objective):
    if cost_objective == "throughput":
        return (
            a["top1"] >= b["top1"]
            and a["throughput"] >= b["throughput"]
            and (a["top1"] > b["top1"] or a["throughput"] > b["throughput"])
        )
    key = cost_objective
    return (
        a["top1"] >= b["top1"]
        and a[key] <= b[key]
        and (a["top1"] > b["top1"] or a[key] < b[key])
    )


def pareto_front(rows, cost_objective):
    valid = [r for r in rows if r.get("top1") is not None and r.get(cost_objective) is not None]
    front = []
    for row in valid:
        if not any(dominates(other, row, cost_objective) for other in valid if other is not row):
            front.append(row)
    return sorted(front, key=lambda r: (-r["top1"], -r.get("throughput", 0.0)))


def op_frequencies(rows):
    position_counts = [Counter() for _ in range(TOTAL_LAYERS)]
    stage_counts = [Counter() for _ in STAGE_DEPTHS]
    for row in rows:
        genotype = row["genotype"]
        for idx, op in enumerate(genotype):
            position_counts[idx][op] += 1
        for stage, (start, end) in enumerate(STAGE_INDICES):
            stage_counts[stage].update(genotype[start:end])

    def normalize(counter):
        total = sum(counter.values())
        if total == 0:
            return {op: 0.0 for op in OPS}
        return {op: counter.get(op, 0) / total for op in OPS}

    return OrderedDict(
        by_position=[normalize(c) for c in position_counts],
        by_stage=[normalize(c) for c in stage_counts],
    )


def make_genotype(stage0, stage1, stage2, stage3):
    genotype = stage0 + stage1 + stage2 + stage3
    if not validate_genotype(genotype):
        raise ValueError(f"Invalid template genotype: {genotype}")
    return genotype


def template_specs():
    specs = [
        ("all_attention", make_genotype("A", "AAA", "AAAAAAAA", "AAAA"), "upper-compute all attention endpoint"),
        ("all_mamba", make_genotype("M", "MMM", "MMMMMMMM", "MMMM"), "all Mamba endpoint"),
        ("all_conv", make_genotype("C", "CCC", "CCCCCCCC", "CCCC"), "fast all convolution endpoint"),
        ("mambavision_original", "CCCCMMMMAAAAMMAA", "published MambaVision-style ordering used by the current project"),
        ("conv_stem_attention_tail", make_genotype("C", "CCC", "AAAAAAAA", "AAAA"), "Conv stem with attention-dominated later stages"),
        ("conv_stem_mamba_tail", make_genotype("C", "CCC", "MMMMMMMM", "MMMM"), "Conv stem with Mamba-dominated later stages"),
        ("mid_m_late_a", make_genotype("C", "CCC", "MMMMAAAA", "AAAA"), "Mamba in high-SPI middle blocks, attention late"),
        ("stage1_m_late_a", make_genotype("C", "MMM", "AAAAAAAA", "AAAA"), "moves Mamba into the highest token/channel conv stage"),
        ("early_m_late_a", make_genotype("M", "MMM", "AAAAAAAA", "AAAA"), "stress test for very early Mamba under 0.5x width"),
        ("c_heavy_late_a", make_genotype("C", "CCC", "CCCCAAAA", "AAAA"), "throughput-oriented Conv-heavy prefix with attention tail"),
        ("mixed_ma_late_a", make_genotype("C", "CCC", "MAMAMAMA", "AAAA"), "alternating Mamba/attention in the middle stage"),
        ("stage2_m_stage3_a", make_genotype("C", "CCC", "MMMMMMMM", "AAAA"), "isolates all-Mamba middle stage against all-attention tail"),
    ]
    return specs


def stage_op_counts(genotype):
    result = []
    for stage, (start, end) in enumerate(STAGE_INDICES):
        counts = Counter(genotype[start:end])
        result.append({op: counts.get(op, 0) for op in OPS})
    return result


def spi_weighted_mamba(genotype, profiles, scale=0.5):
    scale_profiles = {p["stage"]: p for p in profiles if abs(p["scale"] - scale) < 1e-9}
    numerator = 0.0
    denominator = 0.0
    for stage, (start, end) in enumerate(STAGE_INDICES):
        spi = scale_profiles[stage]["spi"]
        depth = end - start
        denominator += depth * spi
        numerator += sum(1 for op in genotype[start:end] if op == "M") * spi
    return 0.0 if denominator == 0.0 else numerator / denominator


def cache_summary(rows, cost_objective, top_fraction):
    top1_values = [r["top1"] for r in rows if r.get("top1") is not None]
    ranked = sorted([r for r in rows if r.get("top1") is not None], key=lambda r: r["top1"], reverse=True)
    top_count = max(1, int(round(len(ranked) * top_fraction))) if ranked else 0
    top_rows = ranked[:top_count]
    front = pareto_front(rows, cost_objective)
    return OrderedDict(
        total_rows=len(rows),
        top_fraction=top_fraction,
        top_count=top_count,
        top1_quantiles=OrderedDict(
            q00=quantile(top1_values, 0.0),
            q25=quantile(top1_values, 0.25),
            q50=quantile(top1_values, 0.5),
            q75=quantile(top1_values, 0.75),
            q95=quantile(top1_values, 0.95),
            q100=quantile(top1_values, 1.0),
        ),
        top_op_frequency=op_frequencies(top_rows),
        pareto_count=len(front),
        pareto_op_frequency=op_frequencies(front),
        top_examples=ranked[:10],
        pareto_examples=front[:10],
    )


def add_template_cache_metrics(specs, rows, profiles):
    row_by_genotype = {row["genotype"]: row for row in rows}
    templates = []
    for name, genotype, rationale in specs:
        templates.append(
            OrderedDict(
                name=name,
                genotype=genotype,
                rationale=rationale,
                stage_op_counts=stage_op_counts(genotype),
                spi_weighted_mamba_0p5=spi_weighted_mamba(genotype, profiles, scale=0.5),
                cache_metrics=row_by_genotype.get(genotype),
            )
        )
    return templates


def write_csv(path, profiles):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profiles[0].keys()))
        writer.writeheader()
        writer.writerows(profiles)


def write_markdown(path, output):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Sequence Pressure Summary\n\n")
        f.write(f"Cache rows: {output['cache_summary']['total_rows']}\n\n")
        f.write("## Stage Profiles\n\n")
        f.write("| Scale | Stage | Resolution | Tokens | Dim | Token/Dim | Window Token/Dim | SPI |\n")
        f.write("|-------|-------|------------|--------|-----|-----------|------------------|-----|\n")
        for row in output["stage_profiles"]:
            f.write(
                f"| {row['scale']} | {row['stage']} | {row['resolution']} | {row['tokens']} | "
                f"{row['dim']} | {row['token_dim_ratio']:.3f} | "
                f"{row['window_token_dim_ratio']:.3f} | {row['spi']:.3f} |\n"
            )
        f.write("\n## Fixed Templates\n\n")
        f.write("| Name | Genotype | SPI-weighted Mamba@0.5 | Cache Top1 | Cache Throughput |\n")
        f.write("|------|----------|-------------------------|------------|------------------|\n")
        for tpl in output["templates"]:
            metrics = tpl.get("cache_metrics") or {}
            top1 = "" if metrics.get("top1") is None else f"{metrics['top1']:.3f}"
            throughput = "" if metrics.get("throughput") is None else f"{metrics['throughput']:.1f}"
            f.write(
                f"| {tpl['name']} | `{tpl['genotype']}` | {tpl['spi_weighted_mamba_0p5']:.3f} | "
                f"{top1} | {throughput} |\n"
            )
        f.write("\n")


def main():
    args = parse_args()
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    cfg = load_yaml(args.config)
    rows = load_cache(args.cache)
    profiles = stage_profiles(cfg, scales)
    summary = cache_summary(rows, args.cost_objective, args.top_fraction)
    templates = add_template_cache_metrics(template_specs(), rows, profiles)

    output = OrderedDict(
        metadata=OrderedDict(
            config=args.config,
            cache=args.cache,
            scales=scales,
            cost_objective=args.cost_objective,
            top_fraction=args.top_fraction,
        ),
        stage_profiles=profiles,
        cache_summary=summary,
        templates=templates,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    write_csv(args.out_csv, profiles)
    write_markdown(args.out_md, output)

    best = summary["top_examples"][0] if summary["top_examples"] else None
    print(f"Wrote {args.out}, {args.out_csv}, and {args.out_md}")
    print(f"Cache rows: {summary['total_rows']}; Pareto rows: {summary['pareto_count']}")
    if best:
        print(f"Best cached top1: {best['genotype']} top1={best['top1']:.3f}")
    print("Templates:")
    for tpl in templates:
        print(f"  {tpl['name']}: {tpl['genotype']}")


if __name__ == "__main__":
    main()

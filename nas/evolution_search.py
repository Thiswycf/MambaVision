"""
NSGA-II evolution search for MambaVision NAS supernet subnets.

The objectives are:
  - maximize ImageNet Top-1 accuracy using nas/evaluate_supernet.py validation logic
  - minimize total FLOPs using the same FLOPs counter as nas/evaluate_supernet.py

Example:
CUDA_DEVICE_ORDER=PCI_BUS_ID conda run -n mambavision \
  python nas/evolution_search.py -c nas/configs/evolution_search.yaml
"""

import argparse
import base64
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import yaml

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
try:
    from tensorboardX import SummaryWriter  # noqa: E402
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
sys.path.insert(0, "/public/zhanghaojie/MambaVision")
sys.path.insert(0, "/public/zhanghaojie/MambaVision/mambavision")

from nas.cache import EvaluationCache  # noqa: E402
from nas.evaluate_supernet import (  # noqa: E402
    build_loader,
    count_flops,
    evaluate_one,
    load_model,
)
from nas.search_space import OPS, TOTAL_LAYERS, random_genotype, validate_genotype, compute_genotype_throughput  # noqa: E402
from nas.evaluate_supernet import count_active_params  # noqa: E402


STATE_FILE = "search_state.json"
FINAL_ARCHIVE = "pareto_archive.json"


def parse_args():
    parser = argparse.ArgumentParser(description="NSGA-II search for MambaVision supernet")
    parser.add_argument("-c", "--config", default="nas/configs/evolution_search.yaml")
    parser.add_argument("--resume", action="store_true", help="resume from nas/logs/evolution/search_state.json")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing search_state.json")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--cost-objective", type=str, default=None, 
                        choices=["flops", "params", "throughput"],
                        help="Cost objective for multi-objective optimization (default: from config)")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_config(cli_args):
    cfg = load_yaml(cli_args.config)
    cfg.setdefault("evolution", {})
    cfg.setdefault("evaluation", {})
    cfg.setdefault("gpu", {})
    cfg.setdefault("cache", {})
    cfg.setdefault("log_dir", "nas/logs/evolution")
    cfg.setdefault("supernet_config", "nas/configs/supernet_tiny.yaml")

    if cli_args.log_dir:
        cfg["log_dir"] = cli_args.log_dir
    if cli_args.checkpoint:
        cfg["checkpoint"] = cli_args.checkpoint
    if cli_args.seed is not None:
        cfg["evolution"]["seed"] = cli_args.seed
    if cli_args.generations is not None:
        cfg["evolution"]["generations"] = cli_args.generations
    if cli_args.population_size is not None:
        cfg["evolution"]["population_size"] = cli_args.population_size
    if cli_args.cost_objective is not None:
        cfg["evolution"]["cost_objective"] = cli_args.cost_objective
    if cli_args.no_resume:
        cfg["resume"] = False
    elif cli_args.resume:
        cfg["resume"] = True

    cfg["evolution"].setdefault("population_size", 128)
    cfg["evolution"].setdefault("generations", 100)
    cfg["evolution"].setdefault("crossover_prob", 1.0)
    cfg["evolution"].setdefault("mutation_prob", 0.2)
    cfg["evolution"].setdefault("mutation_min_sites", 1)
    cfg["evolution"].setdefault("mutation_max_sites", 2)
    cfg["evolution"].setdefault("archive_interval", 10)
    cfg["evolution"].setdefault("seed", 42)
    cfg["evolution"].setdefault("cost_objective", "flops")

    cost_objective = cfg["evolution"]["cost_objective"].lower()
    if cost_objective not in ["flops", "params", "throughput"]:
        raise ValueError(f"Invalid cost_objective: {cost_objective}. Must be one of: flops, params, throughput")
    cfg["evolution"]["cost_objective"] = cost_objective

    cost_subdir = cost_objective
    base_log_dir = cfg["log_dir"]
    cfg["log_dir"] = os.path.join(base_log_dir, cost_subdir)
    cfg["cache"]["global_cache_path"] = os.path.join(base_log_dir, "global_evaluation_cache.json")

    cfg["cache"].setdefault("use_global_cache", True)
    cfg["cache"].setdefault("save_global_cache", True)

    cfg["gpu"].setdefault("device_ids", None)
    return cfg


def build_eval_args(config):
    base_cfg = load_yaml(config["supernet_config"])
    eval_cfg = deepcopy(base_cfg)
    deep_update(eval_cfg, config.get("evaluation", {}))
    eval_cfg["config"] = config["supernet_config"]
    eval_cfg["checkpoint"] = config.get("checkpoint", "nas/checkpoints/supernet_tiny/supernet_tiny_best.pth")
    eval_cfg["use_ema"] = bool(config.get("use_ema", False))

    if eval_cfg.get("data_dir") is None:
        eval_cfg["data_dir"] = base_cfg.get("data_dir", "/home/lqz25zhj/data/ImageNet1k")
    if eval_cfg.get("dataset") is None:
        eval_cfg["dataset"] = base_cfg.get("dataset", "")
    if eval_cfg.get("val_split") is None:
        eval_cfg["val_split"] = base_cfg.get("val_split", "val")
    if eval_cfg.get("crop_pct") is None:
        eval_cfg["crop_pct"] = base_cfg.get("crop_pct", 1.0)

    eval_cfg.setdefault("batch_size", 128)
    eval_cfg.setdefault("workers", 8)
    eval_cfg.setdefault("amp", False)
    eval_cfg.setdefault("channels_last", False)
    eval_cfg.setdefault("limit_batches", 0)
    eval_cfg.setdefault("log_interval", 0)
    eval_cfg.setdefault("img_size", base_cfg.get("img_size", 224))
    eval_cfg.setdefault("input_size", base_cfg.get("input_size", [3, eval_cfg["img_size"], eval_cfg["img_size"]]))
    eval_cfg.setdefault("num_classes", base_cfg.get("num_classes", 1000))
    eval_cfg.setdefault("supernet_dim", base_cfg.get("supernet_dim", 80))
    eval_cfg.setdefault("supernet_in_dim", base_cfg.get("supernet_in_dim", 32))
    eval_cfg.setdefault("supernet_depths", base_cfg.get("supernet_depths", [1, 3, 8, 4]))
    eval_cfg.setdefault("supernet_num_heads", base_cfg.get("supernet_num_heads", [2, 4, 8, 16]))
    eval_cfg.setdefault("supernet_window_size", base_cfg.get("supernet_window_size", [8, 8, 14, 7]))
    eval_cfg.setdefault("drop_path", base_cfg.get("drop_path", 0.1))
    eval_cfg.setdefault("drop_rate", base_cfg.get("drop_rate", 0.0))
    eval_cfg.setdefault("attn_drop_rate", base_cfg.get("attn_drop_rate", 0.0))
    eval_cfg.setdefault("throughput_batch_size", 128)
    eval_cfg.setdefault("throughput_iters", 100)
    eval_cfg.setdefault("throughput_warmup", 10)
    return SimpleNamespace(**eval_cfg)


def encode_random_state():
    return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")


def restore_random_state(value):
    random.setstate(pickle.loads(base64.b64decode(value.encode("ascii"))))


def select_gpu_ids(gpu_cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for supernet evaluation")

    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("No visible CUDA devices")

    configured_ids = gpu_cfg.get("device_ids")
    if not configured_ids:
        raise ValueError("Set gpu.device_ids in the YAML config, for example: device_ids: [1, 2, 3, 4]")

    selected = [int(x) for x in configured_ids]
    invalid = [idx for idx in selected if idx < 0 or idx >= count]
    if invalid:
        raise ValueError(f"Invalid gpu.device_ids {invalid}; visible CUDA device count is {count}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"gpu.device_ids contains duplicate entries: {selected}")

    print(f"Using CUDA devices from YAML gpu.device_ids: {selected}", flush=True)
    return selected


def initial_population(size):
    population = []
    seen = set()
    seeds = ["A" * TOTAL_LAYERS, "M" * TOTAL_LAYERS, "C" * TOTAL_LAYERS, "CCCCMMMMAAAAMMAA"]
    for genotype in seeds:
        if genotype not in seen and len(population) < size:
            population.append(genotype)
            seen.add(genotype)
    while len(population) < size:
        genotype = random_genotype()
        if genotype not in seen:
            population.append(genotype)
            seen.add(genotype)
    return population


def dominates(a, b, cost_objective="flops"):
    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    if cost_objective == "throughput":
        return (
            a["top1"] >= b["top1"]
            and a["throughput"] >= b["throughput"]
            and (a["top1"] > b["top1"] or a["throughput"] > b["throughput"])
        )
    else:
        return (
            a["top1"] >= b["top1"]
            and a[cost_key] <= b[cost_key]
            and (a["top1"] > b["top1"] or a[cost_key] < b[cost_key])
        )


def non_dominated_sort(individuals, cost_objective="flops"):
    fronts = []
    dominated_sets = [[] for _ in individuals]
    domination_counts = [0 for _ in individuals]
    first_front = []

    for p_idx, p in enumerate(individuals):
        for q_idx, q in enumerate(individuals):
            if p_idx == q_idx:
                continue
            if dominates(p, q, cost_objective):
                dominated_sets[p_idx].append(q_idx)
            elif dominates(q, p, cost_objective):
                domination_counts[p_idx] += 1
        if domination_counts[p_idx] == 0:
            p["rank"] = 0
            first_front.append(p_idx)

    fronts.append(first_front)
    rank = 0
    while fronts[rank]:
        next_front = []
        for p_idx in fronts[rank]:
            for q_idx in dominated_sets[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    individuals[q_idx]["rank"] = rank + 1
                    next_front.append(q_idx)
        rank += 1
        fronts.append(next_front)
    return [[individuals[idx] for idx in front] for front in fronts[:-1]]


def assign_crowding_distance(front, cost_objective="flops"):
    if not front:
        return
    for individual in front:
        individual["crowding"] = 0.0
    if len(front) <= 2:
        for individual in front:
            individual["crowding"] = float("inf")
        return

    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    for objective in ("top1", cost_key):
        reverse = objective == "top1" or (objective == "throughput" and cost_objective == "throughput")
        ordered = sorted(front, key=lambda x: x[objective], reverse=reverse)
        ordered[0]["crowding"] = float("inf")
        ordered[-1]["crowding"] = float("inf")
        min_value = min(x[objective] for x in front)
        max_value = max(x[objective] for x in front)
        denom = max(max_value - min_value, 1e-12)
        for idx in range(1, len(ordered) - 1):
            prev_value = ordered[idx - 1][objective]
            next_value = ordered[idx + 1][objective]
            ordered[idx]["crowding"] += abs(next_value - prev_value) / denom


def annotate_rank_and_crowding(individuals, cost_objective="flops"):
    fronts = non_dominated_sort(individuals, cost_objective)
    for front in fronts:
        assign_crowding_distance(front, cost_objective)
    return fronts


def select_next_population(candidates, size, cost_objective="flops"):
    unique = {}
    for item in candidates:
        g = item["genotype"]
        if g not in unique:
            unique[g] = item
    deduped = list(unique.values())

    next_population = []
    fronts = annotate_rank_and_crowding(deduped, cost_objective)
    for front in fronts:
        if len(next_population) + len(front) <= size:
            next_population.extend(front)
        else:
            assign_crowding_distance(front, cost_objective)
            front = sorted(front, key=lambda x: x["crowding"], reverse=True)
            next_population.extend(front[: size - len(next_population)])
            break
    return [item["genotype"] for item in next_population]


def tournament_select(individuals):
    a, b = random.sample(individuals, 2)
    if a.get("rank", math.inf) != b.get("rank", math.inf):
        return a if a["rank"] < b["rank"] else b
    if a.get("crowding", 0.0) != b.get("crowding", 0.0):
        return a if a["crowding"] > b["crowding"] else b
    return a if random.random() < 0.5 else b


def two_point_crossover(parent_a, parent_b):
    if len(parent_a) != TOTAL_LAYERS or len(parent_b) != TOTAL_LAYERS:
        raise ValueError("Invalid parent genotype length")
    left, right = sorted(random.sample(range(1, TOTAL_LAYERS), 2))
    child_a = parent_a[:left] + parent_b[left:right] + parent_a[right:]
    child_b = parent_b[:left] + parent_a[left:right] + parent_b[right:]
    return child_a, child_b


def mutate(genotype, min_sites=1, max_sites=2):
    genes = list(genotype)
    site_count = random.randint(min_sites, max_sites)
    for idx in random.sample(range(TOTAL_LAYERS), site_count):
        choices = [op for op in OPS if op != genes[idx]]
        genes[idx] = random.choice(choices)
    return "".join(genes)


def make_offspring(parent_individuals, size, evo_cfg, cost_objective="flops"):
    annotate_rank_and_crowding(parent_individuals, cost_objective)
    offspring = []
    seen = set()
    crossover_prob = float(evo_cfg.get("crossover_prob", 1.0))
    mutation_prob = float(evo_cfg.get("mutation_prob", 0.2))
    min_sites = int(evo_cfg.get("mutation_min_sites", 1))
    max_sites = int(evo_cfg.get("mutation_max_sites", 2))

    attempts = 0
    while len(offspring) < size and attempts < size * 50:
        attempts += 1
        parent_a = tournament_select(parent_individuals)["genotype"]
        parent_b = tournament_select(parent_individuals)["genotype"]
        if random.random() < crossover_prob:
            children = two_point_crossover(parent_a, parent_b)
        else:
            children = (parent_a, parent_b)
        for child in children:
            if random.random() < mutation_prob:
                child = mutate(child, min_sites=min_sites, max_sites=max_sites)
            if validate_genotype(child) and child not in seen:
                offspring.append(child)
                seen.add(child)
            if len(offspring) >= size:
                break

    while len(offspring) < size:
        child = random_genotype()
        if child not in seen:
            offspring.append(child)
            seen.add(child)
    return offspring


def cache_to_individuals(population, eval_cache, cost_objective="flops"):
    individuals = []
    for genotype in population:
        if not eval_cache.has(genotype):
            continue
        cached_row = eval_cache.get(genotype)
        item = {
            "genotype": genotype,
            "top1": cached_row["top1"],
            "top5": cached_row.get("top5"),
            "loss": cached_row.get("loss"),
        }
        
        if cost_objective == "flops":
            item["flops"] = cached_row.get("flops", float("inf"))
        elif cost_objective == "params":
            item["params"] = cached_row.get("params", float("inf"))
        elif cost_objective == "throughput":
            item["throughput"] = cached_row.get("throughput", 0.0)
        
        individuals.append(item)
    return individuals


def evaluate_population(population, eval_cache, config, gpu_ids, generation):
    cost_objective = config.get("evolution", {}).get("cost_objective", "flops")
    to_evaluate_full = []
    to_evaluate_partial = []
    seen = set()

    for genotype in population:
        if genotype in seen:
            continue
        
        cached = eval_cache.get(genotype)
        if cached is not None and cached.get("top1") is not None and eval_cache.is_complete(genotype, cost_objective):
            print(f"[cache hit - complete] generation={generation} genotype={genotype}", flush=True)
            seen.add(genotype)
            continue
        
        if cached is not None and cached.get("top1") is not None:
            missing_keys = get_missing_keys(cached, cost_objective)
            print(f"[cache hit - partial, need {missing_keys}] generation={generation} genotype={genotype}", flush=True)
            to_evaluate_partial.append((genotype, missing_keys))
        else:
            print(f"[cache miss - full eval] generation={generation} genotype={genotype}", flush=True)
            to_evaluate_full.append(genotype)
        
        seen.add(genotype)

    if to_evaluate_full:
        print(f"Generation {generation}: full evaluating {len(to_evaluate_full)} genotypes on {len(gpu_ids)} GPU(s)", flush=True)
        results = run_evaluation_workers(to_evaluate_full, config, gpu_ids, cost_objective)
        eval_cache.batch_update(results)

    if to_evaluate_partial:
        print(f"Generation {generation}: partial evaluating {len(to_evaluate_partial)} genotypes on {len(gpu_ids)} GPU(s)", flush=True)
        results = run_partial_evaluation_workers(to_evaluate_partial, config, gpu_ids)
        # for result in results:
        #     eval_cache.update(result["genotype"], result)
        eval_cache.batch_update(results)

    if not to_evaluate_full and not to_evaluate_partial:
        print(f"Generation {generation}: all {len(population)} genotypes satisfied by cache", flush=True)

    return eval_cache


def get_missing_keys(cached_row, cost_objective):
    missing = []
    if cached_row.get("top1") is None:
        missing.extend(["top1", "top5", "loss"])
    if cost_objective == "throughput":
        if cached_row.get("throughput") is None:
            missing.append("throughput")
    else:
        cost_key = cost_objective
        if cached_row.get(cost_key) is None:
            missing.append(cost_key)
        if cached_row.get("params") is None:
            missing.append("params")
    return missing if missing else None


def split_partial_even(items, parts):
    chunks = [[] for _ in range(parts)]
    for idx, item in enumerate(items):
        chunks[idx % parts].append(item)
    return [chunk for chunk in chunks if chunk]


def run_partial_evaluation_workers(partial_tasks, config, gpu_ids):
    if len(gpu_ids) == 1:
        return evaluate_partial_worker(0, gpu_ids[0], partial_tasks, config, None)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    chunks = split_partial_even(partial_tasks, len(gpu_ids))
    for rank, chunk in enumerate(chunks):
        proc = ctx.Process(target=evaluate_partial_worker, args=(rank, gpu_ids[rank], chunk, config, queue))
        proc.start()
        processes.append(proc)

    results = []
    errors = []
    finished = 0
    while finished < len(processes):
        message = queue.get()
        if message["type"] == "result":
            results.extend(message["rows"])
            finished += 1
        elif message["type"] == "error":
            errors.append(message)
            finished += 1

    for proc in processes:
        proc.join()
    if errors:
        detail = "\n".join(error["message"] for error in errors)
        raise RuntimeError(f"Partial evaluation worker failed:\n{detail}")
    return results


def evaluate_partial_worker(rank, gpu_id, partial_tasks, config, queue):
    try:
        torch.cuda.set_device(gpu_id)
        args = build_eval_args(config)
        args.local_rank = 0
        
        rows = []
        
        for index, (genotype, missing_keys) in enumerate(partial_tasks, start=1):
            print(
                f"[worker {rank} cuda:{gpu_id}] partial {index}/{len(partial_tasks)} evaluating {genotype} missing={missing_keys}",
                flush=True,
            )
            
            result = evaluate_partial(genotype, missing_keys, config, gpu_id)
            rows.append(result)
            
            torch.cuda.empty_cache()
        
        if queue is not None:
            queue.put({"type": "result", "rows": rows})
        return rows
    except Exception as exc:
        if queue is not None:
            queue.put({"type": "error", "message": repr(exc)})
            return None
        raise


def run_evaluation_workers(genotypes, config, gpu_ids, cost_objective="flops"):
    if len(gpu_ids) == 1:
        return evaluate_worker(0, gpu_ids[0], genotypes, config, None, cost_objective)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    chunks = split_partial_even(genotypes, len(gpu_ids))
    for rank, chunk in enumerate(chunks):
        proc = ctx.Process(target=evaluate_worker, args=(rank, gpu_ids[rank], chunk, config, queue, cost_objective))
        proc.start()
        processes.append(proc)

    results = []
    errors = []
    finished = 0
    while finished < len(processes):
        message = queue.get()
        if message["type"] == "result":
            results.extend(message["rows"])
            finished += 1
        elif message["type"] == "error":
            errors.append(message)
            finished += 1

    for proc in processes:
        proc.join()
    if errors:
        detail = "\n".join(error["message"] for error in errors)
        raise RuntimeError(f"Evaluation worker failed:\n{detail}")
    return results


def evaluate_worker(rank, gpu_id, genotypes, config, queue, cost_objective="flops"):
    try:
        torch.cuda.set_device(gpu_id)
        args = build_eval_args(config)
        args.local_rank = 0
        loader, _ = build_loader(args)
        model, _, _ = load_model(args)
        loss_fn = nn.CrossEntropyLoss().cuda()
        rows = []
        
        throughput_batch_size = getattr(args, 'throughput_batch_size', 128)
        throughput_iters = getattr(args, 'throughput_iters', 100)
        throughput_warmup = getattr(args, 'throughput_warmup', 10)
        
        for index, genotype in enumerate(genotypes, start=1):
            print(
                f"[worker {rank} cuda:{gpu_id}] {index}/{len(genotypes)} evaluating {genotype}",
                flush=True,
            )
            start_time = time.time()
            metrics = evaluate_one(model, loader, loss_fn, genotype, args)
            
            flops = count_flops(model, genotype, shape=(128, *args.input_size))
            params = count_active_params(model, genotype)
            
            throughput_metrics = compute_genotype_throughput(
                model,
                genotype,
                batch_size=throughput_batch_size,
                input_resolution=tuple(args.input_size),
                num_iterations=throughput_iters,
                num_warmup=throughput_warmup,
                use_amp=args.amp,
                channels_last=args.channels_last,
            )
            throughput = throughput_metrics['throughput']
            
            elapsed = time.time() - start_time
            print(
                f"[worker {rank} cuda:{gpu_id}] done {genotype} "
                f"time={elapsed:.2f}s top1={float(metrics['top1']):.4f} "
                f"flops={float(flops):.6f} params={float(params):.6f} throughput={float(throughput):.2f}",
                flush=True,
            )
            rows.append(
                {
                    "genotype": genotype,
                    "top1": float(metrics['top1']),
                    "top5": float(metrics["top5"]),
                    "loss": float(metrics["loss"]),
                    "flops": float(flops),
                    "params": float(params),
                    "throughput": float(throughput),
                }
            )
            torch.cuda.empty_cache()
        if queue is not None:
            queue.put({"type": "result", "rows": rows})
        return rows
    except Exception as exc:
        if queue is not None:
            queue.put({"type": "error", "message": repr(exc)})
            return None
        raise


def evaluate_partial(genotype, missing_keys, config, gpu_id):
    try:
        torch.cuda.set_device(gpu_id)
        args = build_eval_args(config)
        args.local_rank = 0
        
        result = {"genotype": genotype}
        model = None
        loader = None
        loss_fn = None
        
        if "top1" in missing_keys or "top5" in missing_keys or "loss" in missing_keys:
            loader, _ = build_loader(args)
            model, _, _ = load_model(args)
            loss_fn = nn.CrossEntropyLoss().cuda()
            metrics = evaluate_one(model, loader, loss_fn, genotype, args)
            result["top1"] = float(metrics["top1"])
            result["top5"] = float(metrics["top5"])
            result["loss"] = float(metrics["loss"])
        
        if "flops" in missing_keys or "params" in missing_keys or "throughput" in missing_keys:
            if model is None:
                model, _, _ = load_model(args)
        
        if "flops" in missing_keys:
            result["flops"] = float(count_flops(model, genotype, shape=(128, *args.input_size)))
        
        if "params" in missing_keys:
            result["params"] = float(count_active_params(model, genotype))
        
        if "throughput" in missing_keys:
            throughput_metrics = compute_genotype_throughput(
                model,
                genotype,
                batch_size=getattr(args, 'throughput_batch_size', 128),
                input_resolution=tuple(args.input_size),
                num_iterations=getattr(args, 'throughput_iters', 100),
                num_warmup=getattr(args, 'throughput_warmup', 10),
                use_amp=args.amp,
                channels_last=args.channels_last,
            )
            result["throughput"] = float(throughput_metrics['throughput'])
        
        return result
    except Exception as exc:
        raise


def pareto_front(individuals, cost_objective="flops"):
    front = []
    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    for candidate in individuals:
        if not any(dominates(other, candidate, cost_objective) for other in individuals if other["genotype"] != candidate["genotype"]):
            front.append(candidate)
    
    if cost_objective == "throughput":
        return sorted(front, key=lambda x: (-x["throughput"], -x["top1"]))
    else:
        return sorted(front, key=lambda x: (x[cost_key], -x["top1"]))


def compute_hypervolume(front, reference_cost, cost_objective="flops"):
    if not front:
        return 0.0
    
    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    
    if cost_objective == "throughput":
        ordered = sorted(front, key=lambda x: -x["throughput"])
        hv = 0.0
        min_throughput = min(item["throughput"] for item in front) if front else 0.0
        for idx, item in enumerate(ordered):
            next_throughput = ordered[idx + 1]["throughput"] if idx + 1 < len(ordered) else min_throughput
            width = max(0.0, item["throughput"] - next_throughput)
            hv += width * max(0.0, item["top1"])
        return hv
    else:
        if reference_cost <= 0:
            return 0.0
        ordered = sorted(front, key=lambda x: x[cost_key])
        hv = 0.0
        for idx, item in enumerate(ordered):
            next_cost = ordered[idx + 1][cost_key] if idx + 1 < len(ordered) else reference_cost
            width = max(0.0, next_cost - item[cost_key])
            hv += width * max(0.0, item["top1"])
        return hv


def log_generation(writer, generation, individuals, archive, eval_cache, cost_objective="flops"):
    front = pareto_front(individuals, cost_objective)
    
    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    if eval_cache.cache is None:
        eval_cache.load()
    
    if cost_objective == "throughput":
        reference_cost = max(
            item[cost_key]
            for item in eval_cache.cache.values()
            if item.get(cost_key) is not None
        )
    else:
        reference_cost = max(
            item[cost_key]
            for item in eval_cache.cache.values()
            if item.get(cost_key) is not None
        )
    
    hypervolume = compute_hypervolume(front, reference_cost, cost_objective)
    writer.add_scalar("metrics/hypervolume", hypervolume, generation)
    writer.add_figure("pareto_front/scatter", make_scatter_figure(individuals, front, generation, cost_objective), generation, close=True)
    writer.add_text("pareto_front/individuals", make_pareto_text(front, hypervolume, reference_cost, cost_objective), generation)
    archive["history"].append(
        {
            "generation": generation,
            "hypervolume": hypervolume,
            "reference_point": {"top1": 0.0, cost_key: reference_cost},
            "pareto_size": len(front),
        }
    )
    writer.flush()
    return front, hypervolume


def make_scatter_figure(individuals, front, generation, cost_objective="flops"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=120)
    
    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    cost_label = {
        "flops": "FLOPs (G)",
        "params": "Params (M)",
        "throughput": "Throughput (img/s)"
    }.get(cost_objective, cost_objective)
    
    ax.scatter(
        [item[cost_key] for item in individuals],
        [item["top1"] for item in individuals],
        s=22,
        alpha=0.55,
        label="population",
    )
    if front:
        ax.scatter(
            [item[cost_key] for item in front],
            [item["top1"] for item in front],
            s=42,
            color="#d62728",
            label="pareto front",
        )
        ordered = sorted(front, key=lambda x: x[cost_key])
        ax.plot([item[cost_key] for item in ordered], [item["top1"] for item in ordered], color="#d62728", linewidth=1.2)
    ax.set_title(f"Generation {generation}")
    ax.set_xlabel(f"{cost_label} (evaluate_supernet total)")
    ax.set_ylabel("ImageNet Top-1 (%)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend()
    fig.tight_layout()
    return fig


def make_pareto_text(front, hypervolume, reference_cost, cost_objective="flops"):
    cost_key = cost_objective if cost_objective == "throughput" else cost_objective
    cost_label = {
        "flops": "FLOPs",
        "params": "Params",
        "throughput": "Throughput"
    }.get(cost_objective, cost_objective)
    
    lines = [
        f"Hypervolume: `{hypervolume:.6f}`  ",
        f"Reference point: `(top1=0, {cost_objective}={reference_cost:.6f})`",
        "",
        f"| Rank | Genotype | Top-1 | {cost_label} | Top-5 | Loss |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(front, start=1):
        top5 = format_optional(item.get("top5"), ".4f")
        loss = format_optional(item.get("loss"), ".6f")
        cost_value = item[cost_key] if cost_key in item else float("inf")
        lines.append(
            f"| {idx} | `{item['genotype']}` | {item['top1']:.4f} | {cost_value:.6f} | "
            f"{top5} | {loss} |"
        )
    return "\n".join(lines)


def format_optional(value, spec):
    if value is None:
        return "nan"
    return format(float(value), spec)


def individual_for_json(item, cost_objective="flops"):
    result = OrderedDict(
        genotype=item["genotype"],
        top1=float(item["top1"]),
        top5=None if item.get("top5") is None else float(item["top5"]),
        loss=None if item.get("loss") is None else float(item["loss"]),
    )
    if cost_objective == "flops":
        result["flops"] = float(item["flops"])
        result["params"] = float(item["params"]) if item.get("params") is not None else None
        result["throughput"] = float(item["throughput"]) if item.get("throughput") is not None else None
    elif cost_objective == "params":
        result["flops"] = float(item["flops"]) if item.get("flops") is not None else None
        result["params"] = float(item["params"])
        result["throughput"] = float(item["throughput"]) if item.get("throughput") is not None else None
    elif cost_objective == "throughput":
        result["flops"] = float(item["flops"]) if item.get("flops") is not None else None
        result["params"] = float(item["params"]) if item.get("params") is not None else None
        result["throughput"] = float(item["throughput"])
    return result


def write_pareto_archive(path, front, config, generation, hypervolume, history, cost_objective="flops"):
    payload = OrderedDict(
        generation=generation,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        config=config,
        hypervolume=hypervolume,
        pareto_front=[individual_for_json(item, cost_objective) for item in front],
        history=history,
    )
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_state(log_dir, generation, population, config, history):
    state = OrderedDict(
        generation=generation,
        population=population,
        config=config,
        history=history,
        random_state=encode_random_state(),
    )
    path = os.path.join(log_dir, STATE_FILE)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state(log_dir):
    path = os.path.join(log_dir, STATE_FILE)
    with open(path, "r") as f:
        state = json.load(f)
    if state.get("random_state"):
        restore_random_state(state["random_state"])
    return state


def main():
    cli_args = parse_args()
    config = build_config(cli_args)
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    evo_cfg = config["evolution"]
    cost_objective = evo_cfg.get("cost_objective", "flops")
    random.seed(int(evo_cfg["seed"]))
    gpu_ids = select_gpu_ids(config["gpu"])
    writer = SummaryWriter(logdir=log_dir)
    archive = {"history": []}
    
    cache_path = config["cache"]["global_cache_path"]
    eval_cache = EvaluationCache(cache_path)
    eval_cache.load()
    print(f"Loaded evaluation cache: {cache_path} ({len(eval_cache)} entries)", flush=True)

    state_path = os.path.join(log_dir, STATE_FILE)
    if config.get("resume", True) and os.path.exists(state_path):
        state = load_state(log_dir)
        population = state["population"]
        archive["history"] = state.get("history", [])
        start_generation = int(state.get("generation", 0)) + 1
        print(f"Resuming search from generation {start_generation}", flush=True)
    else:
        population = initial_population(int(evo_cfg["population_size"]))
        start_generation = 0

    generations = int(evo_cfg["generations"])
    population_size = int(evo_cfg["population_size"])
    final_front = []
    final_hypervolume = 0.0

    try:
        for generation in range(start_generation, generations + 1):
            eval_cache = evaluate_population(population, eval_cache, config, gpu_ids, generation)
            individuals = cache_to_individuals(population, eval_cache, cost_objective)
            fronts = annotate_rank_and_crowding(individuals, cost_objective)
            final_front, final_hypervolume = log_generation(writer, generation, individuals, archive, eval_cache, cost_objective)

            write_pareto_archive(
                os.path.join(log_dir, FINAL_ARCHIVE),
                final_front,
                config,
                generation,
                final_hypervolume,
                archive["history"],
                cost_objective,
            )
            if generation % int(evo_cfg["archive_interval"]) == 0:
                write_pareto_archive(
                    os.path.join(log_dir, f"pareto_archive_gen_{generation:04d}.json"),
                    final_front,
                    config,
                    generation,
                    final_hypervolume,
                    archive["history"],
                    cost_objective,
                )
            print(
                f"Generation {generation}: pareto={len(final_front)} "
                f"hypervolume={final_hypervolume:.6f}",
                flush=True,
            )

            if generation >= generations:
                save_state(log_dir, generation, population, config, archive["history"])
                break

            offspring = make_offspring(individuals, population_size, evo_cfg, cost_objective)
            eval_cache = evaluate_population(offspring, eval_cache, config, gpu_ids, generation)
            offspring_individuals = cache_to_individuals(offspring, eval_cache, cost_objective)
            population = select_next_population(individuals + offspring_individuals, population_size, cost_objective)
            save_state(log_dir, generation, population, config, archive["history"])

    finally:
        writer.flush()
        writer.close()

    print(f"Search complete. Final Pareto archive: {os.path.join(log_dir, FINAL_ARCHIVE)}", flush=True)


if __name__ == "__main__":
    main()

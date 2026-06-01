"""
搜索空间定义模块

基于MambaVision-Tiny架构定义层级别的操作选择搜索空间。
基因序列长度为16，对应4个阶段，各阶段层数分别为 [1, 3, 8, 4]。
每层的候选操作类型为：
    C: CNN (ConvBlock)
    M: Mamba (MambaVisionMixer)
    A: Attention (自注意力块)
"""

import random
import time
from contextlib import nullcontext
from typing import List, Dict, Tuple

# MambaVision-Tiny 各阶段层数配置
STAGE_DEPTHS = [1, 3, 8, 4]
TOTAL_LAYERS = sum(STAGE_DEPTHS)  # 16

# 候选操作类型
OPS = ['C', 'M', 'A']

# 操作到名称的映射
OP_NAMES = {
    'C': 'ConvBlock',
    'M': 'MambaVisionMixer',
    'A': 'Attention',
}


def get_stage_indices() -> List[Tuple[int, int]]:
    """
    获取每个阶段在基因序列中的起始和结束索引（不包含结束索引）
    返回: [(0,1), (1,4), (4,12), (12,16)]
    """
    indices = []
    start = 0
    for depth in STAGE_DEPTHS:
        indices.append((start, start + depth))
        start += depth
    return indices


STAGE_INDICES = get_stage_indices()


def parse_genotype(genotype: str) -> Dict[int, List[str]]:
    """
    将基因序列解析为各阶段的操作列表

    Args:
        genotype: 长度为16的基因序列，如 'CCCCMMMMAAAAMMAA'

    Returns:
        dict: {stage_idx: [ops...]}
    """
    assert len(genotype) == TOTAL_LAYERS, f"基因序列长度必须为{TOTAL_LAYERS}"
    result = {}
    for stage_idx, (start, end) in enumerate(STAGE_INDICES):
        result[stage_idx] = list(genotype[start:end])
    return result


def genotype_from_stage_ops(stage_ops: Dict[int, List[str]]) -> str:
    """
    从各阶段操作列表构建基因序列

    Args:
        stage_ops: {stage_idx: [ops...]}

    Returns:
        str: 基因序列
    """
    genotype = []
    for stage_idx in range(len(STAGE_DEPTHS)):
        genotype.extend(stage_ops[stage_idx])
    return ''.join(genotype)


def random_genotype() -> str:
    """随机采样一个基因序列"""
    return ''.join(random.choice(OPS) for _ in range(TOTAL_LAYERS))


def get_max_subnet_genotype() -> str:
    """
    最大子网基因序列：全阶段使用Attention（计算量最大）
    """
    return 'A' * TOTAL_LAYERS


def get_mid_subnet_genotype() -> str:
    """
    中等子网基因序列：全阶段使用Mamba
    """
    return 'M' * TOTAL_LAYERS


def get_min_subnet_genotype() -> str:
    """
    最小子网基因序列：全阶段使用CNN（计算量最小）
    """
    return 'C' * TOTAL_LAYERS


def sample_subnet_genotype(num: int = 1, exclude: List[str] = None) -> List[str]:
    """
    随机采样子网基因序列

    Args:
        num: 采样数量
        exclude: 需要排除的基因序列列表

    Returns:
        list: 基因序列列表
    """
    exclude_set = set(exclude or [])
    results = []
    while len(results) < num:
        g = random_genotype()
        if g not in exclude_set:
            results.append(g)
            exclude_set.add(g)
    return results


def get_sandwich_subnet_genotypes(k: int = 5) -> List[Tuple[str, str]]:
    """
    基于Sandwich Rule获取K个子网基因序列

    Args:
        k: 每次迭代训练的子网总数

    Returns:
        list: [(tag, genotype), ...]
        tag格式: max, mid, min, random_0, random_1, ...
    """
    assert k >= 3, "Sandwich Rule要求K >= 3"
    max_g = get_max_subnet_genotype()
    mid_g = get_mid_subnet_genotype()
    min_g = get_min_subnet_genotype()

    random_num = k - 3
    random_gs = sample_subnet_genotype(num=random_num, exclude=[max_g, mid_g, min_g])

    results = [
        ('max', max_g),
        ('mid', mid_g),
        ('min', min_g),
    ]
    for i, g in enumerate(random_gs):
        results.append((f'random_{i}', g))

    return results


def genotype_to_transformer_blocks(genotype: str) -> Dict[int, List[int]]:
    """
    将基因序列转换为MambaVisionLayer需要的transformer_blocks格式

    MambaVisionLayer中，transformer_blocks列表中的层使用Attention，
    其余层使用Mamba。对于CNN阶段（stage 0,1），所有层都是ConvBlock。

    在超网中，我们需要更灵活的控制：根据基因序列决定每层使用什么操作。
    这里返回的transformer_blocks表示在该阶段中哪些层索引使用Attention。

    Args:
        genotype: 基因序列

    Returns:
        dict: {stage_idx: [layer_indices_using_attention]}
    """
    stage_ops = parse_genotype(genotype)
    transformer_blocks = {}
    for stage_idx, ops in stage_ops.items():
        # 只有非conv阶段（stage 2,3）才需要transformer_blocks
        # 但超网中所有阶段都可能混合操作，所以需要统一处理
        attn_indices = [i for i, op in enumerate(ops) if op == 'A']
        transformer_blocks[stage_idx] = attn_indices
    return transformer_blocks


def compute_genotype_throughput(
    model,
    genotype: str,
    batch_size: int = 128,
    input_resolution: Tuple[int, int, int] = (3, 224, 224),
    num_iterations: int = 100,
    num_warmup: int = 10,
    use_amp: bool = False,
    channels_last: bool = False,
) -> Dict[str, float]:
    """
    测量指定基因子网的推理吞吐量。

    方法参考 little_workspace/throughout_measure：使用随机输入，先 warmup，
    再通过 CUDA synchronize 统计固定迭代的平均推理时间。

    Returns:
        dict: {'avg_time': seconds / iter, 'throughput': images / second}
    """
    if not validate_genotype(genotype):
        raise ValueError(f"Invalid genotype: {genotype}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("compute_genotype_throughput requires PyTorch") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for throughput measurement")

    model.cuda().eval()
    inputs = torch.randn(batch_size, *input_resolution, device='cuda')
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)

    autocast = torch.cuda.amp.autocast if use_amp else nullcontext
    with torch.no_grad():
        for _ in range(num_warmup):
            with autocast():
                model(inputs, genotype)
        torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            with autocast():
                model(inputs, genotype)
        torch.cuda.synchronize()
        end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    result = {
        'avg_time': avg_time,
        'throughput': batch_size / avg_time,
    }
    torch.cuda.empty_cache()
    return result


def validate_genotype(genotype: str) -> bool:
    """验证基因序列是否合法"""
    if len(genotype) != TOTAL_LAYERS:
        return False
    if not all(op in OPS for op in genotype):
        return False
    return True


if __name__ == '__main__':
    # 简单测试
    print("Stage indices:", STAGE_INDICES)
    g = random_genotype()
    print("Random genotype:", g)
    print("Parsed:", parse_genotype(g))

    sandwich = get_sandwich_subnet_genotypes(k=5)
    for tag, sg in sandwich:
        print(f"{tag}: {sg}")

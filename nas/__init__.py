"""
NAS模块：基于MambaVision的神经网络架构搜索
"""

from .search_space import (
    STAGE_DEPTHS, TOTAL_LAYERS, OPS, OP_NAMES,
    parse_genotype, genotype_from_stage_ops,
    random_genotype, get_max_subnet_genotype, get_mid_subnet_genotype, get_min_subnet_genotype,
    sample_subnet_genotype, get_sandwich_subnet_genotypes,
    genotype_to_transformer_blocks, compute_genotype_flops_hint, validate_genotype
)

from .supernet import MambaVisionSuperNet, SuperBlock, SuperStage

__all__ = [
    'STAGE_DEPTHS', 'TOTAL_LAYERS', 'OPS', 'OP_NAMES',
    'parse_genotype', 'genotype_from_stage_ops',
    'random_genotype', 'get_max_subnet_genotype', 'get_mid_subnet_genotype', 'get_min_subnet_genotype',
    'sample_subnet_genotype', 'get_sandwich_subnet_genotypes',
    'genotype_to_transformer_blocks', 'compute_genotype_flops_hint', 'validate_genotype',
    'MambaVisionSuperNet', 'SuperBlock', 'SuperStage',
]

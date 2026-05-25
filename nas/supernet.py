"""
超网模型定义模块

基于MambaVision-Tiny构建单路径超网，每层包含三种候选操作（C/M/A），
根据基因序列（子网架构）选择对应的单路径执行。
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp

import sys
sys.path.insert(0, '/public/zhanghaojie/MambaVision/mambavision')

from models.mamba_vision import (
    PatchEmbed, Downsample, ConvBlock, MambaVisionMixer, Attention,
    window_partition, window_reverse
)

from nas.search_space import (
    STAGE_DEPTHS, STAGE_INDICES, parse_genotype, OPS, OP_NAMES
)


class SuperBlock(nn.Module):
    """
    超网中的基本块，包含三种候选操作：ConvBlock, MambaVisionMixer, Attention
    根据传入的op_type选择执行对应的操作
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 drop_path=0.,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 is_conv_stage=False,
                 ):
        super().__init__()
        self.dim = dim
        self.is_conv_stage = is_conv_stage
        self.window_size = window_size

        # 三种候选操作
        self.op_c = ConvBlock(
            dim=dim,
            drop_path=drop_path,
            layer_scale=None,
        )

        # Mamba和Attention需要LayerNorm，且输入是(B,L,C)格式
        self.norm_m = norm_layer(dim)
        self.op_m = MambaVisionMixer(
            d_model=dim,
            d_state=8,
            d_conv=3,
            expand=1,
        )

        self.norm_a = norm_layer(dim)
        self.op_a = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )

        # MLP和残差连接（Attention和Mamba共享）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else None
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else None
        self.use_layer_scale = use_layer_scale

    def forward(self, x, op_type):
        """
        Args:
            x: 输入特征
                - conv阶段: (B, C, H, W)
                - 非conv阶段且已partition: (B*num_windows, L, C), L=window_size*window_size
            op_type: 操作类型，'C', 'M', 或 'A'
        Returns:
            输出特征，与输入格式一致
        """
        if op_type == 'C':
            if x.dim() == 4:
                # conv阶段: 直接给ConvBlock处理
                x = self.op_c(x)
            else:
                # 非conv阶段: x是 (B*num_windows, L, C)
                # 需要reshape为 (B*num_windows, C, window_size, window_size)
                Bn, L, C = x.shape
                ws = int(L ** 0.5)
                x_4d = x.transpose(1, 2).reshape(Bn, C, ws, ws)
                x_4d = self.op_c(x_4d)
                x = x_4d.reshape(Bn, C, L).transpose(1, 2)
        elif op_type == 'M':
            # Mamba操作: 需要(B,L,C)格式
            if x.dim() == 4:
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
                if self.use_layer_scale:
                    x_flat = x_flat + self.drop_path(self.gamma_1 * self.op_m(self.norm_m(x_flat)))
                    x_flat = x_flat + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_flat)))
                else:
                    x_flat = x_flat + self.drop_path(self.op_m(self.norm_m(x_flat)))
                    x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
                x = x_flat.transpose(1, 2).reshape(B, C, H, W)
            else:
                if self.use_layer_scale:
                    x = x + self.drop_path(self.gamma_1 * self.op_m(self.norm_m(x)))
                    x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                else:
                    x = x + self.drop_path(self.op_m(self.norm_m(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif op_type == 'A':
            # Attention操作: 需要(B,L,C)格式
            if x.dim() == 4:
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
                if self.use_layer_scale:
                    x_flat = x_flat + self.drop_path(self.gamma_1 * self.op_a(self.norm_a(x_flat)))
                    x_flat = x_flat + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_flat)))
                else:
                    x_flat = x_flat + self.drop_path(self.op_a(self.norm_a(x_flat)))
                    x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
                x = x_flat.transpose(1, 2).reshape(B, C, H, W)
            else:
                if self.use_layer_scale:
                    x = x + self.drop_path(self.gamma_1 * self.op_a(self.norm_a(x)))
                    x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                else:
                    x = x + self.drop_path(self.op_a(self.norm_a(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            raise ValueError(f"Unknown op_type: {op_type}")
        return x


class SuperStage(nn.Module):
    """
    超网中的一个阶段，包含多个SuperBlock
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 drop_path_rates,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 layer_scale=None,
                 is_conv_stage=False,
                 ):
        super().__init__()
        self.is_conv_stage = is_conv_stage
        self.window_size = window_size
        self.depth = depth

        self.blocks = nn.ModuleList([
            SuperBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                drop_path=drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                layer_scale=layer_scale,
                is_conv_stage=is_conv_stage,
            )
            for i in range(depth)
        ])

        self.downsample = None if not downsample else Downsample(dim=dim)

    def forward(self, x, stage_ops):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            stage_ops: list of str, 该阶段每层的操作类型，如 ['C', 'M', 'A']
        Returns:
            输出特征
        """
        _, _, H, W = x.shape

        # 对于非conv阶段（stage 2,3），统一做window partition
        # 这样所有操作（C/M/A）都在window内执行
        need_partition = False
        if not self.is_conv_stage:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
            need_partition = True

        for i, blk in enumerate(self.blocks):
            op_type = stage_ops[i]
            x = blk(x, op_type)

        if need_partition:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

        if self.downsample:
            x = self.downsample(x)
        return x


class MambaVisionSuperNet(nn.Module):
    """
    MambaVision 超网模型
    """

    def __init__(self,
                 depths=None,
                 num_heads=None,
                 window_size=None,
                 dim=80,
                 in_dim=32,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 ):
        super().__init__()
        if depths is None:
            depths = STAGE_DEPTHS
        if num_heads is None:
            num_heads = [2, 4, 8, 16]
        if window_size is None:
            window_size = [8, 8, 14, 7]

        self.depths = depths
        self.num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            is_conv_stage = (i == 0 or i == 1)
            stage = SuperStage(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                drop_path_rates=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                layer_scale=layer_scale if not is_conv_stage else layer_scale_conv,
                is_conv_stage=is_conv_stage,
            )
            self.stages.append(stage)

        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x, genotype):
        """
        Args:
            x: 输入图像
            genotype: 基因序列，如 'CCCCMMMMAAAAMMAA'
        Returns:
            特征向量
        """
        stage_ops = parse_genotype(genotype)
        x = self.patch_embed(x)
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, stage_ops[stage_idx])
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, genotype):
        x = self.forward_features(x, genotype)
        x = self.head(x)
        return x

    def sample_and_forward(self, x, genotype):
        """兼容接口，直接调用forward"""
        return self.forward(x, genotype)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaVisionSuperNet(num_classes=1000).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    genotype = 'CCCCMMMMAAAAMMAA'
    out = model(x, genotype)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    from nas.search_space import get_sandwich_subnet_genotypes
    subnets = get_sandwich_subnet_genotypes(k=5)
    for tag, g in subnets:
        out = model(x, g)
        print(f"{tag}: {g} -> {out.shape}")

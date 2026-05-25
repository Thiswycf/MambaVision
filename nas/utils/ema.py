"""
指数移动平均（EMA）模块

用于超网训练时对模型权重进行EMA平滑
"""

import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA(nn.Module):
    """
    模型指数移动平均

    维护一个模型参数的EMA副本，在验证时使用EMA权重
    """

    def __init__(self, model, decay=0.9998):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        # EMA模型不参与梯度计算
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """更新EMA权重"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def state_dict(self):
        """返回EMA模型的state_dict"""
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """加载EMA模型的state_dict"""
        self.module.load_state_dict(state_dict, strict=strict)


def apply_ema_to_model(source_model, target_model, decay=0.9998):
    """
    将EMA权重应用到目标模型（用于验证）

    Args:
        source_model: EMA模型
        target_model: 需要加载EMA权重的模型
        decay: EMA衰减率
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)

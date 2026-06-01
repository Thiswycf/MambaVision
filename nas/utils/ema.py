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

    def _get_source_value(self, state_dict, name):
        """Return a matching tensor from plain or DDP-wrapped model state."""
        if name in state_dict:
            return state_dict[name]
        ddp_name = f'module.{name}'
        if ddp_name in state_dict:
            return state_dict[ddp_name]
        raise KeyError(f'Missing EMA source state for {name}')

    def _update(self, model, update_fn):
        source_state = model.state_dict()
        with torch.no_grad():
            for name, ema_value in self.module.state_dict().items():
                model_value = self._get_source_value(source_state, name).to(device=ema_value.device)
                ema_value.copy_(update_fn(ema_value, model_value))

    def update(self, model):
        """更新EMA权重和buffer（包括BatchNorm running stats）"""
        def ema_update(ema_value, model_value):
            if ema_value.is_floating_point():
                return ema_value * self.decay + model_value * (1. - self.decay)
            return model_value

        self._update(model, ema_update)

    def set(self, model):
        """直接同步完整模型状态到EMA模型。"""
        self._update(model, lambda _ema_value, model_value: model_value)

    def sync_buffers(self, model):
        """从模型同步所有buffer，保留EMA参数不变。"""
        source_state = model.state_dict()
        synced = 0
        with torch.no_grad():
            for name, ema_buffer in self.module.named_buffers():
                model_buffer = self._get_source_value(source_state, name).to(device=ema_buffer.device)
                ema_buffer.copy_(model_buffer)
                synced += 1
        return synced

    def sync_stale_buffers(self, model):
        """兼容旧checkpoint：如果EMA BN buffer仍是初始化值，则同步模型buffer。"""
        source_state = model.state_dict()
        stale = False
        for name, ema_buffer in self.module.named_buffers():
            if name.endswith('running_mean'):
                model_buffer = self._get_source_value(source_state, name).to(device=ema_buffer.device)
                if torch.count_nonzero(ema_buffer).item() == 0 and torch.count_nonzero(model_buffer).item() != 0:
                    stale = True
                    break
            elif name.endswith('running_var'):
                model_buffer = self._get_source_value(source_state, name).to(device=ema_buffer.device)
                if torch.all(ema_buffer == 1) and not torch.all(model_buffer == 1):
                    stale = True
                    break

        return self.sync_buffers(model) if stale else 0

    def state_dict(self):
        """返回EMA模型的state_dict"""
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """加载EMA模型的state_dict"""
        self.module.load_state_dict(state_dict, strict=strict)


def apply_ema_to_model(source_model, target_model, decay=0.9998):
    """
    将EMA完整状态应用到目标模型（用于验证）。

    Args:
        source_model: EMA模型
        target_model: 需要加载EMA状态的模型
        decay: 保留兼容旧调用，不参与复制
    """
    del decay

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    with torch.no_grad():
        for name, target_value in target_state.items():
            candidates = [name]
            if name.startswith('module.'):
                candidates.append(name[len('module.'):])
            else:
                candidates.append(f'module.{name}')

            for source_name in candidates:
                if source_name in source_state:
                    target_value.copy_(source_state[source_name].to(device=target_value.device))
                    break
            else:
                raise KeyError(f'Missing EMA source state for {name}')

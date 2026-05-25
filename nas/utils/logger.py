"""
TensorBoard日志记录模块

支持为每个子网路径使用独立的tag前缀记录训练和验证指标
"""

import os
from tensorboardX import SummaryWriter


class SuperNetTensorboardLogger(object):
    """
    超网训练TensorBoard日志记录器

    支持以下tag前缀格式：
        - max/train_loss, max/test_acc1
        - mid/train_loss, mid/test_acc1
        - min/train_loss, min/test_acc1
        - random_0/train_loss, random_0/test_acc1
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = {}

    def set_step(self, tag_prefix='default', step=None):
        if step is not None:
            self.step[tag_prefix] = step
        else:
            self.step[tag_prefix] = self.step.get(tag_prefix, 0) + 1

    def update(self, tag_prefix='default', step=None, **kwargs):
        """
        记录标量值

        Args:
            tag_prefix: 子网标识前缀，如 'max', 'mid', 'min', 'random_0'
            step: 全局步数或epoch
            **kwargs: 需要记录的指标名和值
        """
        current_step = step if step is not None else self.step.get(tag_prefix, 0)
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, float) or isinstance(v, int):
                self.writer.add_scalar(f"{tag_prefix}/{k}", v, current_step)
            elif hasattr(v, 'item'):
                self.writer.add_scalar(f"{tag_prefix}/{k}", v.item(), current_step)

    def update_scalar(self, tag, value, step):
        """直接记录一个标量值"""
        if hasattr(value, 'item'):
            value = value.item()
        self.writer.add_scalar(tag, value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

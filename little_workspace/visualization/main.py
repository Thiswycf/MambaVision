import os, sys
import types
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mambavision import create_model
from mambavision.models.mamba_vision import VISUALIZATION_PT_PATH
from PIL import Image
from timm.data.transforms_factory import create_transform
import json
import torch
import glob
import numpy as np
from matplotlib.gridspec import GridSpec

def dict_to_object(d, parent_key=None):
    if isinstance(d, dict):
        # 如果父键是 id2label 或 label2id，保持为字典
        if parent_key in ('id2label', 'label2id'):
            return {k: dict_to_object(v, k) for k, v in d.items()}
        return types.SimpleNamespace(**{k: dict_to_object(v, k) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_object(item, parent_key) for item in d]
    return d

model = create_model('mamba_vision_T', pretrained=True, model_path="mambavision/_saved_models/mambavision_tiny_1k.pth.tar")
with open('mambavision/_saved_models/mambavision_tiny_1k_config.json', 'r') as f:
    config = dict_to_object(json.load(f))

# eval mode for inference
model.cuda().eval()

# prepare image for the model
# url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('mambavision/assets/cat.jpg')
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions

transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=config.mean,
                             std=config.std,
                             crop_mode=config.crop_mode,
                             crop_pct=config.crop_pct)

inputs = transform(image).unsqueeze(0).cuda()
# model inference
outputs = model(inputs)
# logits = outputs['logits'] 
logits = outputs
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", config.id2label[str(predicted_class_idx)])

import matplotlib.pyplot as plt
# 获取所有pt文件
pt_files = sorted(glob.glob(os.path.join(VISUALIZATION_PT_PATH, "*.pt")))

if len(pt_files) > 0:
    # 读取所有特征图
    all_features = []
    for pt_file in pt_files:
        features = torch.load(pt_file, map_location='cpu')
        all_features.append((os.path.basename(pt_file), features))
    
    # 计算总的子图数量
    num_features = len(all_features)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 4 * num_features))
    gs = GridSpec(num_features, 8, figure=fig, hspace=0.3, wspace=0.3)
    
    for row, (filename, features) in enumerate(all_features):
        # 处理特征图，取第一个样本
        if isinstance(features, torch.Tensor):
            feat = features[0] if features.dim() > 3 else features
        else:
            feat = features
            
        # 转换为numpy并归一化
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        
        # 如果是4维 (C, H, W) 或 (B, C, H, W)，取通道维度
        if feat.ndim == 4:
            feat = feat[0]  # 取第一个batch
        if feat.ndim == 3:
            # 选择前8个通道进行可视化
            num_channels = min(8, feat.shape[0])
            for col in range(num_channels):
                ax = fig.add_subplot(gs[row, col])
                channel_feat = feat[col]
                # 归一化到0-1
                channel_feat = (channel_feat - channel_feat.min()) / (channel_feat.max() - channel_feat.min() + 1e-8)
                ax.imshow(channel_feat, cmap='viridis')
                ax.set_title(f'{filename}\nCh{col}', fontsize=8)
                ax.axis('off')
            
            # 如果通道不足8个，填充空白
            for col in range(num_channels, 8):
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
        else:
            # 对于2维特征，直接显示
            ax = fig.add_subplot(gs[row, :])
            ax.imshow(feat, cmap='viridis', aspect='auto')
            ax.set_title(filename, fontsize=10)
            ax.axis('off')
    
    # 保存合并后的可视化结果
    output_path = os.path.join(VISUALIZATION_PT_PATH, "merged_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"可视化结果已保存到: {output_path}")
else:
    print(f"在 {VISUALIZATION_PT_PATH} 中未找到pt文件")

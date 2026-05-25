import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mambavision import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('mamba_vision_T').to(device).eval()

x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    y = model(x)
print('forward ok:', tuple(y.shape))  # 输出: (1, 1000)
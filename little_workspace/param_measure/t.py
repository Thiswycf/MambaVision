import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mambavision import create_model
import time
import torch
from thop import profile

def get_params(model, inputs):
    _, params = profile(model, inputs=(inputs,), verbose=False)
    return params / 1e6

model = create_model('mamba_vision_T', pretrained=True, model_path="mambavision/_saved_models/mambavision_tiny_1k.pth.tar")

# eval mode for inference
model.cuda().eval()

data_shape = (32, 3, 224, 224)

print(f"Data shape: {data_shape}")
inputs = torch.randn(*data_shape).cuda()

params = get_params(model, inputs)
torch.cuda.synchronize()

print(f"Params: {params:.1f} M")
print("=" * 50)

torch.cuda.empty_cache()

'''
31.8 M
'''
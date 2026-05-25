import os, sys
import types
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mambavision import create_model
from PIL import Image
from timm.data.transforms_factory import create_transform
import json

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
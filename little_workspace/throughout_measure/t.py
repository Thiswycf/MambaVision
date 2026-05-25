import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mambavision import create_model
import time
import torch

model = create_model('mamba_vision_T', pretrained=True, model_path="mambavision/_saved_models/mambavision_tiny_1k.pth.tar")

# eval mode for inference
model.cuda().eval()

test_batches = [8, 16, 32, 64, 128, 256]

input_resolution = (3, 224, 224)
num_iterations = 100
num_warmup = 10

for batch_size in test_batches:
    print(f"Batch size: {batch_size}")
    inputs = torch.randn(batch_size, *input_resolution).cuda()

    for _ in range(num_warmup):
        model(inputs)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iterations):
        model(inputs)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    throughput = batch_size / avg_time
    print(f"Average inference time: {avg_time:.4f} s")
    print(f"Throughput: {throughput:.0f} Img/Sec")
    print("=" * 50)

    torch.cuda.empty_cache()

'''
Batch size: 8
Average inference time: 0.0072 s
Throughput: 1113 Img/Sec
==================================================
Batch size: 16
Average inference time: 0.0073 s
Throughput: 2201 Img/Sec
==================================================
Batch size: 32
Average inference time: 0.0133 s
Throughput: 2411 Img/Sec
==================================================
Batch size: 64
Average inference time: 0.0260 s
Throughput: 2460 Img/Sec
==================================================
Batch size: 128
Average inference time: 0.0531 s
Throughput: 2409 Img/Sec
==================================================
Batch size: 256
Average inference time: 0.1085 s
Throughput: 2359 Img/Sec
==================================================
'''
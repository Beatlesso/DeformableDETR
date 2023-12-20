import torch.utils.benchmark as benchmark
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from torch import Tensor
from models.ops.functions import MSDeformAttnFunction
from models.ops.functions import PyTorchDeformAttnFunction
'''
value: torch.Size([1, 20097, 8, 32])
input_spatial_shapes: torch.Size([4, 2])
all_input_spatial_shapes: tensor([[100, 151],
        [ 50,  76],
        [ 25,  38],
        [ 13,  19]], device='cuda:0')
input_level_start_index: torch.Size([4])
all_input_level_start_index: tensor([    0, 15100, 18900, 19850], device='cuda:0')
sampling_locations: torch.Size([1, 20097, 8, 4, 4, 2])
attention_weights: torch.Size([1, 20097, 8, 4, 4])
self.im2col_step: 64
'''


batch = 4
n_heads = 8
n_levels = 4
n_points = 4
d_model = 512
input_spatial_shapes = [[200, 200], [100,  100], [50,  50], [25,  25]]
dtype = torch.float32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(1)

    
sum = 0
input_level_start_index = []
for (x, y) in input_spatial_shapes:
    input_level_start_index.append(sum)
    sum += x * y


value = torch.randn(batch, sum, n_heads, d_model // n_heads, dtype=dtype, device=device)
input_spatial_shapes = torch.Tensor(input_spatial_shapes).type(torch.long).to(device)
input_level_start_index = torch.Tensor(input_level_start_index).type(torch.long).to(device)
sampling_locations = torch.randn(batch, sum, n_heads, n_levels, n_points, 2, dtype=dtype, device=device)
attention_weights = torch.randn(batch, sum, n_heads, n_levels, n_points, dtype=dtype, device=device)
im2col_step = 64


out1 = None
out2 = None

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    out1 = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, im2col_step)


# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# prof.export_chrome_trace("MSDeformAttnFunction.json")


prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")




with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    out2 = PyTorchDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, im2col_step)


# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# prof.export_chrome_trace("PyTorchDeformAttnFunction.json")




assert np.quantile(torch.abs(out1 - out2).cpu(), 0.99) < 5e-5
print("check ok!")

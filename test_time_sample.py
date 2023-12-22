import torch
from torchvision.models import resnet50
import time
epoch = 100
model = resnet50(pretrained=False)
device = torch.device('cuda:0')
x = torch.ones(1, 3, 640, 640).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
model.to(device)
model.eval()
for _ in range(20):
    output = model(x)
times = torch.zeros(epoch)  # 用来存放每次测试结果
randn_input = torch.randn(1, 3, 640, 640).to(device)
with torch.no_grad():
    for i in range(epoch):
        starter.record()
        output = model(randn_input)
        ender.record()
        # 异步转同步
        torch.cuda.synchronize()
        times[i] = starter.elapsed_time(ender)  # 单位是毫秒
mean_time = times.mean().item()
 
print("predict time : {:.6f}ms,FPS:{}".format(mean_time, 1000/mean_time))
import ray
import torch
from torch import nn

from models.stage_manager import stage_manager

class MicroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optim = torch.optim.Adam

    def forward(self, x, y):
        torch._dynamo.distributed_stage(1)
        x = x + 1
        if x < 0:
            x = x + 1
        else:
            x = x - 1

        torch._dynamo.distributed_stage(2)
        y = y + 1

        torch._dynamo.distributed_stage(3)
        z = x + y
        return z
    
    # def forward(self, x, y):
    #     with stage_manager(1):
    #         x = x + 1
    #     # print("graph break")
    #     with stage_manager(2):
    #         y = y + 1
    #     # print("graph break")
    #     with stage_manager(3, last=True):
    #         z = x + y
    #     return z
    
    # def forward(self, x, y):
    #     def stage1(x):
    #         x = x + 1
    #         return x
    #     def stage2(y):
    #         y = y + 1
    #         return y
    #     def stage3(x, y):
    #         z = x + y
    #         return z
    #     out1 = stage1(x)
    #     out2 = stage2(x)
    #     out3 = stage3(out1, out2)
    #     return out3


model = MicroModel()
compiled_model = torch.compile(model, distribute=True)
compiled_model._set_optimizer(torch.optim.Adam)
compiled_model._set_stage_dependencies([(1,3), (2,3)])

import time

x = torch.Tensor([1])
y = torch.Tensor([2])

out = compiled_model(x, y)
print(out)

out = compiled_model(x, y)
print(out)

# # warmup
# for i in range(5):
#     compiled_model(x)
#     model(x)

# # time
# start_compiled = time.perf_counter()
# for i in range(10):
#     compiled_model(x)
# end_compiled = time.perf_counter()

# start_original = time.perf_counter()
# for i in range(10):
#     model(x)
# end_original = time.perf_counter()

# print(f"Compiled time: {(end_compiled-start_compiled)*1000000:.2f}")
# print(f"Original time: {(end_original-start_original)*1000000:.2f}")
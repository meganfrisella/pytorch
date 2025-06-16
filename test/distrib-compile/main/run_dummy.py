import ray
import torch
from torch import nn

from models.stage_manager import stage_manager

class MicroModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.c = torch.ones((out_dim))
        self.layers = nn.ModuleList([
            nn.Linear(inp_dim, out_dim),
            nn.Linear(inp_dim, out_dim)])
        self.norm = nn.RMSNorm((out_dim))

    def forward(self, x, y, dynamo_mb: int=0):
        torch._dynamo.distributed_stage(1, mb=dynamo_mb, optim=torch.optim.Adam)
        # if x > 0:
        #     x = x + 1
        # else: 
        #     x = x - 1
        x = self.layers[0](x)
        # x = torch.matmul(x, x)

        torch._dynamo.distributed_stage(2, mb=dynamo_mb, optim=torch.optim.Adam)
        # y = y + 1
        y = self.layers[1](y)
        # y = torch.matmul(y, y)

        torch._dynamo.distributed_stage(3, mb=dynamo_mb, optim=torch.optim.Adam)
        # z = x + y
        z = x + y + self.c
        return z


inp_dim = 32
out_dim = 1
model = MicroModel(inp_dim, out_dim)
compiled = torch.compile(model, distribute=True)


import time

# backprop

batch_size = 10
x = torch.randn((batch_size, inp_dim))
y = torch.randn((batch_size, inp_dim))
z = torch.randn((batch_size, 1))

out = compiled(x, y)
stg1 = compiled._ray_actors[1]
stg2 = compiled._ray_actors[2]
stg3 = compiled._ray_actors[3]

out = compiled(x, y)
grad3_x, grad3_y = stg3.backward.options(num_returns=2).remote(0, out.get_ref(), truth=z, loss_fn=torch.nn.CrossEntropyLoss())
grad2 = stg2.backward.remote(0, grad3_y)
grad1 = stg1.backward.remote(0, grad3_x)

upd3 = stg3.update.remote(grad3_x, grad3_y)
upd2 = stg2.update.remote(grad2)
upd1 = stg1.update.remote(grad1)
ray.get([upd3, upd2, upd1])



# sanity 

# batch_size = 10
# x = torch.randn((batch_size, inp_dim))
# y = torch.randn((batch_size, inp_dim))
# x = torch.Tensor([1])
# y = torch.Tensor([2])

# out = model(x, y)
# print(out)

# out = compiled_model(x, y)
# print(out.get())

# x = torch.Tensor([0])

# out = compiled_model(x, y)
# print(out.get())

# x = torch.Tensor([2])

# out = compiled_model(x, y)
# print(out.get())

# x = torch.Tensor([-1])

# out = compiled_model(x, y)
# print(out.get())




# microbenchmark

# @ray.remote
# class DummyActor:
#     def mul(self, x):
#         return torch.matmul(x, x)
#     def sum_mean(self, x, y):
#         return torch.mean(x + y)
# actor1 = DummyActor.remote()
# actor2 = DummyActor.remote()
# actor3 = DummyActor.remote()

# p = 12
# n = 2**p
# x = torch.randn((n, n))
# y = torch.Tensor((n, n))

# for i in range(2):
#     x_ref = actor1.mul.remote(x)
#     y_ref = actor2.mul.remote(y)
#     print(ray.get(actor3.sum_mean.remote(x_ref, y_ref)))
#     print(model(x, y))
#     print(compiled_model(x, y).get())
# iters = 10

# start_compiled = time.perf_counter()
# for i in range(iters):
#     compiled_model(x, y).get()
# end_compiled = time.perf_counter()

# start_original = time.perf_counter()
# for i in range(iters):
#     model(x, y)
# end_original = time.perf_counter()

# start_sanity = time.perf_counter()
# for i in range(iters):
#     ray.get(actor3.sum_mean.remote(actor1.mul.remote(x), actor2.mul.remote(y)))
# end_sanity = time.perf_counter()

# print(f"{n}, {(end_compiled-start_compiled)*1000000/iters:.2f}, {(end_original-start_original)*1000000/iters:.2f}, {(end_sanity-start_sanity)*1000000/iters:.2f}")
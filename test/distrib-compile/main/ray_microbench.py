import ray
import torch


@ray.remote
class Actor:
    def tensor(self, x):
        return torch.zeros(100, 100, int(x/10000))
        # return torch.zeros(x)


a = Actor.remote()

for x in range(0, 100, 20):
    ray.get(a.tensor.remote(x))


import time

for x in range(0, 500_000_000, 50_000_000):
    if x == 0: continue
    start = time.perf_counter()
    for _ in range(10):
        ray.get(a.tensor.remote(x))
    end = time.perf_counter()
    print(f"size {x}: {(end-start)/10*1000:.2f} ms")

import torch

def f(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 1
    return x

f = torch.compile(f, distribute=True)

f(1)
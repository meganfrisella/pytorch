import torch
from torch import nn, optim
import time

print(torch.__path__)

# HINT: top level module
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hints = []
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        _ = "partition"
        x = self.fc2(x)
        _ = "partition"
        x = self.fc3(x)
        return x


input_size, hidden_size, output_size = 10, 5, 1
x_train = torch.randn(10, input_size)
y_train = torch.randn(10, output_size)

model = SimpleNN(input_size, hidden_size, output_size)
compiled = torch.compile(model)


# check equivalence between original and compiled model

out1 = model(x_train)
out2 = compiled(x_train)
assert torch.all(torch.eq(out1, out2))
print("tensors are equal!")


# get the pipeline stages from the compiled model

partitions = [f for f in compiled.__dir__() if 'rayjit' in f]
for f in partitions:
    code = getattr(compiled, f)
    print(code)
    # this doesn't work: cannot `exec` a code object that requires params
    print(exec(code, {"self": compiled, "x": x_train}))


# training loop

"""
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    outputs = compiled.forward(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""
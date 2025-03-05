import torch
from torch import nn, optim
import time

print(torch.__path__)

def partition(*args):
    pass

# HINT: top level module
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hints = []
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.loss = nn.MSELoss()
    
    def forward(self, x):
        x = self.fc1(x)
        partition(x)
        x = self.fc2(x)
        partition(x)
        x = self.fc3(x)
        return x

batch_size, input_size, hidden_size, output_size = 12, 10, 5, 1
torch.manual_seed(0)
x_train = torch.randn(batch_size, input_size)
y_train = torch.randn(batch_size, output_size)

model = SimpleNN(input_size, hidden_size, output_size)
compiled = torch.compile(model)

criterion = nn.MSELoss()

# check equivalence between original and compiled model

x_train.requires_grad_()
out1 = model(x_train)
out2 = compiled(x_train)
assert torch.all(torch.eq(out1, out2))

loss = criterion(out1, y_train)
loss.backward()

print("torch x gradient:")
print(x_train.grad)

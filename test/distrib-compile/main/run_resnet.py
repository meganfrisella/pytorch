from models.resnet import resnet18
from torch._dynamo.backends.debugging import eager

resnet = resnet18().to('cuda')
resnet = torch.compile(resnet, distribute=True, backend=eager)

x = torch.tensor(10, 100).to('cuda')

out = resnet(x)
print(out.get().shape())
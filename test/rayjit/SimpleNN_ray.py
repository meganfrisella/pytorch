
import ray.dag
import torch

@ray.remote
class SimpleNN_Stage_1_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_Stage_1

  def forward(self, x):
    # TODO: what is the activation if there is more than one argument? 
    x.requires_grad_()
    self.prev_activation = x
    self.pred = self.model.forward_stage_1(x)
    return self.pred.clone().detach()

  def backward(self, grad):
    self.pred.backward(grad)
    return self.prev_activation.grad

import ray.dag
import torch

@ray.remote
class SimpleNN_Stage_2_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_Stage_2

  def forward(self, x):
    # TODO: what is the activation if there is more than one argument? 
    x.requires_grad_()
    self.prev_activation = x
    self.pred = self.model.forward_stage_2(x)
    return self.pred.clone().detach()

  def backward(self, grad):
    self.pred.backward(grad)
    return self.prev_activation.grad

import ray.dag
import torch

@ray.remote
class SimpleNN_Stage_3_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_Stage_3

  def forward(self, x):
    # TODO: what is the activation if there is more than one argument? 
    x.requires_grad_()
    self.prev_activation = x
    self.pred = self.model.forward_stage_3(x)
    return self.pred

  def backward(self, pred, truth):
    loss = self.model.loss(pred, truth)
    loss.backward()
    return self.prev_activation.grad


SimpleNN_Stage_1_actor = SimpleNN_Stage_1_Actor.remote()
SimpleNN_Stage_2_actor = SimpleNN_Stage_2_Actor.remote()
SimpleNN_Stage_3_actor = SimpleNN_Stage_3_Actor.remote()
workers = [SimpleNN_Stage_1_actor, SimpleNN_Stage_2_actor, SimpleNN_Stage_3_actor]

import scheduling
dag = scheduling.gpipe(workers, num_microbatches=4)
dag = dag.experimental_compile()

#scheduling.execute_dag(dag, num_microbatches=4)

batch_size, input_size, output_size = 12, 10, 1
torch.manual_seed(0)
x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)
out = dag.execute(x, y)

print("ray x gradient:")
print(ray.get(out))

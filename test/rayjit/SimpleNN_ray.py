
import ray.dag
import torch

@ray.remote
class SimpleNN_Stage_1_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_Stage_1
  
  def forward_stage_1(self, x):
    # TODO: what is the activation if there is more than one argument? 
    x.requires_grad_()
    self.prev_activation = x
    self.pred = self.model.forward_stage_1(x)
    return self.pred

  def backward_stage_1(self, grad):
    self.pred.backward(grad)
    return self.prev_activation.grad

import ray.dag
import torch

@ray.remote
class SimpleNN_Stage_2_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_Stage_2
  
  def forward_stage_2(self, x):
    # TODO: what is the activation if there is more than one argument? 
    x.requires_grad_()
    self.prev_activation = x
    self.pred = self.model.forward_stage_2(x)
    return self.pred

  def backward_stage_2(self, grad):
    self.pred.backward(grad)
    return self.prev_activation.grad

import ray.dag
import torch

@ray.remote
class SimpleNN_Stage_3_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_Stage_3
  
  def forward_stage_3(self, x):
    # TODO: what is the activation if there is more than one argument? 
    x.requires_grad_()
    self.prev_activation = x
    self.pred = self.model.forward_stage_3(x)
    return self.pred

  def backward_stage_3(self, pred, truth):
    loss = self.model.loss(pred, truth)
    loss.backward()
    return self.prev_activation.grad
SimpleNN_Stage_1_actor = SimpleNN_Stage_1_Actor.remote()
SimpleNN_Stage_2_actor = SimpleNN_Stage_2_Actor.remote()
SimpleNN_Stage_3_actor = SimpleNN_Stage_3_Actor.remote()


with ray.dag.InputNode() as inp:
   dag = SimpleNN_Stage_1_actor.forward_stage_1.bind(inp[0])
   dag = SimpleNN_Stage_2_actor.forward_stage_2.bind(dag)
   dag = SimpleNN_Stage_3_actor.forward_stage_3.bind(dag)
   dag = SimpleNN_Stage_3_actor.backward_stage_3.bind(dag, inp[1])
   dag = SimpleNN_Stage_2_actor.backward_stage_2.bind(dag)
   dag = SimpleNN_Stage_1_actor.backward_stage_1.bind(dag)

dag = dag.experimental_compile()
batch_size, input_size, output_size = 10, 10, 1
torch.manual_seed(0)
x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)
out = dag.execute(x, y)
print("ray x gradient: ", out.get())

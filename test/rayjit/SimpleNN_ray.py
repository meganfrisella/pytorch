
import ray.dag
import torch

@ray.remote
class SimpleNN_forward_Stage_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_forward_Stage
  
  def forward(self, x):
    self.pred = self.model.forward(x)
    return self.pred


import ray.dag
import torch

@ray.remote
class SimpleNN_torch_dynamo_resume_in_forward_at_21_Stage_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_torch_dynamo_resume_in_forward_at_21_Stage
  
  def torch_dynamo_resume_in_forward_at_21(self, x):
    self.pred = self.model.torch_dynamo_resume_in_forward_at_21(x)
    return self.pred


import ray.dag
import torch

@ray.remote
class SimpleNN_torch_dynamo_resume_in_forward_at_23_Stage_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_torch_dynamo_resume_in_forward_at_23_Stage
  
  def torch_dynamo_resume_in_forward_at_23(self, x):
    self.pred = self.model.torch_dynamo_resume_in_forward_at_23(x)
    return self.pred

actor_SimpleNN_forward_Stage = SimpleNN_forward_Stage_Actor.remote()
actor_SimpleNN_torch_dynamo_resume_in_forward_at_21_Stage = SimpleNN_torch_dynamo_resume_in_forward_at_21_Stage_Actor.remote()
actor_SimpleNN_torch_dynamo_resume_in_forward_at_23_Stage = SimpleNN_torch_dynamo_resume_in_forward_at_23_Stage_Actor.remote()


with ray.dag.InputNode() as inp:
   dag = actor_SimpleNN_forward_Stage.forward.bind(inp)
   dag = actor_SimpleNN_torch_dynamo_resume_in_forward_at_21_Stage.torch_dynamo_resume_in_forward_at_21.bind(dag)
   dag = actor_SimpleNN_torch_dynamo_resume_in_forward_at_23_Stage.torch_dynamo_resume_in_forward_at_23.bind(dag)
   # TODO: dag = actor3.backward.bind(dag)
   # TODO: dag = actor2.backward.bind(dag)
   # TODO: dag = actor1.backward.bind(dag)

dag = dag.experimental_compile()
batch_size, input_size = 10, 10
torch.manual_seed(0)
x = torch.randn(batch_size, input_size)
out = dag.execute(x)
print("compiled ray: ", out.get())

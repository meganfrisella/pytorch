
import ray.dag

@ray.remote
class SimpleNN_forward_Stage_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_forward_Stage
  
  def forward(self, x):
    self.pred = self.model.forward(x)
    return self.pred


import ray.dag

@ray.remote
class SimpleNN_torch_dynamo_resume_in_forward_at_20_Stage_Actor:
  def __init__(self):
    self.model = compiled.SimpleNN_torch_dynamo_resume_in_forward_at_20_Stage
  
  def torch_dynamo_resume_in_forward_at_20(self, x):
    self.pred = self.model.torch_dynamo_resume_in_forward_at_20(x)
    return self.pred

actor_SimpleNN_forward_Stage = SimpleNN_forward_Stage_Actor.remote()
actor_SimpleNN_torch_dynamo_resume_in_forward_at_20_Stage = SimpleNN_torch_dynamo_resume_in_forward_at_20_Stage_Actor.remote()


with ray.dag.InputNode() as inp:
   dag = actor_SimpleNN_forward_Stage.forward.bind(inp)
   dag = actor_SimpleNN_torch_dynamo_resume_in_forward_at_20_Stage.torch_dynamo_resume_in_forward_at_20.bind(inp)
   # TODO: dag = actor2.backward.bind(dag)
   # TODO: dag = actor1.backward.bind(dag)

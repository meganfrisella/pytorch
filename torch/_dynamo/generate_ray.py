import torch
import types
import inspect

from pathlib import Path
import sys

top_level_path = Path(sys.argv[0]).resolve()


def generate_stage_and_ray_actor(mod: torch.nn.Module, code: types.CodeType):

  ## GENERATE STAGE

  # get the name/args of the top level module
  mod_name = mod.__class__.__name__
  # mod_args = [name for name, _ in inspect.signature(mod.__init__).parameters.items()]
  # mod_args_str = ", ".join(mod_args)

  # get the name/args of the stage's foward method
  fwd_name = code.co_name
  fwd_args = list(code.co_varnames[1:code.co_argcount])
  fwd_args = ", ".join(fwd_args)
  stage_name = f"{mod_name}_{fwd_name}_Stage"

  # turn the code object into a callable function
  callable_code = types.FunctionType(code, {})

  # the stage has all the attributes of the top level module (for now)
  # TODO: the new module should only get attributes necessary for this stage
  attrs = {}
  for name, val in mod.__dict__.items():
    if (not callable(val) and 
        not name.startswith("__") and
        not name == "_modules"):
      attrs[name] = val
  
  for name, val in mod.__dict__["_modules"].items():
    attrs[name] = val

  # additional attributes
  attrs[fwd_name] = callable_code
  attrs["fwd_name"] = fwd_name
  attrs["name"] = stage_name

  # create a new module for this stage
  stage_class = type(stage_name, (), attrs)
  stage = stage_class()

  # add the stage as an attribute of the original module so we know where to find it
  setattr(mod, f"{mod_name}_{fwd_name}_Stage", stage)

  
  ## GENERATE RAY ACTOR

  # output ray code in the same dir as the top-level python file
  # ASSUMPTION: the top-level module is defined in the top-level python file
  out_path = top_level_path.parent / f"{mod_name}_ray.py"
  mod_filename = top_level_path.stem

  # create a ray actor that wraps the new stage
  ray_code = f"""
import ray.dag
import torch

@ray.remote
class {stage_name}_Actor:
  def __init__(self):
    self.model = compiled.{stage_name}
  
  def {fwd_name}(self, {fwd_args}):
    self.pred = self.model.{fwd_name}({fwd_args})
    return self.pred

"""
  # append the actor to the output file
  with out_path.open("a") as f:
    f.write(ray_code)
  
  return stage


def generate_schedule(mod_name: str, stages: list[torch.nn.Module]):
  out_path = top_level_path.parent / f"{mod_name}_ray.py"

  num_stages = len(stages)
  assert(num_stages == 3) # only handles 3-stage pipelines for now

  ray_code = [f"actor_{stage.name} = {stage.name}_Actor.remote()" for stage in stages]

  stage_1 = stages[0]
  stage_2 = stages[1]
  stage_3 = stages[2]

  ray_code.append(f"""\n
with ray.dag.InputNode() as inp:
   dag = actor_{stage_1.name}.{stage_1.fwd_name}.bind(inp)
   dag = actor_{stage_2.name}.{stage_2.fwd_name}.bind(dag)
   dag = actor_{stage_3.name}.{stage_3.fwd_name}.bind(dag)
   # TODO: dag = actor3.backward.bind(dag)
   # TODO: dag = actor2.backward.bind(dag)
   # TODO: dag = actor1.backward.bind(dag)

dag = dag.experimental_compile()
batch_size, input_size = 10, 10
torch.manual_seed(0)
x = torch.randn(batch_size, input_size)
out = dag.execute(x)
print("compiled ray: ", out.get())
""")

  ray_code = '\n'.join(ray_code)

  with out_path.open("a") as f:
    f.write(ray_code)
  
  return out_path
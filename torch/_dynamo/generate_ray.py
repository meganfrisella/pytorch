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

  # the stage has all the attributes of the top level module (for now)
  # TODO: the new module should only get attributes necessary for this stage
  attrs = {}
  for name, val in mod.__dict__.items():
    if not callable(val) and not name.startswith("__"):
      attrs[name] = val

  # the stage's forward method
  attrs[fwd_name] = code

  # create a new module for this stage
  stage_name = f"{mod_name}_{fwd_name}_Stage"
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
  ray_code = f"""\n
import ray
import torch
from {mod_filename} import {mod_name}

@ray.remote
class {stage_name}_Actor:
  def __init__(self):
    self.model = {mod_name}.{stage_name}
  
  def forward(self, {fwd_args}):
    self.pred = self.model.{fwd_name}({fwd_args})
    return self.pred
"""
  # append the actor to the output file
  with out_path.open("a") as f:
    f.write(ray_code)
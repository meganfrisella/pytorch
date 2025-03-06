import torch
import types
import inspect

from pathlib import Path
import sys

top_level_path = Path(sys.argv[0]).resolve()


def generate_stage_and_ray_actor(
    mod: torch.nn.Module, stage_num: int, last_stage: bool, frame_attrs: list[str], code: types.CodeType
):
    ## GENERATE STAGE

    # get the name/args of the top level module
    mod_name = mod.__class__.__name__
    # mod_args = [name for name, _ in inspect.signature(mod.__init__).parameters.items()]
    # mod_args_str = ", ".join(mod_args)

    # get the name/args of the stage's foward method
    fwd_args = list(code.co_varnames[1 : code.co_argcount])
    assert len(fwd_args) == 1 and "only handles passing one tensor between pipeline stages"
    fwd_args = ", ".join(fwd_args)

    # names for the stage and its methods
    stage_name = f"{mod_name}_Stage_{stage_num}"
    fwd_name = f"forward_stage_{stage_num}"
    bwd_name = f"backward_stage_{stage_num}"

    # turn the code object into a callable function
    callable_code = types.FunctionType(code, {})

    # all attrs of the top-level module that are accessed by this stage
    attrs = {}
    for attr in frame_attrs:
        if hasattr(mod.__dict__, attr):
            attrs[attr] = mod.__dict__[attr]
        elif attr in mod.__dict__["_modules"]:
            attrs[attr] = mod.__dict__["_modules"][attr]
        else:
            assert False and "Can't find attribute in module!"
    
    # final stage needs loss function
    # ASSUMPTION: the top-level module has a 'loss' attribute which is a valid loss function
    if last_stage: 
        attrs["loss"] = mod.__dict__["_modules"]["loss"]

    # additional attributes
    attrs[fwd_name] = callable_code
    attrs["fwd_name"] = fwd_name
    attrs["bwd_name"] = bwd_name
    attrs["name"] = stage_name

    # create a new module for this stage
    stage_class = type(stage_name, (), attrs)
    stage = stage_class()

    # add the stage as an attribute of the original module so we know where to find it
    setattr(mod, stage_name, stage)

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
"""

    if last_stage:
        fwd_code = f"""
  def forward(self, {fwd_args}):
    # TODO: what is the activation if there is more than one argument? 
    {fwd_args}.requires_grad_()
    self.prev_activation = {fwd_args}
    self.pred = self.model.{fwd_name}({fwd_args})
    return self.pred
"""
    else:
        fwd_code = f"""
  def forward(self, {fwd_args}):
    # TODO: what is the activation if there is more than one argument? 
    {fwd_args}.requires_grad_()
    self.prev_activation = {fwd_args}
    self.pred = self.model.{fwd_name}({fwd_args})
    return self.pred.clone().detach()
"""
        
    if last_stage:
      bwd_code = f"""
  def backward(self, pred, truth):
    loss = self.model.loss(pred, truth)
    loss.backward(retain_graph=True)
    return self.prev_activation.grad
"""
    elif stage_num == 1:
      bwd_code = f"""
  def backward(self, grad):
    self.pred.backward(grad, retain_graph=True)
    #print(self.prev_activation.grad)
    return self.prev_activation.grad
"""
    else:
      bwd_code = f"""
  def backward(self, grad):
    self.pred.backward(grad, retain_graph=True)
    return self.prev_activation.grad
"""

    ray_code += fwd_code
    ray_code += bwd_code

    # append the actor to the output file
    with out_path.open("a") as f:
        f.write(ray_code)

    return stage


def generate_schedule(mod_name: str, stages: list[torch.nn.Module]):
    out_path = top_level_path.parent / f"{mod_name}_ray.py"

    num_stages = len(stages)

    ray_code = ["\n"]
    ray_code.extend([f"{stage.name}_actor = {stage.name}_Actor.remote()" for stage in stages])
    actor_names = ", ".join([f"{stage.name}_actor" for stage in stages])
    ray_code.append(f"workers = [{actor_names}]")

    ray_code.append(
        f"""
import scheduling
dag = scheduling.build_gpipe_dag(workers)

#dag.visualize()

torch.manual_seed(0)
batch_size, input_size, output_size = 12, 10, 1
x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)

scheduling.execute_dag(dag, x, y, batch_size)
"""
    )

    ray_code = "\n".join(ray_code)

    with out_path.open("a") as f:
        f.write(ray_code)

    return out_path
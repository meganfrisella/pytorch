import torch
import ray
from torch.piper.utils import RemoteTensor, serialize_graphmodule
from .registry import register_debug_backend as register_backend

@register_backend
def piper(gm, args, **kwargs):
    example_inputs, graphargs, better_compile_id = args
    dynamo_tls = torch._dynamo.eval_frame.dynamo_tls
    
    # make sure example inputs are serializable by turning symbolic
    #  ints and fake tensors into concrete values
    serializable_example_inputs = []
    for ex in example_inputs:
        if isinstance(ex, torch.SymInt):
            serializable_example_inputs.append(int(ex))
        elif isinstance(ex, torch._subclasses.fake_tensor.FakeTensor):
            new = torch.full(
                ex.shape,
                0,
                dtype=ex.dtype,
                device=ex.device,
                layout=ex.layout,
                requires_grad=ex.requires_grad,
            )
            serializable_example_inputs.append(new)
        else:
            serializable_example_inputs.append(ex)

    # get all the graphargs that are model parameters in order and send those
    # parameters to the actor. accumulate a new list of graphargs (all the
    # original graphargs minus those which are model parameters)

    # collect model attributes to store them on the actor, update the graph
    # arguments to remove the model attributes
    parameters = []
    new_graphargs = []
    for arg in graphargs:
        if "self" in str(arg):
            if isinstance(arg.example, torch.SymInt):
                parameters.append(int(arg.example))
            else:
                parameters.append(arg.example)
        else:
            parameters.append(None)
            new_graphargs.append(arg)
            
    assert len(new_graphargs) == len(list(filter(lambda a: a is None, parameters)))

    # serialize the fx.Graph
    payload = serialize_graphmodule(gm)

    # send the fx.Graph and model attributes to the actor
    stage_id = dynamo_tls.current_stage
    actor_id = dynamo_tls.current_actor
    actor = dynamo_tls.torch_module._ray_actors[actor_id]
    compile_id = better_compile_id
    ray.get(
        actor.compile_graph.remote(
            compile_id,
            stage_id,
            payload,
            torch._dynamo.backends.debugging.eager,
            serializable_example_inputs,
            parameters,
        )
    )

    # get a list of fake tensor outputs from the fx.Graph
    def symint_to_int(x):
        return int(x) if isinstance(x, torch.SymInt) else x
    def int_to_tensor(x):
        return torch.tensor(x) if isinstance(x, int) else x
    fakes = gm(*list(map(symint_to_int, example_inputs)))
    fakes = list(map(int_to_tensor, fakes))

    # return a wrapper function that runs the fx.Graph on the actor and 
    # returns remote futures for each graph output
    def overwrite_compiled_fn(*args):
        mb_idx = torch._dynamo.eval_frame.dynamo_tls.current_mb
        # track stage dependencies
        for arg in args:
            if isinstance(arg, RemoteTensor):
                torch._dynamo.eval_frame.dynamo_tls.torch_module._dag.add((arg.get_stage_id(), stage_id))

        # get Ray ObjectRefs from RemoteTensors
        def unwrap(x):
            return x.get_ref() if isinstance(x, RemoteTensor) else x
        args = list(map(unwrap, args))

        if torch._dynamo.eval_frame.dynamo_tls.currently_compiling:
            # dispatch task without nccl transport
            refs = actor.call_cpu.options(num_returns=len(fakes)).remote(compile_id, stage_id, mb_idx, *args)
        else:
            # dispatch with nccl transport
            refs = actor.call.options(num_returns=len(fakes)).remote(compile_id, stage_id, mb_idx, *args)

        # wrap the remote futures with RemoteTensor
        if isinstance(refs, list):
            assert len(fakes) == len(refs)
            return [RemoteTensor(fake, ref, stage_id) for fake, ref in zip(fakes, refs)]
        else:
            assert len(fakes) == 1
            return [RemoteTensor(fakes[0], refs, stage_id)]
    return overwrite_compiled_fn, new_graphargs
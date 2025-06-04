import ray
import torch
@ray.remote

class StageActor:
    def __init__(self, id, compiler_fn, example_inputs, parameters):
        print(f"Initializing Ray actor {id}")
        self.id = id
        self.compiler_fn = compiler_fn
        self.example_inputs = example_inputs
        self.parameters = parameters

    def compile_graph(self, filename, data):
        with open(f"/tmp/{filename}", "wb") as f:
            f.write(data)
        self.gm = torch.load(f"/tmp/{filename}", weights_only=False)

        compiled_fn = self.compiler_fn(self.gm, self.example_inputs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fn = compiled_fn
        return "Finished compiling"

    def call(self, *args):
        new_args = list(args)
        all_args = []
        for arg in self.parameters:
            if arg is not None:
                all_args.append(arg)
            else:
                all_args.append(new_args[0])
                del new_args[0]
        return self.compiled_fn(*all_args)
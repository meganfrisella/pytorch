import ray
import torch
from torch import nn, optim
from models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B
from torch.profiler import profile, record_function, ProfilerActivity
from .llama_schedules import build_1f1b_schedule, build_gpipe_schedule, print_schedule
from torch._dynamo.scheduling import Task, DAGEdge, execute_schedule
import time

# ray.shutdown()
ray.init(include_dashboard=True, namespace="llama")
# torch.manual_seed(0)

COMPARE_RAY_BASELINE = 0

llama_config = LLAMA_3B
loss_fn = torch.nn.CrossEntropyLoss()
controller_device = 'cuda:0'

batch_size = 64
num_mbs = 4
seq_len = 32
warmup = 10
iters = 100

x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len)).to(controller_device)
y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long).to(controller_device)

print("loading model")

model = Transformer(llama_config)
# model = model.half()
model.to(controller_device)

print("loaded model")

print("compiling model")

from torch._dynamo.backends.debugging import eager
compiled = torch.compile(model, distribute=True)
compiled(x, dynamo_mb=42).get()
torch._dynamo.eval_frame.dynamo_tls.currently_compiling = None
stg1 = compiled._ray_actors[0]
stg2 = compiled._ray_actors[1]

ray.get(stg1.send_input.remote(x))
ray.get(stg2.send_truth.remote(y))

print("compiled model")

# del model.tok_embeddings
# del model.layers
# del model.norm
# del model.output
# del model.freqs_cis


# time microbatch schedules

# "1F1B" schedule

# manual schedule
def iter_1f1b_manual():
    done_stg1 = []
    done_stg2 = []

    out_ref = compiled(None, dynamo_mb=0)
    for mb in range(1, num_mbs):
        grad2, done2 = stg2.backward.options(num_returns=2).remote(mb-1, out_ref.get_ref(), loss_fn=loss_fn)
        out_ref = compiled(None, dynamo_mb=mb)
        _, done1 = stg1.backward.options(num_returns=2).remote(mb-1, grad2)
        done_stg1.append(done1)
        done_stg2.append(done2)

    grad2, done2 = stg2.backward.options(num_returns=2).remote(num_mbs-1, out_ref.get_ref(), loss_fn=loss_fn)
    _, done1 = stg1.backward.options(num_returns=2).remote(num_mbs-1, grad2)
    done_stg1.append(done1)
    done_stg2.append(done2)

    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd1, upd2])
    # ray.wait([upd1, upd2], fetch_local=False)

# build high-level schedule
schedule = build_1f1b_schedule(num_mbs, 2)

# move bubble to ensure correct scheduling with fused forwards
schedule[0][2] = schedule[0][1]
schedule[0][1] = None

def iter_1f1b():
    out = execute_schedule(compiled, schedule, [None], None, loss_fn)
    ray.get(out)
    # ray.wait(out, fetch_local=False)

# warmup
for _ in range(warmup):
    iter_1f1b()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_1f1b()
end = time.perf_counter()

print(
    f"1F1B execute_schedule throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
)
print(
    f"time: {(end - start)*1000000/iters:.0f} us"
)

time.sleep(1)
# warmup
for _ in range(warmup):
    iter_1f1b_manual()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_1f1b_manual()
end = time.perf_counter()

print(
    f"1F1B manual schedule throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
)
print(
    f"time: {(end - start)*1000000/iters:.0f} us"
)


# "GPIPE" schedule

# manual schedule
def iter_gpipe_manual():
    fwd_refs = []
    done_stg1 = []
    done_stg2 = []

    for mb in range(num_mbs):
        out_ref = compiled(None, dynamo_mb=mb)
        fwd_refs.append(out_ref)

    for mb, out_ref in enumerate(fwd_refs):
        grad2, done2 = stg2.backward.options(num_returns=2).remote(mb, out_ref.get_ref(), loss_fn=loss_fn)
        grad1, done1 = stg1.backward.options(num_returns=2).remote(mb, grad2)
        done_stg1.append(done1)
        done_stg2.append(done2)

    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd1, upd2])
    # ray.wait([upd1, upd2], fetch_local=False)

# build high-level schedule
schedule = build_gpipe_schedule(num_mbs, 2)

def iter_gpipe():
    out = execute_schedule(compiled, schedule, [None], None, loss_fn)
    ray.get(out)
    # ray.wait(out, fetch_local=False)

print("gpipe auto")
# warmup
for _ in range(warmup):
    iter_gpipe()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_gpipe()
end = time.perf_counter()

print(
    f"GPipe execute_schedule throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
)
print(
    f"time: {(end - start)*1000000/iters:.0f} us"
)

# time.sleep(1)
print("gpipe manual")
# warmup
for _ in range(warmup):
    iter_gpipe_manual()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_gpipe_manual()
end = time.perf_counter()

print(
    f"GPipe manual schedule throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
)
print(
    f"time: {(end - start)*1000000/iters:.0f} us"
)


# baseline: no distribution

from models.llama_baseline import Transformer as BaseTransformer
from torch._dynamo.backends.debugging import eager

baseline = BaseTransformer(x, llama_config)
baseline = baseline.half()
baseline.to('cuda')
print("loaded model")
baseline = torch.compile(baseline, distribute=False, backend=eager)
optim = torch.optim.Adam(baseline.parameters())

def iter_baseline():
    out = baseline(x)
    loss = loss_fn(out, y)
    loss.backward()
    optim.step()
    optim.zero_grad()

for _ in range(warmup):
    iter_baseline()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_baseline()
end = time.perf_counter()

print(
    f"Baseline throughput: {(iters * batch_size * seq_len)/(end - start):.0f} tokens/sec"
)
print(
    f"time: {(end - start)*1000000/iters:.0f} us"
)

if COMPARE_RAY_BASELINE:
    # baseline: Ray manual pipeline implementation - 1F1B

    from models.llama_actor import LlamaActor

    stg1_actor = LlamaActor.options(num_gpus=1).remote(llama_config, batch_size, seq_len, 0, num_mbs, 2)
    stg2_actor = LlamaActor.options(num_gpus=1).remote(llama_config, batch_size, seq_len, 1, num_mbs, 2)
    actors = [stg1_actor, stg2_actor]

    # create nccl group and wait for it to be ready
    from ray.experimental.collective import create_collective_group
    create_collective_group(actors, backend="nccl")
    ray.get([actor.init_training.remote() for actor in actors])
    ray.get(stg1_actor.send_input.remote(x))
    ray.get(stg2_actor.send_truth.remote(y))

    for _ in range(warmup):
        out1 = stg1_actor.forward.remote(0, None)
        out2 = stg2_actor.forward.remote(0, out1)
        ray.get(out2)



    def iter_ray_baseline():
        done_stg1 = []
        done_stg2 = []

        for mb in range(num_mbs):
            out1 = stg1_actor.forward.remote(mb, None)
            out2 = stg2_actor.forward.remote(mb, out1)
            grad2 = stg2_actor.backward.remote(mb, out2, y)
            grad1 = stg1_actor.backward.remote(mb, grad2)
            done_stg2.append(grad2)
            done_stg1.append(grad1)

        upd2 = stg2_actor.update.remote(*done_stg2)
        upd1 = stg1_actor.update.remote(*done_stg1)
        ray.get([upd2, upd1])

    # warmup
    for _ in range(warmup):
        iter_ray_baseline()

    # time
    start = time.perf_counter()
    for _ in range(iters):
        iter_ray_baseline()
    end = time.perf_counter()

    print(
        f"Ray 1F1B baseline throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
    )

    # baseline: Ray manual pipeline implementation - GPipe

    from models.llama_actor import LlamaActor

    def iter_ray_baseline():
        fwd_refs = []
        done_stg1 = []
        done_stg2 = []
        
        for mb in range(num_mbs):
            out1 = stg1_actor.forward.remote(mb, None)
            out2 = stg2_actor.forward.remote(mb, out1)
            fwd_refs.append(out2)

        for mb, fwd_ref in enumerate(fwd_refs):
            grad2, done2 = stg2_actor.backward.options(num_returns=2).remote(mb, fwd_ref)
            grad1, done1 = stg1_actor.backward.options(num_returns=2).remote(mb, grad2)
            done_stg2.append(done2)
            done_stg1.append(done1)

        upd2 = stg2_actor.update.remote(*done_stg2)
        upd1 = stg1_actor.update.remote(*done_stg1)
        ray.get([upd2, upd1])

    # warmup
    for _ in range(warmup):
        iter_ray_baseline()

    # time
    start = time.perf_counter()
    for _ in range(iters):
        iter_ray_baseline()
    end = time.perf_counter()

    print(
        f"Ray GPipe baseline throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
    )
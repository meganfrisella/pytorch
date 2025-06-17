import ray
import torch
from torch import nn, optim
from models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B

COMPARE_RAY_BASELINE = 0

llama_config = LLAMA_DEBUG

loss_fn = torch.nn.CrossEntropyLoss()

model = Transformer(llama_config)
compiled = torch.compile(model, distribute=True)

batch_size = 100
seq_len = 32
x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long)

# time microbatch schedules

from .llama_schedules import build_1f1b_schedule, build_gpipe_schedule
from torch._dynamo.scheduling import DAGEdge, execute_schedule
import time

warmup = 3
iters = 10
num_mbs = 4

compiled(x, dynamo_mb=42).get()
stg1 = compiled._ray_actors[0]
stg2 = compiled._ray_actors[1]


# "1F1B" schedule


# manual implementation
def iter_1f1b():
    done_stg1 = []
    done_stg2 = []

    for mb in range(num_mbs):
        out_ref = compiled(x, dynamo_mb=mb)
        print(f"Calling backward stage 1 mb {mb}")
        grad2 = stg2.backward.remote(mb, out_ref.get_ref(), truth=y, loss_fn=loss_fn)
        print(f"Calling backward stage 0 mb {mb}")
        grad1 = stg1.backward.remote(mb, grad2)
        done_stg1.append(grad1)
        done_stg2.append(grad2)

    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd2, upd1])

g
# build high-level schedule
schedule = build_1f1b_schedule(num_mbs, 2)
dag_edges = [DAGEdge(0, 1), DAGEdge(1, 2)]


def iter_1f1b():
    out = execute_schedule(compiled, schedule, dag_edges, [x], y, loss_fn)
    ray.get(out)


# warmup
for _ in range(warmup):
    iter_1f1b()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_1f1b()
end = time.perf_counter()

print(
    f"1F1B throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
)


# "GPIPE" schedule

from torch._dynamo.scheduling import Task, DAGEdge, execute_schedule


# manual implementation
def iter_gpipe():
    fwd_refs = []
    done_stg1 = []
    done_stg2 = []

    for mb in range(num_mbs):
        out_ref = compiled(x, dynamo_mb=mb)
        fwd_refs.append(out_ref)

    for mb, out_ref in enumerate(fwd_refs):
        print(f"Calling backward stage 1 mb {mb}")
        grad2 = stg2.backward.remote(mb, out_ref.get_ref(), truth=y, loss_fn=loss_fn)
        print(f"Calling backward stage 0 mb {mb}")
        grad1 = stg1.backward.remote(mb, grad2)
        done_stg1.append(grad1)
        done_stg2.append(grad2)

    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd2, upd1])


# build high-level schedule
schedule = build_gpipe_schedule(num_mbs, 2)
dag_edges = [DAGEdge(0, 1), DAGEdge(1, 2)]


def iter_gpipe():
    out = execute_schedule(compiled, schedule, dag_edges, [x], y, loss_fn)
    ray.get(out)


# warmup
for _ in range(warmup):
    iter_gpipe()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_gpipe()
end = time.perf_counter()

print(
    f"GPipe throughput: {(iters * batch_size * num_mbs * seq_len)/(end - start):.0f} tokens/sec"
)


# baseline: no distribution

from models.llama_baseline import Transformer as BaseTransformer

baseline = BaseTransformer(llama_config)
baseline = torch.compile(baseline, distribute=False)
optim = torch.optim.Adam(baseline.parameters())


def iter_baseline():
    out = baseline(x)
    loss = loss_fn(out, y)
    loss.backward()
    optim.step()
    optim.zero_grad()


# warmup
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


if COMPARE_RAY_BASELINE:
    # baseline: Ray manual pipeline implementation - 1F1B

    from models.llama_actor import LlamaActor

    stg1_actor = LlamaActor.remote(LLAMA_DEBUG, batch_size, seq_len, 0, num_mbs, 2)
    stg2_actor = LlamaActor.remote(LLAMA_DEBUG, batch_size, seq_len, 1, num_mbs, 2)

    def iter_ray_baseline():
        done_stg1 = []
        done_stg2 = []

        for mb in range(num_mbs):
            out1 = stg1_actor.forward.remote(mb, x)
            out2 = stg2_actor.forward.remote(mb, out1)
            grad2 = stg2_actor.backward.remote(mb, out2, y)
            grad1 = stg1_actor.backward.remote(mb, grad2)
            done_stg2.append(grad2)
            done_stg1.append(grad1)

        upd2 = stg2_actor.update.remote(mb, *done_stg2)
        upd1 = stg1_actor.update.remote(mb, *done_stg1)
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
            out1 = stg1_actor.forward.remote(mb, x)
            out2 = stg2_actor.forward.remote(mb, out1)
            fwd_refs.append(out2)

        for mb, fwd_ref in enumerate(fwd_refs):
            grad2 = stg2_actor.backward.remote(mb, fwd_ref, y)
            grad1 = stg1_actor.backward.remote(mb, grad2)
            done_stg2.append(grad2)
            done_stg1.append(grad1)

        upd2 = stg2_actor.update.remote(mb, *done_stg2)
        upd1 = stg1_actor.update.remote(mb, *done_stg1)
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

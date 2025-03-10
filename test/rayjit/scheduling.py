import torch
import ray.dag

def build_gpipe_dag(workers, num_microbatches=4):
    """
    Constructs and compiles a GPipe DAG for pipeline parallelism.

    Args:
        workers (list): List of Ray actor handles, each representing a pipeline stage.
        num_microbatches (int): Number of microbatches to pipeline per batch.

    Returns:
        A compiled DAG ready for execution.
    """
    x_idx = 0
    y_idx = 1

    num_workers = len(workers)
    # Scheduling queues

    fwd_queues = [[] for _ in range(num_workers)]
    bwd_queues = [[] for _ in range(num_workers)]
    done = []

    with ray.dag.InputNode() as inp:
      # Load the queue for worker 0 with all input micro-batches
      for idx in range(num_microbatches):
        fwd_queues[0].append([idx, inp[num_microbatches * x_idx + idx]])

      while len(done) < num_microbatches:
        for k, worker in enumerate(workers):
            if fwd_queues[k]:
                idx, mb = fwd_queues[k].pop(0)
                mb = worker.forward.bind(mb)
                if k < num_workers - 1:
                    fwd_queues[k + 1].append([idx, mb])
                else:  
                    bwd_queues[k].append([idx, mb])
            elif bwd_queues[k]:
                idx, mb = bwd_queues[k].pop()
                if k == num_workers - 1:
                  mb = worker.backward.bind(mb, inp[num_microbatches * y_idx + idx])
                else:
                  mb = worker.backward.bind(mb)
                if k > 0:                    
                    bwd_queues[k - 1].append([idx, mb])
                else:
                    done.append(mb)

      dag = ray.dag.MultiOutputNode(done)

    # Compile dag built from pipeline and return
    return dag.experimental_compile()

def build_1f1b_dag(workers, num_microbatches=4, num_lead_microbatches=4):
    """
    Constructs and compiles a 1F1B DAG for pipeline parallelism.

    Args:
        workers (list): List of Ray actor handles, each representing a pipeline stage.
        num_microbatches (int): Number of microbatches to pipeline per batch.
        num_lead_microbatches (int): Number of leading microbatches to maintain pipeline balance.

    Returns:
        A compiled DAG ready for execution.
    """
    x_idx = 0
    y_idx = 1

    num_workers = len(workers)

    # Scheduling queues
    fwd_queues = [[] for _ in range(num_workers)]
    bwd_queues = [[] for _ in range(num_workers)]
    
    # Once a worker's counter reaches 0, it cannot execute another fwd until it
    # executes a bwd first.
    fwd_counter = [num_lead_microbatches - p for p in range(num_workers)]
    
    done = []

    with ray.dag.InputNode() as inp:
      # Load the queue for worker 0 with all input micro-batches
      for idx in range(num_microbatches):
        fwd_queues[0].append([idx, inp[num_microbatches * x_idx + idx]])

      while len(done) < num_microbatches:
        for k, worker in enumerate(workers):
            if fwd_counter[k] > 0 and fwd_queues[k]:
                idx, mb = fwd_queues[k].pop(0)
                mb = worker.forward.bind(mb)
                if k < num_workers - 1:
                    fwd_queues[k + 1].append([idx, mb])
                else:  
                    bwd_queues[k].append([idx, mb])
                fwd_counter[k] -= 1
            elif bwd_queues[k]:
                idx, mb = bwd_queues[k].pop()
                if k == num_workers - 1:
                  mb = worker.backward.bind(mb, inp[num_microbatches * y_idx + idx])
                else:
                  mb = worker.backward.bind(mb)
                if k > 0:                    
                    bwd_queues[k - 1].append([idx, mb])
                else:
                    done.append(mb)
                fwd_counter[k] += 1

      dag = ray.dag.MultiOutputNode(done)

    # Compile dag built from pipeline and return
    return dag.experimental_compile()

def no_microbatching(workers):
  num_workers = len(workers)
  with ray.dag.InputNode() as inp:
    x, y = inp[0], inp[1]
    for i, worker in enumerate(workers):
      if i == 0:
        dag = worker.forward.bind(x)
      else:
        dag = worker.forward.bind(dag)
    for i, worker in reversed(list(enumerate(workers))):
      if i == num_workers - 1:
        dag = worker.backward.bind(dag, y)
      else:
        dag = worker.backward.bind(dag)
  return dag

# def execute_dag(dag, batch_size=12, input_size=10, output_size=1, num_microbatches=4):
def execute_dag(dag, x, y, batch_size, num_microbatches=4):

  # torch.manual_seed(0)
  # x = torch.randn(batch_size, input_size)
  # y = torch.randn(batch_size, output_size)

  microbatch_size = batch_size // num_microbatches
  x_mbs = list(torch.split(x, microbatch_size))
  y_mbs = list(torch.split(y, microbatch_size))
  x_mbs.extend(y_mbs)

  out = dag.execute(*x_mbs)

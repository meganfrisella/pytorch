import torch
import ray.dag

def gpipe(workers, num_microbatches=4):
  """
  Generate a GPipe schedule for a set of Ray actors
  Assumption: Each actor in `workers` has methods `forward` and `backward` 
              which obey signatures as used below. The workers are in
              ascending order according to stage. 
  """
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

def execute_dag(dag, batch_size=12, input_size=10, output_size=1, num_microbatches=4):

  torch.manual_seed(0)
  x = torch.randn(batch_size, input_size)
  y = torch.randn(batch_size, output_size)

  # microbatch_size = batch_size // num_microbatches
  # x_mbs = torch.split(x, microbatch_size)
  # y_mbs = torch.split(y, microbatch_size)

  out = dag.execute(x, y)
  print("ray x gradient:")
  print(ray.get(out))

from models.moe import MoE
import torch

from torch._dynamo.backends.debugging import eager

moe = MoE(
    dim = 512,
    num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
    hidden_dim = 512 * 4,           # size of hidden dimension in each expert, defaults to 4 * dimension
    second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
    second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
    second_threshold_train = 0.2,
    second_threshold_eval = 0.2,
    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
    loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
).to('cuda')

moe = torch.compile(moe, distribute=True, backend=eager)

inputs = torch.randn(4, 1024, 512).to('cuda')

out, aux_loss = moe(inputs)
print(out.get().shape)
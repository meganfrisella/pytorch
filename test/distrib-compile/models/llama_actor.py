import ray
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .llama_baseline import Transformer

@ray.remote
class LlamaActor:
    @dataclass
    class BatchParameter:
        # criterion: torch.nn.CrossEntropyLoss
        # optimizer: torch.optim.AdamW
        logits_as_input: Optional[torch.Tensor] = None
        logits_as_output: Optional[torch.Tensor] = None

    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank: int,
        num_batches: int,
        num_partitions: int,
    ):
        self.device = "cuda:0"

        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.num_batches = num_batches
        self.num_partitions = num_partitions

        self.input = None
        self.truth = None

        # manual pipeline
        layers_per_rank = model_args.n_layers // num_partitions
        model_args.n_layers = layers_per_rank
        model = Transformer(model_args)
        # model = torch.compile(model, distribute=False)
        if rank == 0:
            model.norm = None
            model.output = None
        else:
            model.tok_embeddings = None
        model.to(self.device)
        self.model = model

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters())

        # microbatch metadata
        self.bparams: List[LlamaActor.BatchParameter] = []
        for i in range(num_batches):
            torch.manual_seed(2025 + i)
            bparam = self.BatchParameter()
            self.bparams.append(bparam)

    def init_training(self):
        for bparam in self.bparams:
            bparam.logits_as_input = None
            bparam.logits_as_output = None
        return "done"

    def send_input(self, tensor):
        self.input = tensor
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor
        return "done"

    @ray.method(tensor_transport="nccl")
    def forward(
        self, idx: int, logits_as_input: torch.Tensor
    ) -> torch.Tensor:

        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        if self.rank == 0:
            assert self.input is not None
            logits_as_output = self.model.forward(self.input)
            bparam.logits_as_output = logits_as_output
        else:
            logits_as_input = logits_as_input.to(self.device).requires_grad_() # do we need this with nccl transport? 
            bparam.logits_as_input = logits_as_input
            logits_as_output = self.model.forward(logits_as_input)
            bparam.logits_as_output = logits_as_output
        assert logits_as_output is not None
        return logits_as_output.detach()

    def backward_first(self, idx: int, logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]
        
        assert bparam.logits_as_input.requires_grad
        loss = self.criterion(bparam.logits_as_output, truth)
        loss.backward()
        grad = bparam.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_intra(self, idx: int, prev_grad: torch.Tensor) -> None:
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        assert prev_grad is not None
        prev_grad = prev_grad.to(self.device)

        bparam.logits_as_output.backward(prev_grad)
        grad = bparam.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_last(self, idx: int, prev_grad: torch.Tensor) -> None:
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        assert prev_grad is not None
        prev_grad = prev_grad.to(self.device)

        bparam.logits_as_output.backward(prev_grad)
        
        # return None
        return prev_grad # return a tensor

    @ray.method(tensor_transport="nccl")
    def backward(self, idx: int, data: torch.Tensor) -> Optional[torch.Tensor]:
        if self.rank == self.num_partitions - 1:
            assert self.truth is not None
            output = self.backward_first(idx, data, self.truth)
        elif self.rank == 0:
            output = self.backward_last(idx, data)
        else:
            output = self.backward_intra(idx, data)
        assert output is not None
        return output, output

    @ray.method(tensor_transport="nccl")
    def update(self, *grads) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()
        # return "done"
        assert grads[0] is not None
        return grads[0] # return a tensor
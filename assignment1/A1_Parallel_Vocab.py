import torch
import torch.nn as nn
from typing import Optional


class ParallelVocabEmbedding(nn.Module):
    """
    Parallel Vocab Embedding
        vocab_size = v
        world_size = w
        rank       = r
        local_vocab_size = n = v // w
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        rank: int,
        world_size: int,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, {world_size - 1}], got {rank}")
        if vocab_size % world_size != 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by world_size "
                f"({world_size}) in ParallelVocabEmbedding."
            )

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.rank = rank
        self.world_size = world_size
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed

        self.local_vocab_size = vocab_size // world_size
        self.global_start = rank * self.local_vocab_size  
        self.global_end = (rank + 1) * self.local_vocab_size  

        self.weight = nn.Parameter(
            torch.empty(
                self.local_vocab_size,
                self.emb_size,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:

        gen_device = self.weight.device if self.weight.is_cuda else "cpu"
        g = torch.Generator(device=gen_device)
        g.manual_seed(self.init_base_seed + self.rank)

        with torch.no_grad():
            self.weight.normal_(mean=self.init_mean, std=self.init_std, generator=g)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError(
                f"ParallelVocabEmbedding expects input_ids dtype torch.long, "
                f"got {input_ids.dtype}"
            )

        if input_ids.dim() != 2:
            raise ValueError(
                f"ParallelVocabEmbedding expects input_ids of shape [b, s], "
                f"but got shape {tuple(input_ids.shape)}"
            )

        bsz, seqlen = input_ids.shape

        out_dtype = self.weight.dtype
        out_device = input_ids.device

        output = torch.zeros(
            bsz,
            seqlen,
            self.emb_size,
            device=out_device,
            dtype=out_dtype,
        )

        start = self.global_start
        end = self.global_end


        mask_local = (input_ids >= start) & (input_ids < end)
        if not mask_local.any():
            return output

        local_ids = input_ids[mask_local] - start 
        weight = self.weight.to(device=out_device)
        local_embs = weight[local_ids]  


        output[mask_local] = local_embs

        return output

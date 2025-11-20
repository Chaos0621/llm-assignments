import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from assignment2.A2_T1 import DenseMLPWithLoRA, MLPActivationType

class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class SparseMLPWithLoRA(nn.Module):
    """
    Args:
        hidden_size: Hidden size h of the input and output, shape [b, s, h].
        ff_hidden_size: Feedforward hidden size ffh of the dense MLP before expert splitting.
        num_experts: Total number of experts ne.
        top_k: Number of experts k to route for each token.
        rank: Rank index in the process group, in [0, world_size - 1].
        world_size: World size w of the process group.
        activation_type: Activation type used in each expert MLP.
        init_mean: Mean of the normal distribution for gating weight initialization.
        init_std: Std of the normal distribution for gating weight initialization.
        init_base_seed: Base random seed for experts and gating weight; gating uses it directly without offset, experts add expert index offset.
        lora_rank: LoRA rank r; if 0, LoRA is disabled.
        lora_alpha: LoRA scaling factor α; if None, defaults to lora_rank.
        lora_dropout_rate: Dropout rate p applied on the LoRA output.
        lora_dropout_seed: Base random seed for LoRA dropout; experts add expert index offset.
        lora_init_base_seed: Base random seed for LoRA parameters; experts add expert index offset.
        device: Device for learnable parameters.
        dtype: Data type for expert parameters; gating weights are always float32 regardless of this.
    """

    def __init__(
        self,
        hidden_size: int,
        ff_hidden_size: int,
        num_experts: int,
        top_k: int,
        rank: int,
        world_size: int,
        activation_type: "MLPActivationType",
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 0,
        lora_init_base_seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, {world_size - 1}], got {rank}")
        if ff_hidden_size % num_experts != 0:
            raise ValueError(
                f"ff_hidden_size ({ff_hidden_size}) must be divisible by num_experts ({num_experts})"
            )
        if num_experts % world_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
            )

        self.hidden_size = hidden_size
        self.ff_hidden_size = ff_hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.rank = rank
        self.world_size = world_size
        self.activation_type = activation_type
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed
        self.lora_rank = int(lora_rank)
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = float(lora_dropout_rate)
        self.lora_dropout_seed = int(lora_dropout_seed)
        self.lora_init_base_seed = int(lora_init_base_seed)

        self.local_num_experts = self.num_experts // self.world_size
        self.global_expert_start = self.rank * self.local_num_experts
        self.global_expert_end = (self.rank + 1) * self.local_num_experts
        self.per_expert_ff_hidden_size = self.ff_hidden_size // self.num_experts

        self.gate_weight = nn.Parameter(
            torch.empty(
                hidden_size,
                num_experts,
                device=device,
                dtype=torch.float32,
            )
        )

        experts = []
        for global_idx in range(self.global_expert_start, self.global_expert_end):
            expert_init_base_seed = self.init_base_seed + global_idx
            expert_lora_init_base_seed = self.lora_init_base_seed + global_idx
            expert_lora_dropout_seed = self.lora_dropout_seed + global_idx
            expert = DenseMLPWithLoRA(
                hidden_size=hidden_size,
                ff_hidden_size=self.per_expert_ff_hidden_size,
                activation_type=self.activation_type,
                init_base_seed=expert_init_base_seed,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout_rate=self.lora_dropout_rate,
                lora_dropout_seed=expert_lora_dropout_seed,
                lora_init_base_seed=expert_lora_init_base_seed,
                device=device,
                dtype=dtype,
            )
            experts.append(expert)

        self.experts = nn.ModuleList(experts)

        self.reset_parameters()

    def reset_parameters(self) -> None:

        gen_device = self.gate_weight.device if self.gate_weight.is_cuda else "cpu"
        g = torch.Generator(device=gen_device)
        g.manual_seed(self.init_base_seed)
        with torch.no_grad():
            nn.init.normal_(self.gate_weight, mean=self.init_mean, std=self.init_std, generator=g)

        for expert in self.experts:
            expert.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size] containing
            the sparse MoE MLP output with LoRA adapters, where each token's
            output is the weighted sum of the outputs from the routed local experts,
            or an all-zero vector if no routed expert is local.
        """
        if x.dim() != 3:
            raise ValueError(
                f"SparseMLPWithLoRA expects input of shape [b, s, h], got {tuple(x.shape)}"
            )
        bsz, seqlen, h = x.shape
        if h != self.hidden_size:
            raise ValueError(
                f"Last dim of input ({h}) must equal hidden_size ({self.hidden_size})."
            )

        orig_dtype = x.dtype
        orig_device = x.device

        x_fp32 = x.to(torch.float32)
        x_flat = x_fp32.view(-1, h)
        logits = x_flat @ self.gate_weight
        probs = F.softmax(logits, dim=-1)

        top_k = min(self.top_k, self.num_experts)
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        denom = topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights_topk = topk_vals / denom

        out_flat = torch.zeros(
            x_flat.size(0),
            h,
            device=orig_device,
            dtype=orig_dtype,
        )

        for local_idx, expert in enumerate(self.experts):
            global_idx = self.global_expert_start + local_idx
            mask_for_expert = topk_idx.eq(global_idx)
            token_mask = mask_for_expert.any(dim=-1)
            if not token_mask.any():
                continue

            weight_for_expert = (weights_topk * mask_for_expert).sum(dim=-1)
            x_tokens = x[token_mask].view(-1, 1, h)
            expert_out = expert(x_tokens).view(-1, h)
            expert_out = expert_out.to(orig_device, orig_dtype)
            expert_weights = weight_for_expert[token_mask].to(expert_out.dtype).unsqueeze(-1)
            expert_out = expert_out * expert_weights
            out_flat[token_mask] += expert_out

        out = out_flat.view(bsz, seqlen, h)
        out = out.to(dtype=orig_dtype, device=orig_device)
        return out

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GroupRMSNorm(nn.Module):
    """
    Group RMS Normalization
    """

    def __init__(
        self,
        hidden_size: int,
        group_size: int,
        eps: float = 1e-6,
        init_range: Tuple[float, float] = (-1.0, 1.0),
        init_seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size}) in GroupRMSNorm."
            )

        self.hidden_size = hidden_size
        self.group_size = group_size
        self.eps = eps
        self.init_range = init_range
        self.init_seed = init_seed

        self.weight = nn.Parameter(
            torch.empty(hidden_size, device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:

        gen_device = self.weight.device if self.weight.is_cuda else "cpu"
        g = torch.Generator(device=gen_device)
        g.manual_seed(self.init_seed)

        a, b = self.init_range
        with torch.no_grad():
            self.weight.uniform_(a, b, generator=g)

    def forward(self, x: torch.Tensor, gz: Optional[int] = None) -> torch.Tensor:
 
        orig_dtype = x.dtype
        orig_device = x.device

        group_size = gz if gz is not None else self.group_size

        if x.dim() != 3:
            raise ValueError(
                f"GroupRMSNorm expects input of shape [b, s, h], "
                f"but got shape {tuple(x.shape)}"
            )

        bsz, seqlen, h = x.shape
        if h != self.hidden_size:
            raise ValueError(
                f"Last dim of input ({h}) must equal hidden_size ({self.hidden_size})."
            )

        if h % group_size != 0:
            raise ValueError(
                f"hidden_size ({h}) must be divisible by group_size ({group_size})."
            )

        num_groups = h // group_size

        x_fp32 = x.to(torch.float32)

        x_g = x_fp32.view(bsz, seqlen, num_groups, group_size)


        rms_sq = x_g.pow(2).mean(dim=-1, keepdim=True) 
        inv_rms = torch.rsqrt(rms_sq + self.eps)
        x_norm_g = x_g * inv_rms 

        x_norm = x_norm_g.view(bsz, seqlen, h)

        weight = self.weight.to(dtype=torch.float32, device=orig_device)
        y_fp32 = x_norm * weight.view(1, 1, h)

        y = y_fp32.to(dtype=orig_dtype, device=orig_device)
        return y

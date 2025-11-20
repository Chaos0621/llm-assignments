# src/modeling/pos_emb.py

import math
import torch
import torch.nn as nn
from typing import Optional

from src.functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """
    NTK-aware Rotary Position Embedding, LLaMA-style with dynamic NTK scaling.
        - head_dim = hd
        - max_seq_len = ms
        - base 
        - k (NTK scaling ratio)
        - dynamic:

    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        k: float = 1.0,
        dynamic: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE."
            )

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.head_dim = head_dim
        self.max_seq_len_training = max_seq_len
        self.base = float(base)
        self.dynamic = dynamic

        self.k = float(k)
        if self.k < 1.0:

            self.k = 1.0

        init_extended = int(round(self.max_seq_len_training * self.k))
        if init_extended < self.max_seq_len_training:
            init_extended = self.max_seq_len_training

        cos, sin = self._build_cos_sin_cache(
            seq_len=init_extended,
            k=self.k,
            device=device if device is not None else torch.device("cpu"),
            dtype=dtype if dtype is not None else torch.float32,
        )


        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.max_seq_len_cached = init_extended  


    def _build_cos_sin_cache(
        self,
        seq_len: int,
        k: float,
        device: torch.device,
        dtype: torch.dtype,
    ):

        hd = self.head_dim

        positions = torch.arange(
            seq_len, device=device, dtype=torch.float32
        ) 
        dim_idx = torch.arange(
            0, hd, 2, device=device, dtype=torch.float32
        ) 

        inv_freq = 1.0 / (self.base ** (dim_idx / hd))

        if k != 1.0:
            inv_freq = inv_freq * (k ** (dim_idx / (hd - 2)))

        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)

        cos = torch.cos(emb).to(dtype=dtype)
        sin = torch.sin(emb).to(dtype=dtype)

        return cos, sin

    def _maybe_extend_cache(self, seq_len: int, x_device, x_dtype):

        if seq_len <= self.max_seq_len_cached:
            cos = self.cos_cached[:seq_len].to(device=x_device, dtype=x_dtype)
            sin = self.sin_cached[:seq_len].to(device=x_device, dtype=x_dtype)
            return cos, sin

        ms = self.max_seq_len_training
        k_int = math.ceil(seq_len / ms)
        if k_int % 2 == 1:
            k_int += 1 
        k_new = float(k_int)
        new_extended = ms * k_int

        cache_device = self.cos_cached.device
        cache_dtype = self.cos_cached.dtype

        cos_new, sin_new = self._build_cos_sin_cache(
            seq_len=new_extended,
            k=k_new,
            device=cache_device,
            dtype=cache_dtype,
        )

        if self.dynamic:
            self.k = k_new
            self.max_seq_len_cached = new_extended
            self.register_buffer("cos_cached", cos_new, persistent=False)
            self.register_buffer("sin_cached", sin_new, persistent=False)
            cos = self.cos_cached[:seq_len].to(device=x_device, dtype=x_dtype)
            sin = self.sin_cached[:seq_len].to(device=x_device, dtype=x_dtype)
        else:
            cos = cos_new[:seq_len].to(device=x_device, dtype=x_dtype)
            sin = sin_new[:seq_len].to(device=x_device, dtype=x_dtype)

        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, num_heads, head_dim]
       
        """
        if x.dim() != 4:
            raise ValueError(
                f"NTKAwareRoPE expects x of shape [b, s, nh, hd], "
                f"but got {tuple(x.shape)}"
            )
        bsz, seqlen, nh, hd = x.shape
        if hd != self.head_dim:
            raise ValueError(
                f"Last dim of x ({hd}) must equal head_dim ({self.head_dim})."
            )

        cos, sin = self._maybe_extend_cache(
            seq_len=seqlen,
            x_device=x.device,
            x_dtype=x.dtype,
        )

        out = apply_rotary_pos_emb(x, cos, sin)
        return out

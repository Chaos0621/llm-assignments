import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class DenseMLPWithLoRA(nn.Module):
    """
    GLU-style MLP + optional LoRA adaptation.

    input: X, [batch_size, seq_len, hidden_size] = [b, s, h]

   MLP:
        up   : W_up   ∈ R^{h × ffh}
        gate : W_gate ∈ R^{h × ffh}
        down : W_down ∈ R^{ffh × h}

        U = X W_up                      -> [b, s, ffh]
        G = ϕ(X W_gate)                 -> [b, s, ffh]
        H = G ⊙ U                       -> [b, s, ffh]
        O = H W_down                    -> [b, s, h]

    """

    def __init__(
        self,
        hidden_size: int,
        ff_hidden_size: int,
        activation_type: "MLPActivationType",
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

        self.hidden_size = hidden_size
        self.ff_hidden_size = ff_hidden_size
        self.activation_type = activation_type
        self.init_base_seed = init_base_seed

        self.w_up = nn.Parameter(
            torch.empty(hidden_size, ff_hidden_size, device=device, dtype=dtype)
        )
        self.w_gate = nn.Parameter(
            torch.empty(hidden_size, ff_hidden_size, device=device, dtype=dtype)
        )
        self.w_down = nn.Parameter(
            torch.empty(ff_hidden_size, hidden_size, device=device, dtype=dtype)
        )

        self.lora_rank = int(lora_rank)
        self.lora_dropout_rate = float(lora_dropout_rate)
        self.lora_dropout_seed = int(lora_dropout_seed)
        self.lora_init_base_seed = int(lora_init_base_seed)

        if self.lora_rank > 0:
            if lora_alpha is None:
                lora_alpha = float(self.lora_rank)
            self.lora_alpha = float(lora_alpha)
            self.lora_scaling = self.lora_alpha / self.lora_rank

            self.lora_A = nn.Parameter(
                torch.empty(hidden_size, self.lora_rank, device=device, dtype=dtype)
            )
            self.lora_B = nn.Parameter(
                torch.empty(self.lora_rank, hidden_size, device=device, dtype=dtype)
            )
        else:
            self.lora_alpha = None
            self.lora_scaling = None
            self.lora_A = None
            self.lora_B = None

        self.reset_parameters()


    def _get_activation_fn(self):
        at = self.activation_type

        if at.name == "SIGMOID":
            return torch.sigmoid
        if at.name == "BILINEAR":
            return lambda x: x

        if at.name == "RELU":
            return F.relu
        if at.name == "GELU":
            return F.gelu
        if at.name == "SILU":
            return F.silu
        if at.name == "SWISH":
            return lambda x: x * torch.sigmoid(x)

        raise ValueError(f"Unsupported MLPActivationType: {at}")


    def reset_parameters(self) -> None:
        use_xavier = (
            self.activation_type.name == "SIGMOID"
            or self.activation_type.name == "BILINEAR"
        )

 
        def _init_weight_normal(w: torch.Tensor, seed: int):
            gen_device = w.device if w.is_cuda else "cpu"
            g = torch.Generator(device=gen_device)
            g.manual_seed(seed)
            if use_xavier:
                nn.init.xavier_normal_(w, generator=g)
            else:

                nn.init.kaiming_normal_(
                    w, a=0.0, mode="fan_out", nonlinearity="relu", generator=g
                )

        _init_weight_normal(self.w_up, self.init_base_seed + 1)
        _init_weight_normal(self.w_gate, self.init_base_seed + 2)
        _init_weight_normal(self.w_down, self.init_base_seed + 3)

        if self.lora_rank > 0:
            def _init_weight_uniform(w: torch.Tensor, seed: int):
                gen_device = w.device if w.is_cuda else "cpu"
                g = torch.Generator(device=gen_device)
                g.manual_seed(seed)
                if use_xavier:
                    nn.init.xavier_uniform_(w, generator=g)
                else:
                    nn.init.kaiming_uniform_(
                        w, a=0.0, mode="fan_out", nonlinearity="relu", generator=g
                    )

            _init_weight_uniform(self.lora_A, self.lora_init_base_seed + 1)
            _init_weight_uniform(self.lora_B, self.lora_init_base_seed + 2)


    def _lora_dropout(self, x: torch.Tensor) -> torch.Tensor:

        p = self.lora_dropout_rate
        if p <= 0.0 or not self.training:
            return x

        keep_prob = 1.0 - p

        g = torch.Generator(device=x.device)
        g.manual_seed(self.lora_dropout_seed)

        mask = torch.rand_like(x, dtype=torch.float32, generator=g) < keep_prob
        mask = mask.to(dtype=x.dtype)

        return x * mask / keep_prob


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, s, h]
        return: [b, s, h]
        """
        if x.dim() != 3:
            raise ValueError(
                f"DenseMLPWithLoRA expects input of shape [b, s, h], "
                f"but got {tuple(x.shape)}"
            )

        bsz, seqlen, h = x.shape
        if h != self.hidden_size:
            raise ValueError(
                f"Last dim of input ({h}) must equal hidden_size ({self.hidden_size})."
            )

        orig_dtype = x.dtype
        orig_device = x.device

        act_fn = self._get_activation_fn()

        x_2d = x.view(-1, h)  # [b*s, h]

        up = x_2d @ self.w_up      # [b*s, ffh]
        gate_pre = x_2d @ self.w_gate  # [b*s, ffh]
        gate = act_fn(gate_pre)    # [b*s, ffh]

        hidden = gate * up         # [b*s, ffh]
        out_core = hidden @ self.w_down  # [b*s, h]
        out_core = out_core.view(bsz, seqlen, h)


        if self.lora_rank > 0:
            lora_mid = x_2d @ self.lora_A              # [b*s, r]
            lora_out_2d = lora_mid @ self.lora_B       # [b*s, h]
            lora_out_2d = lora_out_2d * self.lora_scaling
            lora_out = lora_out_2d.view(bsz, seqlen, h)
            lora_out = self._lora_dropout(lora_out)
            out = out_core + lora_out
        else:
            out = out_core

        out = out.to(dtype=orig_dtype, device=orig_device)
        return out

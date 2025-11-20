import torch
from torch import Tensor


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Args:
        x:   [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        out: [batch, seq_len, num_heads, head_dim]
    """
    if x.dim() != 4:
        raise ValueError(
            f"apply_rotary_pos_emb expects x of shape [b, s, nh, hd], "
            f"but got {tuple(x.shape)}"
        )
    bsz, seqlen, num_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be even for rotary embedding."
        )

    if cos.shape != (seqlen, head_dim) or sin.shape != (seqlen, head_dim):
        raise ValueError(
            f"cos/sin must have shape [seq_len, head_dim], "
            f"got cos {tuple(cos.shape)}, sin {tuple(sin.shape)}"
        )

    cos = cos.to(device=x.device, dtype=x.dtype)[None, :, None, :]
    sin = sin.to(device=x.device, dtype=x.dtype)[None, :, None, :]

    x_even = x[..., ::2]  
    x_odd = x[..., 1::2]  

    cos_even = cos[..., ::2]  
    sin_even = sin[..., ::2]


    x_rot_even = x_even * cos_even - x_odd * sin_even
    x_rot_odd = x_even * sin_even + x_odd * cos_even


    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1) 
    x_rot = x_rot.reshape(bsz, seqlen, num_heads, head_dim)

    return x_rot

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from assignment1.assignment1_RMS import GroupRMSNorm


class OfflineSlidingWindowAttn(nn.Module):
    """
    Args:
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Per-head hidden dimension.
        window_size: Sliding window radius in tokens.
        causal: Whether to restrict each query to attend only to keys at non-future positions.
        group_size: Group size for GroupRMSNorm.
        use_qk_norm: Whether to apply GroupRMSNorm to queries and keys.
        pack_format: Packing format of Q, K, V.
        layout: Layout of Q, K, V.
        softmax_scale: Optional scaling factor applied to QK^T before softmax; if None, uses 1/sqrt(head_dim).
        softmax_temp: Softmax temperature; only used when softmax_cap is None.
        softmax_cap: Softmax capping parameter; if not None, capping is used instead of temperature.
        softmax_clip_lower: Lower affine bound for softmax clipping.
        softmax_clip_upper: Upper affine bound for softmax clipping.
        attn_dropout: Dropout rate applied to attention weights.
        attn_dropout_seed: Random seed controlling the attention dropout mask.
        rmsnorm_eps: Epsilon for GroupRMSNorm.
        rmsnorm_init_range: Initialization range for GroupRMSNorm scaling parameters.
        rmsnorm_init_seed_q: Random seed for the query GroupRMSNorm parameters.
        rmsnorm_init_seed_k: Random seed for the key GroupRMSNorm parameters.
        device: Device for learnable parameters.
        dtype: Data type for learnable parameters in GroupRMSNorm.
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int,
        causal: bool = True,
        group_size: int = 1,
        use_qk_norm: bool = True,
        pack_format=None,
        layout=None,
        softmax_scale: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_cap: Optional[float] = None,
        softmax_clip_lower: Optional[float] = 0.0,
        softmax_clip_upper: Optional[float] = 1.0,
        attn_dropout: float = 0.0,
        attn_dropout_seed: int = 0,
        rmsnorm_eps: float = 1e-6,
        rmsnorm_init_range = (-1.0, 1.0),
        rmsnorm_init_seed_q: int = 42,
        rmsnorm_init_seed_k: int = 43,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if head_dim % group_size != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by group_size ({group_size})")
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads for MQA/GQA")

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.causal = causal
        self.group_size = group_size
        self.use_qk_norm = use_qk_norm
        self.pack_format = pack_format
        self.layout = layout
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.softmax_temp = softmax_temp
        self.softmax_cap = softmax_cap
        self.softmax_clip_lower = softmax_clip_lower
        self.softmax_clip_upper = softmax_clip_upper
        self.attn_dropout = attn_dropout
        self.attn_dropout_seed = attn_dropout_seed

        self.q_norm = (
            GroupRMSNorm(
                hidden_size=num_q_heads * head_dim,
                group_size=group_size,
                eps=rmsnorm_eps,
                init_range=rmsnorm_init_range,
                init_seed=rmsnorm_init_seed_q,
                device=device,
                dtype=dtype,
            )
            if use_qk_norm
            else None
        )
        self.k_norm = (
            GroupRMSNorm(
                hidden_size=num_q_heads * head_dim,
                group_size=group_size,
                eps=rmsnorm_eps,
                init_range=rmsnorm_init_range,
                init_seed=rmsnorm_init_seed_k,
                device=device,
                dtype=dtype,
            )
            if use_qk_norm
            else None
        )

    def _sliding_window_mask(
        self,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        s_q = seq_len_q
        s_k = seq_len_kv
        max_len = max(s_q, s_k)
        q_idx = torch.arange(s_q, device=device).unsqueeze(1)
        k_idx = torch.arange(s_k, device=device).unsqueeze(0)
        q_pos = (max_len - s_q) + q_idx
        k_pos = (max_len - s_k) + k_idx
        diff = q_pos - k_pos
        allow = diff.abs() <= self.window_size
        if self.causal:
            allow = allow & (diff >= 0)
        mask = torch.zeros(s_q, s_k, device=device, dtype=dtype)
        neg_inf = torch.finfo(dtype).min
        mask = torch.where(allow, mask, torch.full_like(mask, neg_inf))
        return mask

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):

        if not self.use_qk_norm:
            return q, k
        bq, sq, hq, hd = q.shape
        bk, sk, hk, _ = k.shape
        q_3d = q.reshape(bq, sq, hq * hd)
        k_3d = k.reshape(bk, sk, hk * hd)
        q_3d = self.q_norm(q_3d)
        k_3d = self.k_norm(k_3d)
        q_out = q_3d.reshape(bq, sq, hq, hd)
        k_out = k_3d.reshape(bk, sk, hk, hd)
        return q_out, k_out

    def _core_attention_bshd(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        bsz, seq_len_q, hq, hd = q.shape
        _, seq_len_kv, hkv, _ = k.shape
        if hq != hkv:
            if hq % hkv != 0:
                raise ValueError("num_q_heads must be divisible by num_kv_heads")
            repeat_factor = hq // hkv
            k = torch.repeat_interleave(k, repeats=repeat_factor, dim=2)
            v = torch.repeat_interleave(v, repeats=repeat_factor, dim=2)
        q, k = self._apply_qk_norm(q, k)
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        q_flat = q.reshape(bsz * hq, seq_len_q, hd)
        k_flat = k.reshape(bsz * hq, seq_len_kv, hd)
        logits = torch.bmm(q_flat, k_flat.transpose(1, 2))
        logits = logits * self.softmax_scale
        if self.softmax_cap is None:
            logits = logits / self.softmax_temp
        else:
            cap = self.softmax_cap
            logits = cap * torch.tanh(logits / cap)
        mask = self._sliding_window_mask(seq_len_q, seq_len_kv, logits.device, logits.dtype)
        logits = logits + mask.unsqueeze(0)
        attn = F.softmax(logits, dim=-1)
        if self.softmax_clip_lower is not None and self.softmax_clip_upper is not None:
            l = self.softmax_clip_lower
            r = self.softmax_clip_upper
            attn = (r - l) * attn + l
            attn = torch.clamp(attn, 0.0, 1.0)
        if self.attn_dropout > 0.0 and self.training:
            keep_prob = 1.0 - self.attn_dropout
            g = torch.Generator(device=attn.device)
            g.manual_seed(self.attn_dropout_seed)
            mask_d = (torch.rand_like(attn, generator=g) < keep_prob).to(attn.dtype)
            attn = attn * mask_d / keep_prob
        v_flat = v.reshape(bsz * hq, seq_len_kv, hd)
        out_flat = torch.bmm(attn, v_flat)
        out = out_flat.reshape(bsz, seq_len_q, hq, hd)
        return out

    def _unpack_qkv_bshd(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
    ):
        """
        Args:
            q: Packed query tensor according to pack_format in BSHD layout.
            k: Packed key tensor or None depending on pack_format.
            v: Packed value tensor or None depending on pack_format.
        Returns:
            Tuple of (q_bshd, k_bshd, v_bshd) all in BSHD layout with explicit num_q_heads and num_kv_heads.
        """
        from  assignment3.A3_T1 import AttnQKVPackFormat

        if self.pack_format == AttnQKVPackFormat.Q_K_V:
            if k is None or v is None:
                raise ValueError("K and V must be provided for Q_K_V pack_format")
            q_bshd = q
            k_bshd = k
            v_bshd = v
        elif self.pack_format == AttnQKVPackFormat.Q_KV:
            if k is None:
                raise ValueError("Packed KV tensor must be provided for Q_KV pack_format")
            kv = k
            b, s_kv, h2, d = kv.shape
            if h2 != 2 * self.num_kv_heads:
                raise ValueError("Packed KV heads dimension does not match 2 * num_kv_heads")
            k_bshd = kv[:, :, : self.num_kv_heads, :]
            v_bshd = kv[:, :, self.num_kv_heads :, :]
            q_bshd = q
        elif self.pack_format == AttnQKVPackFormat.QKV:
            qkv = q
            b, s, h_all, d = qkv.shape
            expected_h_all = self.num_q_heads + 2 * self.num_kv_heads
            if h_all != expected_h_all:
                raise ValueError("Packed QKV heads dimension does not match num_q_heads + 2 * num_kv_heads")
            hq = self.num_q_heads
            hkv = self.num_kv_heads
            q_bshd = qkv[:, :, :hq, :]
            k_bshd = qkv[:, :, hq : hq + hkv, :]
            v_bshd = qkv[:, :, hq + hkv :, :]
        else:
            raise ValueError("Unsupported pack_format")
        return q_bshd, k_bshd, v_bshd

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor or packed Q/K/V tensor, in the layout specified by layout and pack_format.
            k: Key tensor or packed KV tensor depending on pack_format; may be None when using QKV format.
            v: Value tensor; only used when pack_format is Q_K_V.
            cu_seqlens_q: Cumulative sequence lengths for THD layout queries, shape [batch_size + 1].
            cu_seqlens_kv: Cumulative sequence lengths for THD layout keys/values, shape [batch_size + 1].
        Returns:
            Output tensor with the same layout as Q for the query heads.
        """
        from assignment3 import AttnQKVLayout, AttnQKVPackFormat

        orig_dtype = q.dtype
        orig_device = q.device

        if self.layout == AttnQKVLayout.BSHD:
            q_bshd, k_bshd, v_bshd = self._unpack_qkv_bshd(q, k, v)
            out_bshd = self._core_attention_bshd(q_bshd, k_bshd, v_bshd)
            return out_bshd.to(device=orig_device, dtype=orig_dtype)

        if self.layout == AttnQKVLayout.SBHD:
            q_b = q.permute(1, 0, 2, 3)
            k_b = None if k is None else k.permute(1, 0, 2, 3)
            v_b = None if v is None else v.permute(1, 0, 2, 3)
            q_bshd, k_bshd, v_bshd = self._unpack_qkv_bshd(q_b, k_b, v_b)
            out_bshd = self._core_attention_bshd(q_bshd, k_bshd, v_bshd)
            out_sbhd = out_bshd.permute(1, 0, 2, 3)
            return out_sbhd.to(device=orig_device, dtype=orig_dtype)

        if self.layout == AttnQKVLayout.THD:
            if cu_seqlens_q is None or cu_seqlens_kv is None:
                raise ValueError("cu_seqlens_q and cu_seqlens_kv must be provided for THD layout")
            if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
                raise ValueError("cu_seqlens_q and cu_seqlens_kv must have the same length")
            batch_size = cu_seqlens_q.numel() - 1
            outputs = []
            for b in range(batch_size):
                qs = cu_seqlens_q[b].item()
                qe = cu_seqlens_q[b + 1].item()
                ks = cu_seqlens_kv[b].item()
                ke = cu_seqlens_kv[b + 1].item()
                if self.pack_format == AttnQKVPackFormat.Q_K_V:
                    q_seg = q[qs:qe]
                    k_seg = k[ks:ke] if k is not None else None
                    v_seg = v[ks:ke] if v is not None else None
                    q_bshd, k_bshd, v_bshd = self._unpack_qkv_bshd(
                        q_seg.unsqueeze(0),
                        k_seg.unsqueeze(0) if k_seg is not None else None,
                        v_seg.unsqueeze(0) if v_seg is not None else None,
                    )
                elif self.pack_format == AttnQKVPackFormat.Q_KV:
                    q_seg = q[qs:qe]
                    kv_seg = k[ks:ke] if k is not None else None
                    q_bshd, k_bshd, v_bshd = self._unpack_qkv_bshd(
                        q_seg.unsqueeze(0),
                        kv_seg.unsqueeze(0) if kv_seg is not None else None,
                        None,
                    )
                elif self.pack_format == AttnQKVPackFormat.QKV:
                    qkv_seg = q[qs:qe]
                    q_bshd, k_bshd, v_bshd = self._unpack_qkv_bshd(
                        qkv_seg.unsqueeze(0),
                        None,
                        None,
                    )
                else:
                    raise ValueError("Unsupported pack_format for THD layout")
                out_bshd = self._core_attention_bshd(q_bshd, k_bshd, v_bshd)
                outputs.append(out_bshd.squeeze(0))
            out_thd = torch.cat(outputs, dim=0)
            return out_thd.to(device=orig_device, dtype=orig_dtype)

        raise ValueError("Unsupported layout")

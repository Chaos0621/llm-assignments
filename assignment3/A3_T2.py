import torch
from typing import Optional

from assignment3.A3_T1 import OfflineSlidingWindowAttn
class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """
    Args:
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Per-head hidden dimension.
        window_size: Sliding window radius in tokens.
        causal: Whether to restrict each query to attend only to non-future keys.
        group_size: Group size for GroupRMSNorm.
        use_qk_norm: Whether to apply GroupRMSNorm to queries and keys.
        block_size_q: Block size along the query sequence dimension.
        block_size_kv: Block size along the key/value sequence dimension.
        seqlen_q: Global query sequence length before padding to multiples of block_size_q.
        seqlen_kv: Global key/value sequence length before padding to multiples of block_size_kv.
        softmax_scale: Optional scaling factor applied to QK^T; if None, uses 1/sqrt(head_dim).
        softmax_temp: Softmax temperature; only used when softmax_cap is None.
        softmax_cap: Softmax capping parameter; if not None, capping is used instead of temperature.
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
        block_size_q: int = 128,
        block_size_kv: int = 128,
        seqlen_q: int = 0,
        seqlen_kv: int = 0,
        softmax_scale: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_cap: Optional[float] = None,
        rmsnorm_eps: float = 1e-6,
        rmsnorm_init_range = (-1.0, 1.0),
        rmsnorm_init_seed_q: int = 42,
        rmsnorm_init_seed_k: int = 43,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from assignment3 import AttnQKVPackFormat, AttnQKVLayout

        super().__init__(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            window_size=window_size,
            causal=causal,
            group_size=group_size,
            use_qk_norm=use_qk_norm,
            pack_format=AttnQKVPackFormat.Q_K_V,
            layout=AttnQKVLayout.BSHD,
            softmax_scale=softmax_scale,
            softmax_temp=softmax_temp,
            softmax_cap=softmax_cap,
            softmax_clip_lower=None,
            softmax_clip_upper=None,
            attn_dropout=0.0,
            attn_dropout_seed=0,
            rmsnorm_eps=rmsnorm_eps,
            rmsnorm_init_range=rmsnorm_init_range,
            rmsnorm_init_seed_q=rmsnorm_init_seed_q,
            rmsnorm_init_seed_k=rmsnorm_init_seed_k,
            device=device,
            dtype=dtype,
        )

        if seqlen_q <= 0 or seqlen_kv <= 0:
            raise ValueError("seqlen_q and seqlen_kv must be positive for online attention")

        self.block_size_q = int(block_size_q)
        self.block_size_kv = int(block_size_kv)
        self.seqlen_q = int(seqlen_q)
        self.seqlen_kv = int(seqlen_kv)

        mask_device = device if device is not None else torch.device("cpu")
        mask_full = self._sliding_window_mask(
            seq_len_q=self.seqlen_q,
            seq_len_kv=self.seqlen_kv,
            device=mask_device,
            dtype=torch.float32,
        )
        self.register_buffer("mask_full", mask_full, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """
        Args:
            q: Query block tensor Q_bqi of shape [batch_size, block_size_q, num_q_heads, head_dim].
            k: Key block tensor K_bkvj of shape [batch_size, block_size_kv, num_kv_heads, head_dim].
            v: Value block tensor V_bkvj of shape [batch_size, block_size_kv, num_kv_heads, head_dim].
            global_o: Global output buffer O of shape [batch_size, total_padded_seqlen_q, num_q_heads, head_dim],
                      updated in-place with the aggregated online attention outputs.
            global_lse: Global log-sum-exp statistics tensor of shape
                        [batch_size, total_padded_seqlen_q, num_q_heads] in float32, updated in-place.
            block_idx_q: Block index along the query sequence dimension.
            block_idx_kv: Block index along the key/value sequence dimension.
        Returns:
            None. The method updates global_o and global_lse in-place.
        """
        bsz, bq_len, hq, hd = q.shape
        _, bkv_len, hkv, _ = k.shape

        if hq != self.num_q_heads or hkv != self.num_kv_heads:
            if hq != self.num_q_heads:
                raise ValueError("Mismatch between q heads and num_q_heads")
            if hkv != self.num_kv_heads:
                raise ValueError("Mismatch between kv heads and num_kv_heads")

        start_q = block_idx_q * self.block_size_q
        start_k = block_idx_kv * self.block_size_kv
        end_q = start_q + bq_len
        end_k = start_k + bkv_len

        valid_q = max(0, min(end_q, self.seqlen_q) - start_q)
        valid_k = max(0, min(end_k, self.seqlen_kv) - start_k)
        if valid_q <= 0 or valid_k <= 0:
            return

        if hq != hkv:
            if hq % hkv != 0:
                raise ValueError("num_q_heads must be divisible by num_kv_heads")
            repeat_factor = hq // hkv
            k = torch.repeat_interleave(k, repeats=repeat_factor, dim=2)
            v = torch.repeat_interleave(v, repeats=repeat_factor, dim=2)

        q_block, k_block = self._apply_qk_norm(q, k)
        q_block = q_block.to(dtype=torch.float32)
        k_block = k_block.to(dtype=torch.float32)
        v_block = v.to(dtype=torch.float32)

        q_flat = q_block.reshape(bsz * hq, bq_len, hd)
        k_flat = k_block.reshape(bsz * hq, bkv_len, hd)

        logits = torch.bmm(q_flat, k_flat.transpose(1, 2))
        logits = logits * self.softmax_scale
        if self.softmax_cap is None:
            logits = logits / self.softmax_temp
        else:
            cap = self.softmax_cap
            logits = cap * torch.tanh(logits / cap)

        neg_inf = torch.finfo(logits.dtype).min
        block_mask = torch.full(
            (bq_len, bkv_len),
            fill_value=neg_inf,
            device=logits.device,
            dtype=logits.dtype,
        )
        sub_mask = self.mask_full[
            start_q : start_q + valid_q,
            start_k : start_k + valid_k,
        ].to(device=logits.device, dtype=logits.dtype)
        block_mask[:valid_q, :valid_k] = sub_mask
        logits = logits + block_mask.unsqueeze(0)

        lse_block = torch.logsumexp(logits, dim=-1)
        attn = torch.exp(logits - lse_block.unsqueeze(-1))

        v_flat = v_block.reshape(bsz * hq, bkv_len, hd)
        out_block_flat = torch.bmm(attn, v_flat)
        out_block = out_block_flat.view(bsz, hq, bq_len, hd)[:, :, :valid_q, :]

        lse_block_bhq = lse_block.view(bsz, hq, bq_len)[:, :, :valid_q]

        global_lse_slice = global_lse[:, start_q : start_q + valid_q, :]
        global_lse_bhq = global_lse_slice.permute(0, 2, 1)

        new_lse = torch.logaddexp(global_lse_bhq, lse_block_bhq)

        valid_mask = torch.isfinite(new_lse)
        c_prev = torch.zeros_like(new_lse)
        c_local = torch.zeros_like(new_lse)
        if valid_mask.any():
            c_prev_val = torch.exp(global_lse_bhq[valid_mask] - new_lse[valid_mask])
            c_local_val = torch.exp(lse_block_bhq[valid_mask] - new_lse[valid_mask])
            c_prev[valid_mask] = c_prev_val
            c_local[valid_mask] = c_local_val

        global_o_slice = global_o[:, start_q : start_q + valid_q, :, :]
        global_o_bhqv = global_o_slice.permute(0, 2, 1, 3)

        c_prev_exp = c_prev.unsqueeze(-1)
        c_local_exp = c_local.unsqueeze(-1)
        new_o_bhqv = c_prev_exp * global_o_bhqv + c_local_exp * out_block

        global_o[:, start_q : start_q + valid_q, :, :] = new_o_bhqv.permute(0, 2, 1, 3)
        global_lse[:, start_q : start_q + valid_q, :] = new_lse.permute(0, 2, 1)

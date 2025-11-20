import math
from typing import Optional

import torch
import torch.nn as nn

from assignment3 import (
    AttnQKVLayout,
    AttnQKVPackFormat,
    OfflineSlidingWindowAttn,
)
from assignment3 import DenseMLPWithLoRA, SparseMLPWithLoRA
from assignment1 import GroupRMSNorm
from assignment2 import NTKAwareRoPE
from assignment4.A4_T1 import TransformerDecoderKVCache


class TransformerDecoderLayer(nn.Module):
    """
    Args:
        config: TransformerConfig instance providing all hyper-parameters and initialization settings.
        layer_idx: Index of this decoder layer in [0, config.num_layers - 1].
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()

        if config.group_size is not None:
            norm_group_size = config.group_size
        else:
            norm_group_size = config.hidden_size

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_q_head
        self.num_kv_heads = config.num_kv_head
        self.qkv_layout = config.qkv_layout
        self.qkv_pack_format = config.qkv_pack_format
        self.max_seq_len = config.max_seq_len
        self.param_dtype = config.param_dtype
        self.param_device = torch.device(config.param_device)

        self.attn_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=norm_group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 1,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_q_head * config.head_dim,
            bias=False,
            device=self.param_device,
            dtype=self.param_dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_head * config.head_dim,
            bias=False,
            device=self.param_device,
            dtype=self.param_dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_head * config.head_dim,
            bias=False,
            device=self.param_device,
            dtype=self.param_dtype,
        )
        self.o_proj = nn.Linear(
            config.num_q_head * config.head_dim,
            config.hidden_size,
            bias=False,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        if config.qk_norm_group_size is not None:
            qk_group_size = config.qk_norm_group_size
        else:
            qk_group_size = norm_group_size

        if config.window_size is not None:
            window_size = config.window_size
        else:
            window_size = config.max_seq_len

        self.attn = OfflineSlidingWindowAttn(
            num_q_heads=config.num_q_head,
            num_kv_heads=config.num_kv_head,
            head_dim=config.head_dim,
            window_size=window_size,
            causal=config.causal,
            group_size=qk_group_size,
            use_qk_norm=config.apply_qk_norm,
            pack_format=config.qkv_pack_format,
            layout=config.qkv_layout,
            softmax_scale=config.softmax_scale,
            softmax_temp=config.softmax_temp,
            softmax_cap=config.softmax_cap,
            softmax_clip_lower=config.softmax_clip_range[0],
            softmax_clip_upper=config.softmax_clip_range[1],
            attn_dropout=config.softmax_dropout_rate,
            attn_dropout_seed=config.softmax_dropout_seed + layer_idx,
            rmsnorm_eps=config.eps,
            rmsnorm_init_range=config.norm_init_range,
            rmsnorm_init_seed_q=config.init_base_seed + layer_idx + 2,
            rmsnorm_init_seed_k=config.init_base_seed + layer_idx + 2,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        self.rope = NTKAwareRoPE(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            k=config.rope_ratio,
            dynamic=config.rope_dynamic,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        self.mlp_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=norm_group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 3,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        mlp_init_seed = config.init_base_seed + layer_idx + 4
        lora_init_base_seed = config.lora_init_base_seed + layer_idx
        lora_dropout_seed = config.lora_dropout_seed + layer_idx

        if config.num_experts is None:
            self.mlp = DenseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ff_hidden_size=config.ffh_size,
                activation_type=config.activation_type,
                init_base_seed=mlp_init_seed,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=lora_dropout_seed,
                lora_init_base_seed=lora_init_base_seed,
                device=self.param_device,
                dtype=self.param_dtype,
            )
        else:
            self.mlp = SparseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ff_hidden_size=config.ffh_size,
                num_experts=config.num_experts,
                top_k=config.moe_topk,
                rank=config.rank,
                world_size=config.world_size,
                activation_type=config.activation_type,
                init_mean=config.gate_init_mean,
                init_std=config.gate_init_std,
                init_base_seed=mlp_init_seed,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=lora_dropout_seed,
                lora_init_base_seed=lora_init_base_seed,
                device=self.param_device,
                dtype=self.param_dtype,
            )

        self._reset_proj_parameters()

    def _reset_proj_parameters(self) -> None:
        """
        Args:
            None.
        Returns:
            None. Initializes projection matrices with normal distribution using config.proj_init_* and layer-specific seeds.
        """
        g_qkv = torch.Generator(device=self.param_device)
        g_qkv.manual_seed(self.config.proj_init_seed + self.layer_idx + 1)
        with torch.no_grad():
            torch.nn.init.normal_(
                self.q_proj.weight,
                mean=self.config.proj_init_mean,
                std=self.config.proj_init_std,
                generator=g_qkv,
            )
            torch.nn.init.normal_(
                self.k_proj.weight,
                mean=self.config.proj_init_mean,
                std=self.config.proj_init_std,
                generator=g_qkv,
            )
            torch.nn.init.normal_(
                self.v_proj.weight,
                mean=self.config.proj_init_mean,
                std=self.config.proj_init_std,
                generator=g_qkv,
            )

        g_o = torch.Generator(device=self.param_device)
        g_o.manual_seed(self.config.proj_init_seed + self.layer_idx + 2)
        with torch.no_grad():
            torch.nn.init.normal_(
                self.o_proj.weight,
                mean=self.config.proj_init_mean,
                std=self.config.proj_init_std,
                generator=g_o,
            )

    def _apply_rope_bshd(self, q_bshd: torch.Tensor, k_bshd: torch.Tensor, offset: int = 0):
      
        q_rope = self.rope(q_bshd)
        k_rope = self.rope(k_bshd)
        return q_rope, k_rope

    def _run_attention(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
    ) -> torch.Tensor:

        layout = self.qkv_layout
        pack_format = self.qkv_pack_format

        if layout == AttnQKVLayout.BSHD:
            if pack_format == AttnQKVPackFormat.Q_K_V:
                out = self.attn(q_bshd, k_bshd, v_bshd)
            elif pack_format == AttnQKVPackFormat.Q_KV:
                kv = torch.cat([k_bshd, v_bshd], dim=2)
                out = self.attn(q_bshd, kv)
            elif pack_format == AttnQKVPackFormat.QKV:
                qkv = torch.cat([q_bshd, k_bshd, v_bshd], dim=2)
                out = self.attn(qkv)
            else:
                raise ValueError("Unsupported qkv_pack_format")
            return out

        if layout == AttnQKVLayout.SBHD:
            q_sbhd = q_bshd.permute(1, 0, 2, 3)
            k_sbhd = k_bshd.permute(1, 0, 2, 3)
            v_sbhd = v_bshd.permute(1, 0, 2, 3)
            if pack_format == AttnQKVPackFormat.Q_K_V:
                out_sbhd = self.attn(q_sbhd, k_sbhd, v_sbhd)
            elif pack_format == AttnQKVPackFormat.Q_KV:
                kv_sbhd = torch.cat([k_sbhd, v_sbhd], dim=2)
                out_sbhd = self.attn(q_sbhd, kv_sbhd)
            elif pack_format == AttnQKVPackFormat.QKV:
                qkv_sbhd = torch.cat([q_sbhd, k_sbhd, v_sbhd], dim=2)
                out_sbhd = self.attn(qkv_sbhd)
            else:
                raise ValueError("Unsupported qkv_pack_format")
            out_bshd = out_sbhd.permute(1, 0, 2, 3)
            return out_bshd

        if layout == AttnQKVLayout.THD:
            if cu_seqlens is None:
                raise ValueError("cu_seqlens must be provided for THD layout")
            bsz, seqlen_q, hq, hd = q_bshd.shape
            _, seqlen_kv, hkv, _ = k_bshd.shape
            if bsz != 1:
                raise ValueError("For THD layout, batch size of hidden states must be 1")
            q_thd = q_bshd.view(seqlen_q, hq, hd)
            k_thd = k_bshd.view(seqlen_kv, hkv, hd)
            v_thd = v_bshd.view(seqlen_kv, hkv, hd)
            if pack_format == AttnQKVPackFormat.Q_K_V:
                out_thd = self.attn(q_thd, k_thd, v_thd, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
            elif pack_format == AttnQKVPackFormat.Q_KV:
                kv_thd = torch.cat([k_thd, v_thd], dim=1)
                out_thd = self.attn(q_thd, kv_thd, None, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
            elif pack_format == AttnQKVPackFormat.QKV:
                qkv_thd = torch.cat([q_thd, k_thd, v_thd], dim=1)
                out_thd = self.attn(qkv_thd, None, None, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
            else:
                raise ValueError("Unsupported qkv_pack_format")
            out_bshd = out_thd.view(1, seqlen_q, hq, hd)
            return out_bshd

        raise ValueError("Unsupported qkv_layout")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional["TransformerDecoderKVCache"] = None,
    ) -> torch.Tensor:
    
        orig_dtype = x.dtype
        orig_device = x.device

        x_layer = x.to(device=self.param_device, dtype=self.param_dtype)

        residual = x_layer
        x_norm = self.attn_norm(x_layer)

        q_proj = self.q_proj(x_norm)
        k_proj = self.k_proj(x_norm)
        v_proj = self.v_proj(x_norm)

        bsz, seqlen, _ = x_norm.shape

        q_bshd = q_proj.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k_bshd = k_proj.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v_bshd = v_proj.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q_bshd_rope, k_bshd_rope = self._apply_rope_bshd(q_bshd, k_bshd)

        if (
            kv_cache is not None
            and self.qkv_layout == AttnQKVLayout.BSHD
            and self.qkv_pack_format == AttnQKVPackFormat.Q_K_V
        ):
            if kv_cache.has(self.layer_idx):
                k_prev, v_prev, _ = kv_cache.get(self.layer_idx)
                if k_prev is not None and v_prev is not None:
                    k_all = torch.cat([k_prev, k_bshd_rope], dim=1)
                    v_all = torch.cat([v_prev, v_bshd], dim=1)
                else:
                    k_all = k_bshd_rope
                    v_all = v_bshd
            else:
                k_all = k_bshd_rope
                v_all = v_bshd
            kv_cache.set(self.layer_idx, k_all, v_all, None)
        else:
            k_all = k_bshd_rope
            v_all = v_bshd

        attn_out_bshd = self._run_attention(q_bshd_rope, k_all, v_all, cu_seqlens)
        attn_out_2d = attn_out_bshd.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        attn_out = self.o_proj(attn_out_2d)
        x_after_attn = attn_out + residual

        residual2 = x_after_attn
        x_mlp_norm = self.mlp_norm(x_after_attn)
        mlp_out = self.mlp(x_mlp_norm)
        x_out = mlp_out + residual2

        return x_out.to(device=orig_device, dtype=orig_dtype)

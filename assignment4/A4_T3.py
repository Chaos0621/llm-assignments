import torch
import torch.nn as nn
from typing import Optional

from assignment3 import ParallelVocabEmbedding
from assignment2 import TransformerDecoderLayer, TransformerDecoderKVCache
from assignment1 import GroupRMSNorm


class TransformerDecoderBlock(nn.Module):
    """
    Args:
        config: TransformerConfig instance providing all hyper-parameters and initialization settings.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.param_dtype = config.param_dtype
        self.param_device = torch.device(config.param_device)

        self.vocab_embed = ParallelVocabEmbedding(
            vocab_size=config.vocab_size,
            emb_size=config.hidden_size,
            rank=config.rank,
            world_size=config.world_size,
            init_mean=config.vocab_init_mean,
            init_std=config.vocab_init_std,
            init_base_seed=config.init_base_seed,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        layers = []
        for layer_idx in range(config.num_layers):
            layer = TransformerDecoderLayer(config=config, layer_idx=layer_idx)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        if config.group_size is not None:
            norm_group_size = config.group_size
        else:
            norm_group_size = config.hidden_size

        self.final_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=norm_group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=self.param_device,
            dtype=self.param_dtype,
        )

        if config.lm_head_tied:
            if not hasattr(self.vocab_embed, "weight"):
                raise AttributeError("ParallelVocabEmbedding must have attribute 'weight' to tie lm_head")
            self.lm_head.weight = self.vocab_embed.weight
        else:
            g = torch.Generator(device=self.param_device)
            g.manual_seed(self.config.proj_init_seed)
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.lm_head.weight,
                    mean=self.config.proj_init_mean,
                    std=self.config.proj_init_std,
                    generator=g,
                )

        self.kv_cache = TransformerDecoderKVCache(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers,
        )

    def get_kv_cache(self) -> TransformerDecoderKVCache:
 
        return self.kv_cache

    def set_kv_cache(self, kv_cache: "TransformerDecoderKVCache") -> None:
    
        self.kv_cache = kv_cache

    def reset_kv_cache(self) -> None:

        if self.kv_cache is not None:
            self.kv_cache.reset()

    def num_parameters(self, learnable_only: bool = False, unit: str = "1") -> float:
   
        if learnable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        else:
            params = self.parameters()
        total = sum(p.numel() for p in params)
        if unit == "1":
            return float(total)
        if unit == "K":
            return total / 1e3
        if unit == "M":
            return total / 1e6
        if unit == "B":
            return total / 1e9
        raise ValueError(f"Unsupported unit: {unit}")

    def num_memory_footprint(self, unit: str = "B") -> float:

        total_bytes = 0
        for p in self.parameters():
            total_bytes += p.numel() * p.element_size()
        if unit == "B":
            return float(total_bytes)
        if unit == "KB":
            return total_bytes / 1024.0
        if unit == "MB":
            return total_bytes / (1024.0 ** 2)
        if unit == "GB":
            return total_bytes / (1024.0 ** 3)
        raise ValueError(f"Unsupported unit: {unit}")

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        orig_device = input_ids.device
        ids = input_ids.to(self.param_device)
        hidden = self.vocab_embed(ids)

        kv_cache = self.kv_cache if (not self.training and self.kv_cache is not None) else None

        x = hidden
        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, kv_cache=kv_cache)

        x_norm = self.final_norm(x)
        logits = self.lm_head(x_norm)
        logits = logits.to(orig_device)
        return logits

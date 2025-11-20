import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from assignment3 import AttnQKVLayout


class TransformerDecoderKVCache(nn.Module):
    """
    Args:
        qkv_layout: Layout of K/V tensors, one of AttnQKVLayout.BSHD, SBHD, THD.
        num_layers: Optional number of decoder layers for which this cache may be used.
    """

    def __init__(self, qkv_layout: AttnQKVLayout, num_layers: int = 1):
        super().__init__()
        if not isinstance(qkv_layout, AttnQKVLayout):
            raise TypeError(f"qkv_layout must be an AttnQKVLayout, got {type(qkv_layout)}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.qkv_layout = qkv_layout
        self.num_layers = num_layers
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}

    def has(self, layer_idx: int) -> bool:
     
        return layer_idx in self._cache

    def get(
        self,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            layer_idx: Index of the decoder layer.
        Returns:
            A tuple (k, v, cu_seqlens) for the given layer index, where:
                k: Cached key tensor or None if not present.
                v: Cached value tensor or None if not present.
                cu_seqlens: Cached cumulative sequence lengths tensor if qkv_layout is THD, otherwise None.
        """
        return self._cache.get(layer_idx, (None, None, None))

    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
   
        if self.qkv_layout == AttnQKVLayout.THD:
            if cu_seqlens is None:
                raise ValueError("cu_seqlens must be provided when qkv_layout is THD")
        else:
            if cu_seqlens is not None:
                raise ValueError("cu_seqlens must be None when qkv_layout is not THD")

        if k.dtype != v.dtype or k.device != v.device:
            raise ValueError("Key and value tensors must have the same dtype and device")

        if self.qkv_layout == AttnQKVLayout.BSHD:
            if k.dim() != 4 or v.dim() != 4:
                raise ValueError("K/V must be 4D tensors for BSHD layout")
            if k.shape[0] != v.shape[0] or k.shape[2:] != v.shape[2:]:
                raise ValueError("Batch and head/hidden dimensions of K/V must match for BSHD layout")
        elif self.qkv_layout == AttnQKVLayout.SBHD:
            if k.dim() != 4 or v.dim() != 4:
                raise ValueError("K/V must be 4D tensors for SBHD layout")
            if k.shape[1:] != v.shape[1:]:
                raise ValueError("Head/hidden dimensions of K/V must match for SBHD layout")
        elif self.qkv_layout == AttnQKVLayout.THD:
            if k.dim() != 3 or v.dim() != 3:
                raise ValueError("K/V must be 3D tensors for THD layout")
            if k.shape[1:] != v.shape[1:]:
                raise ValueError("Head/hidden dimensions of K/V must match for THD layout")
            if cu_seqlens is not None and cu_seqlens.dim() != 1:
                raise ValueError("cu_seqlens must be 1D for THD layout")

        self._cache[layer_idx] = (k, v, cu_seqlens)

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
  
        if not self.has(layer_idx):
            self.set(layer_idx, k, v, cu_seqlens)
            return

        k_prev, v_prev, cu_prev = self._cache[layer_idx]

        if k_prev is None or v_prev is None:
            self.set(layer_idx, k, v, cu_seqlens)
            return

        if k.dtype != k_prev.dtype or v.dtype != v_prev.dtype:
            raise ValueError("Appended K/V dtype must match existing cached tensors")
        if k.device != k_prev.device or v.device != v_prev.device:
            raise ValueError("Appended K/V device must match existing cached tensors")

        if self.qkv_layout == AttnQKVLayout.BSHD:
            if k_prev.dim() != 4 or k.dim() != 4:
                raise ValueError("K must be 4D for BSHD layout")
            if k_prev.shape[0] != k.shape[0] or k_prev.shape[2:] != k.shape[2:]:
                raise ValueError("Batch and head/hidden dims of appended K must match cache for BSHD layout")
            if v_prev.shape[0] != v.shape[0] or v_prev.shape[2:] != v.shape[2:]:
                raise ValueError("Batch and head/hidden dims of appended V must match cache for BSHD layout")
            k_new = torch.cat([k_prev, k], dim=1)
            v_new = torch.cat([v_prev, v], dim=1)
            self._cache[layer_idx] = (k_new, v_new, None)

        elif self.qkv_layout == AttnQKVLayout.SBHD:
            if k_prev.dim() != 4 or k.dim() != 4:
                raise ValueError("K must be 4D for SBHD layout")
            if k_prev.shape[1:] != k.shape[1:]:
                raise ValueError("Head/hidden dims of appended K must match cache for SBHD layout")
            if v_prev.shape[1:] != v.shape[1:]:
                raise ValueError("Head/hidden dims of appended V must match cache for SBHD layout")
            k_new = torch.cat([k_prev, k], dim=0)
            v_new = torch.cat([v_prev, v], dim=0)
            self._cache[layer_idx] = (k_new, v_new, None)

        elif self.qkv_layout == AttnQKVLayout.THD:
            if cu_seqlens is None:
                raise ValueError("cu_seqlens must be provided when appending with THD layout")
            if k_prev.dim() != 3 or k.dim() != 3:
                raise ValueError("K must be 3D for THD layout")
            if k_prev.shape[1:] != k.shape[1:]:
                raise ValueError("Head/hidden dims of appended K must match cache for THD layout")
            if v_prev.shape[1:] != v.shape[1:]:
                raise ValueError("Head/hidden dims of appended V must match cache for THD layout")
            if cu_prev is None:
                raise ValueError("Existing cu_seqlens must not be None for THD layout")
            if cu_prev.shape != cu_seqlens.shape:
                raise ValueError("Appended cu_seqlens shape must match existing cu_seqlens for THD layout")

            k_new = torch.cat([k_prev, k], dim=0)
            v_new = torch.cat([v_prev, v], dim=0)

            old_lens = cu_prev[1:] - cu_prev[:-1]
            new_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            if not torch.all(old_lens >= 0) or not torch.all(new_lens >= 0):
                raise ValueError("cu_seqlens must be non-decreasing for THD layout")
            total_lens = old_lens + new_lens
            cu_new = torch.zeros_like(cu_prev)
            cu_new[1:] = total_lens.cumsum(0)

            self._cache[layer_idx] = (k_new, v_new, cu_new)

        else:
            raise ValueError(f"Unsupported qkv_layout: {self.qkv_layout}")

    def reset(self) -> None:
    
        self._cache.clear()

"""
PROPER BATCHED FLASH ATTENTION
==============================

The issue: flash_attn_varlen_func is designed for variable-length sequences packed together.
Even when all sequences have the same length, it doesn't give true batched parallelism.

The fix: Use flash_attn.flash_attn_func which is the TRUE batched attention function.
It takes input shape [batch_size, seqlen, nheads, headdim] and processes all batches in parallel.

Replace your wan/modules/attention.py with this file.
"""

import torch
import warnings

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    from flash_attn import flash_attn_func  # THE BATCHED VERSION!
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    flash_attn_func = None

__all__ = [
    'flash_attention',
    'flash_attention_batched',
    'attention',
]


def flash_attention_batched(
    q,
    k,
    v,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    TRUE batched flash attention using flash_attn_func.
    
    This is the efficient version for uniform-length batches.
    All samples in the batch are processed in parallel on the GPU.
    
    Args:
        q: [B, Lq, Nq, C] - query tensor
        k: [B, Lk, Nk, C] - key tensor  
        v: [B, Lk, Nk, C] - value tensor
        dropout_p: dropout probability
        softmax_scale: scaling factor (default: 1/sqrt(head_dim))
        causal: whether to use causal masking
        window_size: sliding window attention (if supported)
        deterministic: use deterministic algorithm
        dtype: compute dtype (float16 or bfloat16)
        
    Returns:
        output: [B, Lq, Nq, C]
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda'
    
    out_dtype = q.dtype
    
    # Convert to half precision if needed
    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype not in half_dtypes:
        k = k.to(dtype)
    if v.dtype not in half_dtypes:
        v = v.to(dtype)
    
    # Ensure same dtype
    k = k.to(q.dtype)
    v = v.to(q.dtype)
    
    # Use the BATCHED flash attention function
    # This processes all B samples in parallel!
    if FLASH_ATTN_2_AVAILABLE and flash_attn_func is not None:
        # flash_attn_func signature: (q, k, v, dropout_p, softmax_scale, causal, window_size, ...)
        # Input shapes: [batch_size, seqlen, nheads, headdim]
        
        # Check if window_size is supported (FA2 >= 2.1)
        try:
            out = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        except TypeError:
            # Older version without window_size/deterministic
            out = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )
    else:
        # Fallback to PyTorch SDPA
        # Need to transpose for SDPA: [B, L, N, D] -> [B, N, L, D]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        out = out.transpose(1, 2).contiguous()
    
    return out.to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Flash attention with automatic batched vs varlen selection.
    
    When q_lens and k_lens are None (uniform lengths), uses the efficient
    batched flash_attn_func. Otherwise falls back to flash_attn_varlen_func.
    
    Args:
        q:              [B, Lq, Nq, C1].
        k:              [B, Lk, Nk, C1].
        v:              [B, Lk, Nk, C2].
        q_lens:         [B] or None. If None, assumes uniform length Lq.
        k_lens:         [B] or None. If None, assumes uniform length Lk.
        ...
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Apply q_scale if provided
    if q_scale is not None:
        q = q * q_scale

    # === FAST PATH: Uniform lengths - use batched attention ===
    if q_lens is None and k_lens is None:
        # TRUE BATCHED ATTENTION - this is the efficient path!
        q = half(q)
        k = half(k)
        v = half(v)
        
        # Use flash_attn_func for batched attention
        if FLASH_ATTN_2_AVAILABLE and flash_attn_func is not None:
            try:
                x = flash_attn_func(
                    q, k, v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                )
            except TypeError:
                # Fallback for older flash_attn versions
                x = flash_attn_func(
                    q, k, v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            return x.to(out_dtype)
        else:
            # Fallback to PyTorch SDPA
            q_t = half(q).transpose(1, 2)
            k_t = half(k).transpose(1, 2)
            v_t = half(v).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            return out.transpose(1, 2).contiguous().to(out_dtype)
    
    # === SLOW PATH: Variable lengths - use varlen function ===
    # This path has Python loops for packing sequences
    
    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # Compute cumulative sequence lengths
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Main attention entry point."""
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. '
                'It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
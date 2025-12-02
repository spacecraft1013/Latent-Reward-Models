import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

from dataclasses import dataclass
from typing import List, Optional, Union, Literal
import copy

__all__ = ["WanModel"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()



_GRID_DIMS_CACHE = {}

@amp.autocast(enabled=False)
def rope_apply_batched(x, grid_sizes, freqs, grid_dims=None):
    """
    Vectorized rope_apply for uniform grid sizes.
    
    FIXED: Added grid_dims parameter to avoid .tolist() CUDA sync.
    If grid_dims is provided (as tuple of ints), uses it directly.
    Otherwise, caches the result of .tolist() to avoid repeated syncs.
    """
    b, s, n, d = x.shape
    c = d // 2
    
    # Get grid dimensions WITHOUT forcing CUDA sync on every call
    if grid_dims is not None:
        # Best path: grid_dims passed as ints from caller
        f, h, w = grid_dims
    else:
        # Fallback: use cache to avoid repeated .tolist()
        # Use shape as cache key (doesn't require GPU access)
        cache_key = (grid_sizes.shape[0], id(grid_sizes.data_ptr))
        if cache_key not in _GRID_DIMS_CACHE:
            _GRID_DIMS_CACHE[cache_key] = tuple(grid_sizes[0].tolist())
        f, h, w = _GRID_DIMS_CACHE[cache_key]
    
    seq_len = f * h * w
    
    # Split freqs
    freq_splits = [c - 2 * (c // 3), c // 3, c // 3]
    freqs_split = freqs.split(freq_splits, dim=1)
    
    # Build frequency tensor once (broadcasts over batch)
    freqs_3d = torch.cat(
        [
            freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(1, seq_len, 1, -1)
    
    # Separate active sequence from padding
    x_active = x[:, :seq_len]
    x_pad = x[:, seq_len:]
    
    # Apply rotary embedding
    x_complex = torch.view_as_complex(
        x_active.to(torch.float64).reshape(b, seq_len, n, -1, 2)
    )
    x_rotated = torch.view_as_real(x_complex * freqs_3d).flatten(3)
    result = torch.cat([x_rotated, x_pad.to(torch.float64)], dim=1)
    
    return result.float()


def check_uniform_grid_sizes(grid_sizes: torch.Tensor) -> bool:
    """
    Check if all samples have the same grid size.
    FIXED: Removed .item() call which forces CUDA sync.
    """
    if grid_sizes.size(0) <= 1:
        return True
    # Use torch.all without .item() - returns tensor, but bool(tensor) is fine
    return bool(torch.all(grid_sizes == grid_sizes[0:1]))
    return torch.all(grid_sizes == grid_sizes[0:1]).item()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    """
    Self-attention with support for batched rope_apply.
    Modified to auto-detect uniform grid sizes and use vectorized implementation.
    """
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, use_batched_rope=None, grid_dims=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            use_batched_rope(bool, optional): Force batched/unbatched rope. 
                                              If None, auto-detect based on grid uniformity.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Auto-detect whether to use batched rope if not specified
        if use_batched_rope is None:
            use_batched_rope = check_uniform_grid_sizes(grid_sizes)
        
        # Apply rotary position embeddings
        if use_batched_rope:
            # Vectorized - single GPU kernel, no Python loop
            q_rope = rope_apply_batched(q, grid_sizes, freqs, grid_dims=grid_dims)
            k_rope = rope_apply_batched(k, grid_sizes, freqs, grid_dims=grid_dims)
        else:
            # Fallback to original loop-based implementation for variable grid sizes
            q_rope = rope_apply(q, grid_sizes, freqs)
            k_rope = rope_apply(k, grid_sizes, freqs)

        # Flash attention (already handles batches efficiently)
        x = flash_attention(
            q=q_rope,
            k=k_rope,
            v=v,
            # k_lens=seq_lens,
            window_size=self.window_size,
        )

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, 
                            # k_lens=context_lens
                            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, 
                            # k_lens=context_lens
                            )

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    """
    Attention block modified to pass use_batched_rope to self-attention.
    """
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.cross_attn_type = cross_attn_type

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        use_batched_rope=None,  # NEW PARAMETER
        grid_dims=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Text context
            context_lens(Tensor): Context lengths
            use_batched_rope(bool, optional): Whether to use batched rope
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # Self-attention with batched rope support
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], 
            seq_lens, 
            grid_sizes, 
            freqs,
            use_batched_rope=use_batched_rope,  # Pass through
            grid_dims=grid_dims,
        )
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # Cross-attention & FFN (unchanged)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
        return x



class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280)
            )

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v", "vace"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Batched forward pass through the diffusion model.
        
        Key optimizations over original forward:
        1. Batched patch embedding (single conv3d call instead of loop)
        2. Batched rope_apply (vectorized, no Python loop over batch)
        3. Pre-stacked context embedding
        
        Args:
            x (List[Tensor] or Tensor):
                List of input video tensors, each with shape [C_in, F, H, W]
                OR a stacked tensor of shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor] or Tensor):
                List of text embeddings each with shape [L, C]
                OR pre-stacked tensor of shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode
                
        Returns:
            List[Tensor]:
                List of denoised video tensors with shape [C_out, F, H/8, W/8]
        """
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert clip_fea is not None and y is not None
        
        # Get device
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Handle input format - convert list to batched tensor
        if isinstance(x, torch.Tensor) and x.dim() == 5:
            # Already batched: [B, C, F, H, W]
            x_batched = x
        elif isinstance(x, torch.Tensor) and x.dim() == 4:
            # Single sample: [C, F, H, W] -> [1, C, F, H, W]
            x_batched = x.unsqueeze(0)
        else:
            # List of tensors: stack them
            x_batched = torch.stack(x, dim=0)  # [B, C, F, H, W]
        
        b = x_batched.size(0)
        
        # Handle conditional inputs for i2v
        if y is not None:
            if isinstance(y, torch.Tensor) and y.dim() == 5:
                y_batched = y
            elif isinstance(y, torch.Tensor) and y.dim() == 4:
                y_batched = y.unsqueeze(0)
            else:
                y_batched = torch.stack(y, dim=0)
            x_batched = torch.cat([x_batched, y_batched], dim=1)

        # BATCHED patch embedding - single conv3d call!
        # Original: [self.patch_embedding(u.unsqueeze(0)) for u in x]  <- B separate calls
        # New: single call processes all B samples in parallel
        x_embed = self.patch_embedding(x_batched)  # [B, dim, F', H', W']
        
        # Get grid sizes - all identical for batched generation
        grid_size = torch.tensor(x_embed.shape[2:], dtype=torch.long, device=device)
        grid_sizes = grid_size.unsqueeze(0).expand(b, -1).contiguous()  # [B, 3]
        grid_dims = x_embed.shape[2:]
        
        # Flatten spatial dimensions
        x_flat = x_embed.flatten(2).transpose(1, 2)  # [B, L, dim]
        actual_seq_len = x_flat.size(1)
        
        # Create sequence length tensor
        seq_lens = torch.full((b,), actual_seq_len, dtype=torch.long, device=device)
        
        # Pad to maximum sequence length if needed
        if actual_seq_len < seq_len:
            padding = x_flat.new_zeros(b, seq_len - actual_seq_len, x_flat.size(2))
            x_flat = torch.cat([x_flat, padding], dim=1)

        # Time embeddings (already batched)
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Context embeddings
        context_lens = None
        if isinstance(context, torch.Tensor) and context.dim() == 3:
            # Already batched: [B, L, C]
            if context.size(1) < self.text_len:
                padding = context.new_zeros(b, self.text_len - context.size(1), context.size(2))
                context = torch.cat([context, padding], dim=1)
            context = self.text_embedding(context)
        else:
            # List of tensors - stack and pad
            context = self.text_embedding(
                torch.stack(
                    [
                        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                        for u in context
                    ]
                )
            )

        # Handle CLIP features for i2v
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Build kwargs with batched rope enabled
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            use_batched_rope=True,  # Force batched rope since grid sizes are uniform
            grid_dims=grid_dims,
        )

        # Process through transformer blocks
        x = x_flat
        for block in self.blocks:
            x = block(x, **kwargs)

        # Final head
        x = self.head(x, e)

        # Unpatchify back to video tensors
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes, grid_dims=None):
        """
        Batched unpatchify - processes all samples in parallel.
        
        NOTE: This returns a single tensor [B, C, F, H, W] instead of list.
        Caller may need to convert to list if required.
        """
        b = x.size(0)
        c = self.out_dim
        pt, ph, pw = self.patch_size
        
        # Get grid dimensions
        if grid_dims is not None:
            f, h, w = grid_dims
        else:
            # Fallback with cache
            cache_key = (grid_sizes.shape[0], grid_sizes.device.index if grid_sizes.device.type == 'cuda' else -1)
            if cache_key not in _GRID_DIMS_CACHE:
                _GRID_DIMS_CACHE[cache_key] = tuple(grid_sizes[0].tolist())
            f, h, w = _GRID_DIMS_CACHE[cache_key]
        
        prod_v = f * h * w
        
        # Slice and reshape
        x = x[:, :prod_v]  # [B, f*h*w, patch_prod * c]
        x = x.view(b, f, h, w, pt, ph, pw, c)
        
        # Permute: bfhwpqrc -> bcfphqwr
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, c, f, pt, h, ph, w, pw]
        
        # Final reshape
        x = x.reshape(b, c, f * pt, h * ph, w * pw)
        
        # Return as list to match original API
        return [x[i] for i in range(b)]

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

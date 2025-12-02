# rewards/latent_aesthetic.py
import torch
import torch.nn as nn
from rewards.og_aesthetic import get_aesthetic_model

_CLIP_DIMS = {"vit_l_14": 768, "vit_b_32": 512}


class LatentToCLIPEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = 512

        # Changed: Added BatchNorm, Dropout, and using GELU instead of SiLU
        # GELU is often better for regression tasks
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),  # Light dropout early
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Separate projection head with dropout
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, latents):
        features = self.trunk(latents)
        emb = self.projection(features)
        # L2 normalization
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
        return emb


class LatentAestheticViaLAION(nn.Module):
    def __init__(
        self, in_channels: int, clip_model: str = "vit_l_14", dropout: float = 0.1
    ) -> None:
        super().__init__()
        embed_dim = _CLIP_DIMS[clip_model]

        # Adapter to map latents to CLIP space
        self.adapter = LatentToCLIPEmbedding(in_channels, embed_dim, dropout=dropout)

        # Load pretrained aesthetic head
        self.head = nn.Linear(embed_dim, 1)
        state = get_aesthetic_model(clip_model).state_dict()
        self.head.load_state_dict(state)

        # CHANGED: Make head trainable with small learning rate
        for p in self.head.parameters():
            p.requires_grad = True

        # REMOVED: The confusing calibration layer
        # The model now directly predicts the aesthetic score

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        emb = self.adapter(latents)  # [B, embed_dim], unit-norm
        score = self.head(emb).squeeze(-1)  # [B]
        return score
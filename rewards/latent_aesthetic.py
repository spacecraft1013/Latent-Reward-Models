import torch
import torch.nn as nn
from rewards.og_aesthetic import get_aesthetic_model

_CLIP_DIMS = {"vit_l_14": 768, "vit_b_32": 512}


class LatentToCLIPEmbedding(nn.Module):
    """
    Map (B,C,H,W) latents to CLIP-sized embedding with L2 normalization.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        emb = self.trunk(latents)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
        return emb


class LatentAestheticViaLAION(nn.Module):
    """
    Trainable adapter (latents->CLIP embedding) + frozen official LAION linear head.
    """

    def __init__(self, in_channels: int, clip_model: str = "vit_l_14") -> None:
        super().__init__()
        embed_dim = _CLIP_DIMS[clip_model]
        self.adapter = LatentToCLIPEmbedding(in_channels, embed_dim)
        self.head = nn.Linear(embed_dim, 1)
        state = get_aesthetic_model(clip_model).state_dict()
        self.head.load_state_dict(state)
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        emb = self.adapter(latents)
        return self.head(emb).squeeze(-1)



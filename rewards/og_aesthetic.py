import os
from os.path import expanduser
from urllib.request import urlretrieve
from typing import List

import torch
import torch.nn as nn
import open_clip


def get_aesthetic_model(clip_model: str = "vit_l_14") -> nn.Linear:
    """
    https://github.com/LAION-AI/aesthetic-predictor
    """
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m


class AestheticFromPixels(nn.Module):
    """
    image -> open_clip.encode_image -> L2 norm -> LAION linear head -> score
    Matches the LAION example flow.
    """
    def __init__(self, clip_model: str = "vit_l_14", device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.device = device
        model_name = "ViT-L-14" if clip_model == "vit_l_14" else "ViT-B-32"
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        self.clip = self.clip.to(device).eval()
        self.head = get_aesthetic_model(clip_model).to(device)
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B,3,H,W) in [0,1]
        b = images.size(0)
        items: List[torch.Tensor] = []
        for i in range(b):
            items.append(self.preprocess(images[i].cpu()))
        imgs = torch.stack(items, dim=0).to(self.device)

        image_features = self.clip.encode_image(imgs)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        pred = self.head(image_features).squeeze(-1)
        return pred


def load_og_aesthetic_model(device: torch.device, clip_model: str = "vit_l_14") -> nn.Module:
    return AestheticFromPixels(clip_model=clip_model, device=device).eval()



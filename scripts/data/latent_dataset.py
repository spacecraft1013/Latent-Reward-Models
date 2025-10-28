
"""
- do a sanity check to make sure the latents are loaded correctly
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def default_image_loader(path: str, image_size: Optional[int] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


# loads latents from the directory, yields (image_id, latent_tensor)
class LatentOnlyDataset(Dataset):
    """
    Iterates over latent files and yields (image_id, latent_tensor).

    Assumptions:
    - Latent directory contains files with stems that act as IDs (e.g., 000123.pt or 000123.npy)
    - Supported latent formats: .pt (torch tensor), .npy/.npz (numpy arrays)
    - Optionally can filter IDs via an allowlist (ids present in a mapping file of precomputed scores)
    """

    def __init__(
        self,
        latents_dir: str,
        allowed_ids: Optional[set] = None,
        latent_key: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.latents_dir = Path(latents_dir)
        self.latent_key = latent_key
        self.dtype = dtype

        all_files: List[Path] = []
        for ext in ("*.pt", "*.pth", "*.npy", "*.npz"):
            all_files.extend(sorted(self.latents_dir.glob(ext)))

        if allowed_ids is not None:
            filtered = []
            for f in all_files:
                _id = f.stem
                if _id in allowed_ids:
                    filtered.append(f)
            self.files = filtered
        else:
            self.files = all_files

        if len(self.files) == 0:
            raise FileNotFoundError(f"No latent files found in {self.latents_dir}")

    def __len__(self) -> int:
        return len(self.files)

    # loads a latent tensor from a file and returns it as a tensor
    def _load_latent(self, path: Path) -> torch.Tensor:
        if path.suffix in (".pt", ".pth"):
            obj = torch.load(path, map_location="cpu").squeeze()
            if isinstance(obj, torch.Tensor):
                lat = obj
            elif isinstance(obj, dict):
                key = self.latent_key or next(iter(obj.keys()))
                lat = obj[key]
            else:
                raise ValueError(f"Unsupported .pt content type in {path}: {type(obj)}")
        elif path.suffix == ".npy":
            lat = torch.from_numpy(np.load(path))
        elif path.suffix == ".npz":
            npz = np.load(path)
            key = self.latent_key or list(npz.files)[0]
            lat = torch.from_numpy(npz[key])
        else:
            raise ValueError(f"Unsupported latent format: {path.suffix}")

        return lat.to(dtype=self.dtype)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.files[idx]
        image_id = path.stem
        latent = self._load_latent(path)
        return image_id, latent


# loads images from the directory, yields (image_id, image_tensor)
class ImageFolderWithIds(Dataset):
    """
    Iterates over images and yields (image_id, image_tensor).
    The image_id is derived from filename stem.
    """

    def __init__(
        self,
        images_dir: str,
        image_size: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transform = transform
        self.files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            self.files.extend(sorted(self.images_dir.glob(ext)))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.files[idx]
        image_id = path.stem
        try:
            tensor = default_image_loader(str(path), self.image_size)
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {e}")
        if self.transform is not None:
            tensor = self.transform(tensor)
        return image_id, tensor



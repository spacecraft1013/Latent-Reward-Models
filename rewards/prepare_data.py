import os
import json
import torch
import torch.nn as nn
import numpy as np
import io
import math
from os.path import expanduser
from urllib.request import urlretrieve
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

# External library
import open_clip

# Your local VAE module
from wan.modules.vae import WanVAE

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "hf_dataset": "tomg-group-umd/pixelprose",
    "split": "redcaps",
    "target_count": 100_000,
    # Paths
    "output_dir": "data",
    "vae_checkpoint": "/fs/nexus-projects/mt_sec/img_noise_opt/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    # Image Processing
    "target_resolution": (1280, 720),  # (W, H)
    "vae_stride": 16,  # Ensure dims are divisible by this
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_images": True,  # Save .webp copies for visualization
}


# -----------------------------------------------------------------------------
# AESTHETIC MODEL CLASSES (Integrated from og_aesthetic.py)
# -----------------------------------------------------------------------------
def get_aesthetic_model(clip_model="vit_l_14"):
    """Downloads and loads the linear layer for aesthetic scoring."""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"

    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        print(f"Downloading aesthetic head to {path_to_model}...")
        urlretrieve(url_model, path_to_model)

    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError(f"Unknown model {clip_model}")

    s = torch.load(path_to_model, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m


class AestheticScorer(nn.Module):
    """Encodes image with CLIP and applies the aesthetic linear head."""

    def __init__(self, clip_model="vit_l_14", device="cpu"):
        super().__init__()
        self.device = device
        model_name = "ViT-L-14" if clip_model == "vit_l_14" else "ViT-B-32"

        print(f"Loading CLIP {model_name}...")
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.clip = self.clip.to(device).eval()

        print("Loading Aesthetic Head...")
        self.head = get_aesthetic_model(clip_model).to(device)

    @torch.no_grad()
    def score(self, image_pil: Image.Image) -> float:
        """Helper to process a single PIL image and return float score."""
        # Use OpenCLIP's preprocessing
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        # Encode
        features = self.clip.encode_image(image_tensor)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)

        # Score
        score = self.head(features).squeeze().item()
        return score


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def smart_resize_and_crop(
    img: Image.Image, target_w: int, target_h: int
) -> Image.Image:
    """
    Resizes image to cover target dims (maintaining aspect ratio), then crops.
    Prevents 'squishing' artifacts.
    """
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(math.ceil(orig_w * scale))
    new_h = int(math.ceil(orig_h * scale))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return img.crop((left, top, right, bottom))


def setup_directories():
    os.makedirs(os.path.join(CONFIG["output_dir"], "latents"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "images"), exist_ok=True)


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    setup_directories()
    device = torch.device(CONFIG["device"])

    # 1. Initialize Models
    print(">>> Initializing Models...")

    # Aesthetic Scorer
    scorer = AestheticScorer(clip_model="vit_l_14", device=device)

    # WanVAE
    print(f"Loading WanVAE from {CONFIG['vae_checkpoint']}...")
    vae = WanVAE(vae_pth=CONFIG["vae_checkpoint"], device=device)

    # Transforms for VAE
    vae_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    # 2. Setup Dataset Stream
    print(f">>> Streaming from {CONFIG['hf_dataset']}...")
    dataset = load_dataset(CONFIG["hf_dataset"], split=CONFIG["split"], streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    # 3. Load Existing State
    scores_dict = {}
    json_path = os.path.join(CONFIG["output_dir"], "scores.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            scores_dict = json.load(f)
        print(f"Resuming from {len(scores_dict)} previously processed images.")

    # 4. Processing Loop
    success_count = 0
    target_w, target_h = CONFIG["target_resolution"]

    # Initialize progress bar
    pbar = tqdm(total=CONFIG["target_count"], initial=len(scores_dict))

    for row in shuffled_dataset:
        if len(scores_dict) >= CONFIG["target_count"]:
            break

        # Generate a unique ID if not present
        uid = row.get("uid", str(len(scores_dict) + success_count))

        if uid in scores_dict:
            continue

        try:
            # --- A. Load Image ---
            if "image" in row and isinstance(row["image"], Image.Image):
                img_pil = row["image"]
            elif "url" in row:
                import urllib.request

                with urllib.request.urlopen(row["url"], timeout=5) as url:
                    f = io.BytesIO(url.read())
                img_pil = Image.open(f)
            else:
                continue

            img_pil = img_pil.convert("RGB")

            # Skip small images
            if img_pil.width < 512 or img_pil.height < 512:
                continue

            # --- B. Smart Resize (No Squish) ---
            img_processed = smart_resize_and_crop(img_pil, target_w, target_h)

            # --- C. Score Image (Aesthetic Model) ---
            score = scorer.score(img_processed)

            # --- D. Encode Image (VAE) ---
            vae_input = vae_transforms(img_processed).to(torch.float32).to(device)
            # Add dimensions: (C, H, W) -> (C, 1, H, W) -> VAE wants list of inputs
            vae_input = vae_input.unsqueeze(1)

            with torch.no_grad():
                latents_list = vae.encode([vae_input])
                latent = latents_list[0]  # Take the tensor from the list

            # --- E. Save Everything ---
            # Save Latent
            latent_path = os.path.join(CONFIG["output_dir"], "latents", f"{uid}.pt")
            torch.save(latent.clone().cpu(), latent_path)

            # Save Preview Image
            if CONFIG["save_images"]:
                img_path = os.path.join(CONFIG["output_dir"], "images", f"{uid}.webp")
                img_processed.save(img_path, "WEBP", quality=80)

            # Update Metadata
            scores_dict[uid] = score
            success_count += 1
            pbar.update(1)

            # Periodic Save
            if len(scores_dict) % 50 == 0:
                with open(json_path, "w") as f:
                    json.dump(scores_dict, f)

        except Exception as e:
            # Uncomment for debugging, otherwise keep clean
            # print(f"Error processing {uid}: {e}")
            continue

    # Final Save
    with open(json_path, "w") as f:
        json.dump(scores_dict, f)

    print(f"\nCompleted! Total processed: {len(scores_dict)}")


if __name__ == "__main__":
    main()

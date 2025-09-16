import os

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from wan.modules.vae import WanVAE

INPUTDIR = "data/inputs"

vae = WanVAE(vae_pth="./Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", device="cuda")
for filename in os.listdir(INPUTDIR):
    input_path = os.path.join(INPUTDIR, filename)
    latent_path = os.path.join("data/latents", filename.split(".")[0] + ".pt")
    recon_path = os.path.join("data/reconstructions", filename)
    input_img = Image.open(input_path).convert("RGB").resize((1280, 720))

    img_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    input_data = img_transforms(input_img).to(torch.float32).cuda()
    input_data = input_data.unsqueeze(1)

    latents = vae.encode([input_data])
    torch.save(latent_path, latents[0])

    orig = vae.decode(latents)[0].squeeze(1)

    value_range = (-1, 1)
    orig = orig.clamp(min(value_range), max(value_range))
    torchvision.utils.save_image(
        orig, recon_path, nrow=8, normalize=True, value_range=value_range
    )

    print(
        f"Processed {filename}, saved latents to {latent_path} and reconstruction to {recon_path}"
    )

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from wan.modules.vae import WanVAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--wan_model", type=str, required=True)
    args = parser.parse_args()

    vae = WanVAE(vae_pth=args.wan_model, device="cuda")
    for filename in os.listdir(args.images_dir):
        input_path = os.path.join(args.images_dir, filename)
        latent_path = os.path.join(args.latents_dir, filename.split(".")[0] + ".pt")
        input_img = Image.open(input_path).convert("RGB").resize((1280, 720))

        img_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        input_data = img_transforms(input_img).to(torch.float32).cuda()
        input_data = input_data.unsqueeze(1)

        latents = vae.encode([input_data])
        torch.save(latents[0], latent_path)

        print(f"Processed {filename}, saved latents to {latent_path}")


if __name__ == "__main__":
    main()

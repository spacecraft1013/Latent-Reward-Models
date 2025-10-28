import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import tqdm
from scripts.data.latent_dataset import ImageFolderWithIds
from rewards.og_aesthetic import load_og_aesthetic_model

# will return scores for each image in the folder
# {image_id: score}
def main() -> None:

    # this is just to be able to run the script from the command line and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_model", type=str, default="vit_l_14", choices=["vit_l_14","vit_b_32"])
    args = parser.parse_args()

    # loads images from a folder and returns (image_id, image tesor) --> found in latent_dataset.py
    ds = ImageFolderWithIds(args.images_dir, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = load_og_aesthetic_model(device, clip_model=args.clip_model)

    # run the model and collect the scores for each image
    id_to_score: Dict[str, float] = {}
    with torch.no_grad():

        for (image_ids, images) in tqdm.tqdm(dl, desc="Computing scores"):
            images = images.to(device, non_blocking=True)
            preds = model(images).detach().float().cpu()
            for _id, score in zip(image_ids, preds):
                id_to_score[_id] = float(score.item())

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(id_to_score, f)

    print(f"Wrote {len(id_to_score)} scores to {out_path}")


if __name__ == "__main__":
    main()


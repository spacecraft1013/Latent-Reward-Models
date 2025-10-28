import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.data.latent_dataset import LatentOnlyDataset
from rewards.latent_aesthetic import LatentAestheticViaLAION


def main() -> None:
    
    # again this is to just parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--scores_json", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no","fp16","bf16"])
    parser.add_argument("--clip_model", type=str, default="vit_l_14", choices=["vit_l_14","vit_b_32"])
    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.scores_json, "r") as f:
        id_to_score: Dict[str, float] = json.load(f)
    allowed_ids = set(id_to_score.keys())

    ds = LatentOnlyDataset(args.latents_dir, allowed_ids=allowed_ids)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = LatentAestheticViaLAION(in_channels=args.in_channels, clip_model=args.clip_model).to(device)
    optim = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision=="fp16"))
    use_autocast = args.mixed_precision in ("fp16","bf16")
    amp_dtype = torch.float16 if args.mixed_precision=="fp16" else (torch.bfloat16 if args.mixed_precision=="bf16" else torch.float32)

    mse = nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for image_ids, latents in dl:
            latents = latents.to(device, non_blocking=True)
            target = torch.tensor([id_to_score[_id] for _id in image_ids], dtype=torch.float32, device=device)

            optim.zero_grad(set_to_none=True)
            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred = model(latents)
                    loss = mse(pred, target)
            else:
                pred = model(latents)
                loss = mse(pred, target)

            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg = running_loss / 50
                print(f"step {global_step} | loss {avg:.4f}")
                running_loss = 0.0

        # save checkpoint each epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
        }
        torch.save(ckpt, save_dir / f"lrm_epoch{epoch}.pt")
        print(f"Saved {save_dir / f'lrm_epoch{epoch}.pt'}")


if __name__ == "__main__":
    main()



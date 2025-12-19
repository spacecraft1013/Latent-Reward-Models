# Latent Reward Models

This repository implements Latent Reward Models, a novel method leveraging efficient latent space representations of generative models (such as Diffusion models and VAEs) to compute reward scores. By operating in the latent space rather than the pixel space, this method improves efficiency by reducing the computation wasted on decoding for poor seeds.

## Usage

### Setup
- Install: `pip install -r scripts/requirements.txt`

### Data Layout
- Images: `images/*.jpg|png|...` with filenames as IDs (e.g., `12345.jpg`).
- Latents: `latents/*.pt|npy|npz` with matching stems (e.g., `12345.pt`).

### 1) Precompute original aesthetic scores
```bash
python scripts/data/precompute_scores.py \
  --images_dir images \
  --out_json og_scores.json \
  --clip_model vit_l_14 \
  --batch_size 64 --num_workers 8 --image_size 224
```

### 2) Precompute image latents
```bash
python scripts/data/generate_latents.py \
  --images_dir images \
  --latents_dir latents \
  --wan_model Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

### 3) Train latent aesthetic model with MSE(og_pred, latent_pred)
```bash
python -m train.train_lrm \
  --latents_dir latents \
  --scores_json og_scores.json \
  --in_channels 16 \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 1000 \
  --clip_model vit_l_14 \
  --save_dir checkpoints_lrm \
  --val_size 100 \
  --split_seed 123 \
  --best_metric val_pearson
```

### Notes
- Uses official LAION aesthetic predictor weights with OpenCLIP embeddings.
- Latent model learns adapter from latents to CLIP embedding space, then applies frozen LAION linear head.
- Dataloader yields `(image_id, latent_tensor)`; trainer maps `image_id -> og_pred` for targets.
- Logging prints avg loss every 50 steps; checkpoints saved per epoch.


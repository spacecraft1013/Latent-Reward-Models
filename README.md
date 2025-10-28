## Latent Reward Model (LRM) Aesthetic Consistency Training

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

### 2) Train latent aesthetic model with MSE(og_pred, latent_pred)
```bash
python train/train_lrm.py \
  --latents_dir latents \
  --scores_json og_scores.json \
  --in_channels 4 \
  --clip_model vit_l_14 \
  --batch_size 128 --lr 1e-4 --epochs 10 \
  --mixed_precision bf16 \
  --save_dir checkpoints
```

### Notes
- Uses official LAION aesthetic predictor weights with OpenCLIP embeddings.
- Latent model learns adapter from latents to CLIP embedding space, then applies frozen LAION linear head.
- Dataloader yields `(image_id, latent_tensor)`; trainer maps `image_id -> og_pred` for targets.
- Logging prints avg loss every 50 steps; checkpoints saved per epoch.



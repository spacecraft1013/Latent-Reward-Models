#!/bin/bash
# run_ablations.sh

# Base configuration
BASE_CMD="python -m train.train_lrm \
  --latents_dir data/latents \
  --scores_json data/scores.json \
  --in_channels 16 \
  --batch_size 32 \
  --epochs 200 \
  --clip_model vit_l_14 \
  --val_size 2000 \
  --split_seed 123 \
  --best_metric val_pearson \
  --num_workers 4 \
  --early_stop_patience 30"

# Experiment 1: Baseline (MSE loss, no augmentation)
echo "=== Experiment 1: Baseline ===" 
$BASE_CMD \
  --loss_fn mse \
  --lr 1e-4 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --save_dir checkpoints_lrm/exp1_baseline

# Experiment 2: With dropout regularization
echo "=== Experiment 2: Dropout Regularization ==="
$BASE_CMD \
  --loss_fn mse \
  --lr 1e-4 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --save_dir checkpoints_lrm/exp2_dropout

# Experiment 3: Huber loss (robust to outliers)
echo "=== Experiment 3: Huber Loss ==="
$BASE_CMD \
  --loss_fn huber \
  --huber_delta 1.0 \
  --lr 1e-4 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --save_dir checkpoints_lrm/exp3_huber

# Experiment 4: Smooth L1 loss
echo "=== Experiment 4: Smooth L1 Loss ==="
$BASE_CMD \
  --loss_fn smooth_l1 \
  --lr 1e-4 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --save_dir checkpoints_lrm/exp4_smoothl1

# Experiment 5: With latent augmentation
echo "=== Experiment 5: Latent Augmentation ==="
$BASE_CMD \
  --loss_fn mse \
  --lr 1e-4 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --noise_std 0.01 \
  --latent_dropout 0.05 \
  --save_dir checkpoints_lrm/exp5_augmentation

# Experiment 6: Lower learning rate with plateau scheduler
echo "=== Experiment 6: Lower LR + Plateau Scheduler ==="
$BASE_CMD \
  --loss_fn mse \
  --lr 5e-5 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --scheduler plateau \
  --save_dir checkpoints_lrm/exp6_lower_lr_plateau

# Experiment 7: Higher weight decay
echo "=== Experiment 7: Higher Weight Decay ==="
$BASE_CMD \
  --loss_fn mse \
  --lr 1e-4 \
  --dropout 0.15 \
  --weight_decay 0.05 \
  --save_dir checkpoints_lrm/exp7_high_wd

# Experiment 8: Combined best practices
echo "=== Experiment 8: Combined Best ==="
$BASE_CMD \
  --loss_fn huber \
  --huber_delta 0.5 \
  --lr 8e-5 \
  --dropout 0.12 \
  --weight_decay 0.02 \
  --noise_std 0.01 \
  --scheduler plateau \
  --save_dir checkpoints_lrm/exp8_combined

echo "All experiments completed!"
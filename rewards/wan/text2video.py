# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import copy
from contextlib import contextmanager
from functools import partial
import time
import numpy as np
from typing import Callable

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import cache_image
from typing import Optional, List
import math, gc, sys, time, random
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple
import torch.nn.functional as F


class WanT2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        args_config=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.args_config = args_config

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        self.model = WanModel.from_pretrained(checkpoint_dir)

        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            # This will be applied after block manipulation
            self.use_usp = True
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.use_usp = False
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        # Move model to device BEFORE block manipulation
        if not dit_fsdp:
            self.model.to(self.device)

        print(f"Total blocks in model: {len(self.model.blocks)}")

        # Apply USP modifications if needed (after block manipulation)
        if self.use_usp:
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.model.forward = types.MethodType(usp_dit_forward, self.model)

        # Apply FSDP if needed (after all modifications)
        if dit_fsdp:
            self.model = shard_fn(self.model)

        self.sample_neg_prompt = config.sample_neg_prompt

        logging.info(
            f"Model initialized with {len(self.model.blocks)} blocks after manipulation"
        )

    def generate(
        self,
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        start_time = time.time()
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]  ## [torch.Size([16, 21, 60, 104])]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            self.set_timestep = timesteps
            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}
            self.model.to(self.device)

            for t_idx, t in enumerate(tqdm(timesteps)):
                x_t = latents[0]  # keep current x_t before stepping
                timestep = torch.stack([t])

                noise_pred_cond = self.model([x_t], t=timestep, **arg_c)[0]

                noise_pred_uncond = self.model(
                    [x_t],
                    t=timestep,
                    **arg_null,
                )[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                out = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    x_t.unsqueeze(0),
                    return_dict=True,
                    generator=seed_g,
                )

                # always advance the chain
                latents = [out.prev_sample.squeeze(0)]

                # >>> robust x0 estimate for this step (works even if you break early)
                if (
                    hasattr(out, "pred_original_sample")
                    and out.pred_original_sample is not None
                ):
                    x0_hat = out.pred_original_sample.squeeze(0).detach()
                else:
                    # common for flow/EDM-style sigmas: x = x0 + sigma * eps  =>  x0 = x - sigma * eps
                    if hasattr(sample_scheduler, "sigmas"):
                        sigma_t = sample_scheduler.sigmas[t_idx]
                        x0_hat = (x_t - sigma_t * noise_pred).detach()
                    else:
                        # fallback: if we reached the last step, prev_sample Ã¢â€°Ë† x0; otherwise this is x_{t-1}
                        x0_hat = latents[0]

            if self.rank == 0:
                vae_decode_start_time = time.time()

            x0 = [x0_hat]
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)
                vae_decode_end_time = time.time()

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        end_time = time.time()

        return {
            "media_asset": videos[0] if self.rank == 0 else None,
            "vae_runtime": vae_decode_end_time - vae_decode_start_time,
            "runtime": end_time - start_time,
        }
    
    ## NOTE: Add batching support for the base generate function
    def generate_batch(
        self,
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed_list=[1, 2],
        offload_model=True,
    ):
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Normalize seeds
        seed_list = [s if s >= 0 else random.randint(0, sys.maxsize) for s in seed_list]
        B = len(seed_list)
        cmm_nfes = 0

        # ----- Encode text once -----
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            ctx_once = self.text_encoder([input_prompt], self.device)[0]  # [L, text_dim]
            ctx_null_once = self.text_encoder([n_prompt], self.device)[0]  # [L, text_dim]
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            ctx_once = self.text_encoder([input_prompt], torch.device("cpu"))[0]
            ctx_null_once = self.text_encoder([n_prompt], torch.device("cpu"))[0]
            ctx_once = ctx_once.to(self.device)
            ctx_null_once = ctx_null_once.to(self.device)

        # PRE-STACK context for batched processing
        # Instead of: context = ctx_once * B  (list replication)
        # Do: stack into [B, L, text_dim] tensor
        context_stacked = ctx_once.unsqueeze(0).expand(B, -1, -1).contiguous()
        context_null_stacked = ctx_null_once.unsqueeze(0).expand(B, -1, -1).contiguous()

        # ----- Initial noise per sample (as batched tensor) -----
        latents = torch.stack(
            [
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=torch.Generator(device=self.device).manual_seed(s),
                )
                for s in seed_list
            ],
            dim=0,
        )  # [B, C, Fp, Hp, Wp]

        # One generator per sample (for scheduler noise injection)
        gens = [torch.Generator(device=self.device).manual_seed(s) for s in seed_list]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # Move model to device once
        self.model.to(self.device)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # ----- Initialize Scheduler -----
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            # ----- Sampling loop -----
            for _, t in enumerate(tqdm(timesteps)):
                # Batch timesteps: shape [B]
                t_b = torch.full(
                    (B,),
                    t.item() if isinstance(t, torch.Tensor) else t,
                    device=self.device,
                    dtype=timesteps.dtype,
                )

                # Conditional prediction (batched)
                eps_c_list = self.model.forward(
                    latents,  # [B, C, Fp, Hp, Wp] tensor directly
                    t=t_b,
                    context=context_stacked,  # [B, L, text_dim] tensor
                    seq_len=seq_len,
                )
                cmm_nfes += 1
                
                # Unconditional prediction (batched)
                eps_u_list = self.model.forward(
                    latents,
                    t=t_b,
                    context=context_null_stacked,  # [B, L, text_dim] tensor
                    seq_len=seq_len,
                )
                cmm_nfes += 1
                
                # Stack outputs and apply CFG
                eps_c = torch.stack(eps_c_list, dim=0)  # [B, C, ...]
                eps_u = torch.stack(eps_u_list, dim=0)  # [B, C, ...]
                eps = eps_u + guide_scale * (eps_c - eps_u)  # [B, C, ...]

                # Scheduler step on the whole batch
                latents = sample_scheduler.step(
                    eps, t, latents, return_dict=False, generator=gens
                )[0]  # [B, C, Fp, Hp, Wp]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                # VAE expects list-of-latents
                x0_list = [latents[i] for i in range(B)]
                videos = self.vae.decode(x0_list)

        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return {"media_assets": videos if self.rank == 0 else None, "cmm_nfes": cmm_nfes}



    def generate_zero_order_search(
        self,
        input_prompt,
        pivot_latent=None,
        neighborhood_size=4,
        neighborhood_lambda=0.1,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        return_latents=False,
    ):
        cmm_nfes = 0
        # 1. Preprocess & Setup Dimensions
        start_time = time.time()
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # 2. Text Encoding (Done once, then replicated)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            ctx_once = self.text_encoder([input_prompt], self.device)
            ctx_null_once = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            ctx_once = self.text_encoder([input_prompt], torch.device("cpu"))
            ctx_null_once = self.text_encoder([n_prompt], torch.device("cpu"))
            ctx_once = [t.to(self.device) for t in ctx_once]
            ctx_null_once = [t.to(self.device) for t in ctx_null_once]

        # 3. Latent Generation (Pivot & Neighbors)
        # Generate pivot if not provided
        if pivot_latent is None:
            seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)
            pivot_latent = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        
        # Generate neighbors -> This creates a batch of latents (N, C, T, H, W)
        # Note: _generate_neighborhood returns (neighborhood_size-1) neighbors. 
        # If you strictly want the batch size to match neighborhood_size, we use the output directly.
        latents = self._generate_neighborhood(
            pivot_latent, neighborhood_size, neighborhood_lambda
        )
        
        # If the neighborhood generation returns 0 candidates (e.g. size=1), we process just the pivot (edge case)
        if latents.shape[0] == 0:
            latents = pivot_latent.unsqueeze(0)

        # Batch Size
        B = latents.shape[0]
        
        # Replicate context for the batch
        context = ctx_once * B
        context_null = ctx_null_once * B

        # Keep a CPU copy if requested for return
        used_latents = [l.cpu() for l in latents] if return_latents else []

        # 4. Prepare Scheduler
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # 5. Batched Diffusion Loop
        self.model.to(self.device)
        
        # Create dummy generators for the batch (required by some scheduler steps)
        # Since ZOS usually relies on the specific latent perturbations, the generator here 
        # is mostly for scheduler internal randomness (if any).
        gens = [torch.Generator(device=self.device) for _ in range(B)]

        videos = []
        
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Init Scheduler
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            # Denoising
            for _, t in enumerate(tqdm(timesteps, desc="ZOS Diffusion")):
                # Broadcast timestep to batch
                t_b = torch.full(
                    (B,),
                    t.item() if isinstance(t, torch.Tensor) else t,
                    device=self.device,
                    dtype=timesteps.dtype,
                )

                # WanModel expects a LIST of latents
                latent_list = [latents[i] for i in range(B)]

                # Forward pass (Batched via list comprehension inside model, or internal logic)
                # We handle the stacking here to perform CFG in batch
                eps_c_list = self.model(
                    latent_list, t=t_b, context=context, seq_len=seq_len
                )
                cmm_nfes += 1
                
                eps_u_list = self.model(
                    latent_list, t=t_b, context=context_null, seq_len=seq_len
                )
                cmm_nfes += 1
                
                # Stack results: List[Tensor] -> Tensor [B, C, ...]
                eps_c = torch.stack(eps_c_list, dim=0)
                eps_u = torch.stack(eps_u_list, dim=0)

                # CFG
                eps = eps_u + guide_scale * (eps_c - eps_u)

                # Scheduler Step (Batched)
                # scheduler.step usually handles batched inputs if 'latents' is a tensor
                latents = sample_scheduler.step(
                    eps, t, latents, return_dict=False, generator=gens
                )[0]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            # 6. Batched VAE Decoding
            if self.rank == 0:
                # Convert tensor back to list of latents for VAE
                x0_list = [latents[i] for i in range(B)]
                # Decode all at once (or relies on VAE internal handling)
                videos_tensor = self.vae.decode(x0_list)
                
                # WanVAE usually returns a tensor or list. 
                # Ensure we return a LIST of videos to match original API expectations.
                if isinstance(videos_tensor, torch.Tensor):
                    # Unbind [B, C, T, H, W] -> List of [C, T, H, W]
                    videos = [v for v in videos_tensor]
                else:
                    videos = videos_tensor

        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        end_time = time.time()

        result = {
            "videos": videos if self.rank == 0 else None,
            "runtime": end_time - start_time,
            "cmm_nfes": cmm_nfes,
        }
        if return_latents:
            result["latents"] = used_latents

        return result

    def generate_neighbors_spherical_cap_unbatched(
        self,
        x: torch.Tensor,
        neighborhood_size: int,
        *,
        threshold: float = None,           # preferred: cosine similarity in (-1,1)
        lambda_dist: float = None,         # optional: Euclidean distance; auto-mapped to threshold
        generator: torch.Generator = None, # optional RNG for reproducibility
    ) -> torch.Tensor:
        """
        Unbatched latent x: (C, T, H, W)
        Returns neighbors of shape: (neighborhood_size-1, C, T, H, W)

        Matches the author's approach:
        neighbor = r * (thr * u + sqrt(1 - thr^2) * w)
        where:
        - r = ||x||_2 (constant radius)
        - u = x / ||x||
        - w is a random unit vector in the tangent plane (w âŸ‚ u)
        - thr is either given (threshold) or derived from lambda_dist via:
            d^2 = 2 r^2 (1 - cosÎ¸)  =>  cosÎ¸ = 1 - d^2/(2 r^2)
        """
        if (threshold is None) == (lambda_dist is None):
            raise ValueError("Specify exactly one of `threshold` or `lambda_dist` (but not both).")

        # number of neighbors to RETURN (excluding the pivot itself)
        N = max(int(neighborhood_size) - 1, 0)
        if N == 0:
            # return an empty tensor with correct shape
            return x.unsqueeze(0).expand(0, *x.shape).clone()

        device = x.device
        # do numerics in float64 for stability, cast back at the end
        x64 = x.to(torch.float64)
        xf = x64.flatten()                        # (D,)
        D = xf.numel()
        eps = 1e-12

        r = xf.norm().clamp_min(eps)              # scalar radius
        u = xf / r                                # (D,) unit vector

        # map lambda_dist -> cosine threshold if needed
        if lambda_dist is not None:
            d2 = float(lambda_dist) ** 2
            thr = 1.0 - d2 / (2.0 * float(r.item()) ** 2)
            # clamp to valid open interval to avoid sqrt negatives / degenerate caps
            thr = float(max(min(thr, 1.0 - 1e-9), -1.0 + 1e-6))
        else:
            thr = float(threshold)
            thr = float(max(min(thr, 1.0 - 1e-9), -1.0 + 1e-6))

        # random directions, then project to tangent plane of u
        g = generator if generator is not None else torch.Generator(device=device)
        v = torch.randn(N, D, dtype=torch.float64, device=device, generator=g)  # (N, D)

        # w = v - (vÂ·u) u  (so w âŸ‚ u), then normalize
        dot = (v @ u.unsqueeze(-1)).squeeze(-1)           # (N,)
        w = v - dot.unsqueeze(-1) * u.unsqueeze(0)        # (N, D)
        w = F.normalize(w, dim=-1, eps=1e-12)             # unit vectors in tangent plane

        sqrt_term = (1.0 - thr ** 2) ** 0.5               # scalar
        new = r * (thr * u.unsqueeze(0) + sqrt_term * w)  # (N, D)

        # reshape back and cast to original dtype/device
        new = new.reshape(N, *x.shape).to(x.dtype)
        return new

    def _generate_neighborhood(self, pivot: torch.Tensor, neighborhood_size: int, lambda_dist: float):
        """
        pivot: (C, T, H, W) unbatched latent
        returns: (neighborhood_size-1, C, T, H, W)
        """
        return self.generate_neighbors_spherical_cap_unbatched(
            pivot,
            neighborhood_size=neighborhood_size,
            lambda_dist=lambda_dist,
            generator=None,  # or pass your torch.Generator if you want deterministic neighbors
        )

    def _prepare_text_context(
        self, input_prompt: str, n_prompt: str, offload_model: bool, size, frame_num
    ):
        """Encodes text once and returns (context, context_null, seq_len)."""
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        return context, context_null, seq_len

    def _set_scheduler(self, sample_solver: str, sampling_steps: int, shift: float):
        """Constructs scheduler and returns (sample_scheduler, timesteps)."""
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift
            )
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=self.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler, timesteps

    def _tempering_weight(
        self,
        t_idx: int,
        T: int,
        scheme: str = "increase",
        base_tau: float = 1.0,
        gamma: float = 0.06,
    ):
        """Ï„_t schedule. Paper discusses several; we implement Constant/Increase/Inf."""  # see Sec. 4.1.1
        if scheme == "constant":
            return base_tau
        elif scheme == "increase":
            # An increasing emphasis toward later (cleaner) steps.
            # Smooth variant of ( (1+Î³)^{T-t} - 1 ) * Ï„
            return base_tau * ((1.0 + gamma) ** (T - t_idx) - 1.0 + 1e-6)
        elif scheme == "inf":
            return float("inf")
        else:
            return base_tau

    def _resample_children(self, weights: np.ndarray, N: int, method: str = "ssp"):
        """
        Low-variance resampling: deterministic rounding (SSP-like).
        Fallback to multinomial if requested. Return child counts for each particle.
        """
        weights = np.maximum(weights, 1e-12)
        weights /= weights.sum()
        if method == "multinomial":
            # sample indices N times with replacement then count
            idx = np.random.choice(len(weights), size=N, p=weights)
            counts = np.bincount(idx, minlength=len(weights))
            return counts.astype(int)
        # SSP-style: expected counts then distribute remainders by fractional parts
        exp_counts = N * weights
        base = np.floor(exp_counts).astype(int)
        remainder = N - base.sum()
        if remainder > 0:
            frac = exp_counts - base
            order = np.argsort(-frac)
            base[order[:remainder]] += 1
        return base


    def _in_resampling_window(self, t_idx: int, T: int, freq: int, t_start: int, t_end: int) -> bool:
        """
        Should we resample at depth t_idx?
        - Only within [t_start, t_end] (inclusive, clamped to [0, T-1]).
        - Every `freq` steps starting from t_start.
        - If freq <= 0, never resample (leaf still verifies).
        """
        if freq is None or freq <= 0:
            return False
        lo = max(0, int(t_start))
        hi = min(T - 1, int(t_end))
        if t_idx < lo or t_idx > hi:
            return False
        return ((t_idx - lo) % int(freq)) == 0


    def _em_noise_std(self, sample_scheduler, t_idx: int) -> float:
        """
        Euler-Maruyama noise std used by Flow-GRPO for WAN:
            std = std_dev_t * sqrt(-Î”t)
        where
            Î”t = sigma_prev - sigma   (note: sigmas are decreasing, so Î”t <= 0)
            std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
        """
        if not hasattr(sample_scheduler, "sigmas"):
            return 0.0
        sigmas = sample_scheduler.sigmas
        # clamp index so idx+1 is valid
        idx = min(int(t_idx), len(sigmas) - 2)
        sigma      = float(sigmas[idx])
        sigma_prev = float(sigmas[idx + 1])
        dt = sigma_prev - sigma                    # â‰¤ 0 (schedule is decreasing)
        if dt >= 0:
            # fallback (shouldn't happen if schedule is strictly decreasing)
            dt = -1e-8
        sigma_max = float(sigmas[1])               # follows Flow-GRPO impl
        sigma_min = float(sigmas[-1])
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
        return float(std_dev_t * np.sqrt(-dt))     # scalar std for this step

    def _denoise_batch_separate_schedulers(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        timesteps,
        schedulers: List,  # List of per-particle schedulers
        gens: List[torch.Generator],
        guide_scale: float,
        context: List[torch.Tensor],      
        context_null: List[torch.Tensor], 
        seq_len: int,
        cmm_nfes: int,
    ):
        """
        Batched denoising with per-particle scheduler states.
        x_t: (N, C, T, H, W)
        """
        B = x_t.shape[0]
        t = timesteps[t_idx]
        
        # 1. Broadcast timestep
        t_b = torch.full(
            (B,),
            t.item() if isinstance(t, torch.Tensor) else t,
            device=self.device,
            dtype=timesteps.dtype,
        )
        
        # 2. Batched model forward (efficient)
        latent_list = [x_t[i] for i in range(B)]
        eps_c_list = self.model(latent_list, t=t_b, context=context, seq_len=seq_len)
        cmm_nfes += 1
        eps_u_list = self.model(latent_list, t=t_b, context=context_null, seq_len=seq_len)
        cmm_nfes += 1
        
        eps_c = torch.stack(eps_c_list, dim=0).to(torch.float32)
        eps_u = torch.stack(eps_u_list, dim=0).to(torch.float32)
        noise_pred = eps_u + guide_scale * (eps_c - eps_u)
        
        # 3. Per-particle scheduler step (maintains separate states)
        x_next_list = []
        x0_hat_list = []
        
        for i in range(B):
            out = schedulers[i].step(
                noise_pred[i:i+1],  # Single particle
                t,
                x_t[i:i+1].to(torch.float32),
                return_dict=True,
                generator=gens[i]
            )
            
            x_next_list.append(out.prev_sample.squeeze(0))
            
            if hasattr(out, "pred_original_sample") and out.pred_original_sample is not None:
                x0_hat_list.append(out.pred_original_sample.squeeze(0).detach())
            else:
                if hasattr(schedulers[i], "sigmas"):
                    sigma_t = schedulers[i].sigmas[t_idx]
                    x0_hat = (x_t[i] - sigma_t * noise_pred[i].to(x_t.dtype)).detach()
                else:
                    x0_hat = out.prev_sample.squeeze(0).detach()
                x0_hat_list.append(x0_hat)
        
        x_next = torch.stack(x_next_list)
        x0_hat = torch.stack(x0_hat_list)
        
        return x_next, x0_hat, cmm_nfes
    
    
    def _initialize_particles(self, N: int, seeds_batch: List[int], size: tuple, frame_num: int):
        """Initialize particles with given seeds."""
        F = frame_num
        z_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        
        gens = []
        particles_list = []
        
        # Extend seeds if needed
        if len(seeds_batch) < N:
            extra = np.random.randint(0, 1000000, N - len(seeds_batch)).tolist()
            seeds_batch.extend(extra)
        
        for k in range(N):
            g = torch.Generator(device=self.device)
            g.manual_seed(int(seeds_batch[k]))
            gens.append(g)
            particles_list.append(
                torch.randn(z_shape, dtype=torch.float32, device=self.device, generator=g)
            )
        
        particles = torch.stack(particles_list).to(self.device)
        return particles, gens
    
    def _compute_ess(self, weights: np.ndarray) -> float:
        """Compute Effective Sample Size for adaptive resampling."""
        weights = np.maximum(weights, 1e-12)
        normalized_w = weights / weights.sum()
        ess = 1.0 / (normalized_w ** 2).sum()
        return ess
    
    def _stratified_resample(self, weights: np.ndarray, N: int) -> np.ndarray:
        """Stratified resampling for lower variance."""
        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()
        
        positions = (np.arange(N) + np.random.uniform(size=N)) / N
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        return np.clip(indices, 0, len(weights) - 1)
    
    def _apply_tangent_space_roughening(
        self,
        particles: torch.Tensor,
        gens: List[torch.Generator],
        base_std: float,
        diversity_factor: float = 1.5
    ) -> torch.Tensor:
        """Apply diversity-preserving tangent space perturbations."""
        N = particles.shape[0]
        
        # Compute mean particle
        mean_particle = particles.mean(dim=0, keepdim=True)
        
        # Add orthogonal perturbations for each particle
        for i in range(N):
            # Generate random noise
            noise = torch.empty_like(particles[i]).normal_(generator=gens[i])
            
            # Compute direction from mean
            direction = (particles[i] - mean_particle[0]).flatten()
            noise_flat = noise.flatten()
            
            # Project out the component parallel to direction
            dir_norm_sq = (direction @ direction).item()
            if dir_norm_sq > 1e-8:
                projection = ((noise_flat @ direction) / dir_norm_sq) * direction
                orthogonal_noise = noise_flat - projection
            else:
                orthogonal_noise = noise_flat
            
            # Reshape and apply
            orthogonal_noise = orthogonal_noise.view_as(noise)
            particles[i] = particles[i] + base_std * diversity_factor * orthogonal_noise
        
        return particles
    
    def _compute_fkd_weights(
        self,
        scores: np.ndarray,
        population_rewards: np.ndarray,
        running_max: np.ndarray,
        tau: float,
        potential_type: str = "max"
    ) -> np.ndarray:
        """Compute weights using FKD-style potential functions."""
        
        if potential_type == "max":
            # Weight by maximum reward seen
            rewards = np.maximum(scores, running_max)
            weights = np.exp(tau * rewards)
        elif potential_type == "diff":
            # Weight by improvement
            diffs = scores - population_rewards
            weights = np.exp(tau * diffs)
        elif potential_type == "add":
            # Weight by cumulative reward
            rewards = scores + population_rewards
            weights = np.exp(tau * rewards)
        else:  # "rt" or "current"
            # Weight by current reward only
            weights = np.exp(tau * scores)
        
        # Safety: clamp and handle NaNs
        weights = np.clip(weights, 0, 1e10)
        weights[np.isnan(weights)] = 1e-10
        
        return weights

    def generate_bfs(
        self,
        input_prompt: str,
        verifier: Callable[[torch.Tensor], float],
        N: int = 8,
        size=(1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seeds_batch: List[int] = None,
        offload_model: bool = True,
        # Original BFS parameters
        scoring: str = "max",
        tempering: str = "increase",
        resampling: str = "stratified",
        resample_frequency: int = 10,
        resample_t_start: int = 15,
        resample_t_end: int = 45,
        # Enhancement parameters
        use_adaptive_resampling: bool = True,
        ess_threshold: float = 0.5,
        diversity_factor: float = 1.5,
        potential_type: str = "max",
        debug: bool = False,
    ):
        """
        Enhanced BFS with diversity preservation and adaptive resampling.
        
        Enhancement parameters:
            use_adaptive_resampling: Only resample when ESS is low
            ess_threshold: ESS threshold for adaptive resampling
            diversity_factor: How much to boost diversity (>1.0)
            potential_type: Weight computation method ("max", "diff", "add", "rt")
        """
        cmm_nfes = 0
        start = time.time()
        
        # 1. Setup & Context
        context_single, context_null_single, seq_len = self._prepare_text_context(
            input_prompt, n_prompt, offload_model, size, frame_num
        )
        
        # Replicate context
        context = context_single * N
        context_null = context_null_single * N
        
        # Create per-particle schedulers
        schedulers = []
        for _ in range(N):
            scheduler, timesteps = self._set_scheduler(
                sample_solver, sampling_steps, shift
            )
            schedulers.append(scheduler)
        
        T = len(timesteps)
        
        # 2. Initialize Particles
        if seeds_batch is None:
            seeds_batch = list(range(42, 42 + N))
        
        particles, gens = self._initialize_particles(N, seeds_batch, size, frame_num)
        
        # Tracking variables
        population_rewards = np.zeros(N)
        running_max = np.full((N,), -1e9, dtype=np.float32)
        
        # Results storage
        final_best_score = -1e9
        final_best_media = None
        score_for_final_particles = []
        final_particles_list = []
        
        # Statistics
        resample_count = 0
        ess_history = []
        
        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            for t_idx in range(T):
                
                do_resample = self._in_resampling_window(
                    t_idx, T, resample_frequency, resample_t_start, resample_t_end
                )
                is_final_step = (t_idx == T - 1)
                do_verify = do_resample or is_final_step
                
                # --- A. BATCHED DENOISE with separate schedulers ---
                self.model.to(self.device)
                
                x_next, x0_hats, cmm_nfes = self._denoise_batch_separate_schedulers(
                    particles, t_idx, timesteps, schedulers, gens, guide_scale,
                    context, context_null, seq_len, cmm_nfes
                )
                
                # Update state
                particles = x_next
                
                # --- B. BATCHED VERIFY ---
                scores = None
                weights = None
                
                if do_verify:
                    if offload_model:
                        self.model.cpu()
                        torch.cuda.empty_cache()
                    
                    if self.rank == 0:
                        # Decode and score
                        x0_list = [x0_hats[i] for i in range(N)]
                        media_batch = self.vae.decode(x0_list)
                        
                        verifier_input = [media.squeeze(1)[None] for media in media_batch]
                        raw_scores = verifier(verifier_input)
                        
                        if isinstance(raw_scores, torch.Tensor):
                            s_raw_batch = raw_scores.cpu().numpy().flatten()
                        else:
                            s_raw_batch = np.array(raw_scores).flatten()
                        
                        # Compute weights
                        tau_t = self._tempering_weight(t_idx, T, tempering)
                        
                        # FKD-style weight computation
                        if potential_type != "current":
                            weights = self._compute_fkd_weights(
                                s_raw_batch, population_rewards, running_max,
                                tau_t, potential_type
                            )
                        else:
                            # Standard weight computation
                            if tau_t == float("inf"):
                                scores = 1e6 * s_raw_batch
                            else:
                                scores = tau_t * s_raw_batch
                            
                            if scoring == "max":
                                running_max = np.maximum(running_max, scores)
                                scores = running_max
                            
                            sm = np.exp(scores - scores.max())
                            weights = sm / np.maximum(sm.sum(), 1e-12)
                        
                        # Update tracking
                        population_rewards = s_raw_batch
                        if scoring == "max":
                            running_max = np.maximum(running_max, s_raw_batch * tau_t)
                        
                        # Store final particles
                        if is_final_step:
                            for k in range(N):
                                score_for_final_particles.append(s_raw_batch[k])
                                final_particles_list.append(media_batch[k].detach().cpu().clone())
                                
                                if s_raw_batch[k] > final_best_score:
                                    final_best_score = s_raw_batch[k]
                                    final_best_media = media_batch[k]
                
                if is_final_step:
                    break
                
                # --- C. ADAPTIVE RESAMPLING ---
                if do_resample and weights is not None and self.rank == 0:
                    # Compute ESS
                    ess = self._compute_ess(weights)
                    ess_history.append(ess)
                    
                    # Decide whether to actually resample
                    should_actually_resample = (
                        not use_adaptive_resampling or
                        ess < ess_threshold * N or
                        t_idx == resample_t_end  # Force at end of window
                    )
                    
                    if debug:
                        print(f"Step {t_idx}: ESS={ess:.2f}/{N}, Resample={should_actually_resample}")
                    
                    if should_actually_resample:
                        resample_count += 1
                        
                        # Choose resampling method
                        if resampling == "stratified":
                            selection_indices = self._stratified_resample(weights, N)
                        elif resampling == "ssp":
                            counts = self._resample_children(weights, N, method="ssp")
                            selection_indices = []
                            for k_old in range(N):
                                for _ in range(counts[k_old]):
                                    selection_indices.append(k_old)
                            while len(selection_indices) < N:
                                selection_indices.append(np.argmax(weights))
                            selection_indices = selection_indices[:N]
                        else:  # multinomial
                            weights_normalized = weights / weights.sum()
                            selection_indices = np.random.choice(N, size=N, p=weights_normalized)
                        
                        # Resample particles, schedulers, and generators
                        new_particles = []
                        new_schedulers = []
                        new_gens = []
                        
                        for idx in selection_indices:
                            new_particles.append(particles[idx].clone())
                            new_schedulers.append(copy.deepcopy(schedulers[idx]))
                            new_gens.append(gens[idx])
                        
                        particles = torch.stack(new_particles)
                        schedulers = new_schedulers
                        gens = new_gens
                        
                        # Update tracking arrays
                        population_rewards = population_rewards[selection_indices]
                        running_max = running_max[selection_indices]
                        
                        # Enhanced roughening with diversity preservation
                        roughen_std = self._em_noise_std(schedulers[0], t_idx)
                        if roughen_std > 0:
                            if diversity_factor > 1.0:
                                # Use tangent space roughening for better diversity
                                particles = self._apply_tangent_space_roughening(
                                    particles, gens, roughen_std, diversity_factor
                                )
                            else:
                                # Standard per-particle roughening
                                for i in range(N):
                                    noise = torch.empty_like(particles[i]).normal_(generator=gens[i])
                                    particles[i] = particles[i] + roughen_std * noise
        
        end = time.time()
        if offload_model:
            self.model.cpu()
            gc.collect()
            torch.cuda.synchronize()
        
        return {
            "media_asset": final_best_media if self.rank == 0 else None,
            "best_score": final_best_score,
            "score_for_final_particles": score_for_final_particles,
            "final_particles": final_particles_list,
            "runtime": end - start,
            "steps": T,
            "particles": N,
            "resample_count": resample_count,
            "avg_ess": np.mean(ess_history) if ess_history else N,
            "cmm_nfes": cmm_nfes,
        }
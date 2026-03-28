import gc
import logging

from model import CausalDiffusion
from utils.dataset import cycle, LatentLMDBDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        # I2V: 将 CLIP 编码器移到 GPU
        if self.model.clip_encoder is not None:
            self.model.clip_encoder = self.model.clip_encoder.to(
                device=self.device, dtype=self.dtype)

        if not config.no_visualize or config.load_raw_video or getattr(config, 'i2v', False):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))
       
        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.loss_ema = None  # loss 移动平均
        self.loss_ema_decay = 0.99
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9) 
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder
            
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):
        self.log_iters = 1
        step_start_time = time.time()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
           
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts) 
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            # I2V: 编码 CLIP 特征并构造 y
            if getattr(self.config, 'i2v', False) and self.model.clip_encoder is not None:
                ref_image = batch.get("image", None)
                if ref_image is not None:
                    ref_image = ref_image.to(device=self.device, dtype=self.dtype)
                else:
                    # LatentLMDBDataset 没有 image 字段，从 clean_latent 首帧解码获取参考图
                    first_frame_latent = clean_latent[:, 0:1]  # [B, 1, C, H, W]
                    first_frame_pixel = self.model.vae.decode_to_pixel(
                        first_frame_latent).to(self.dtype)  # [B, 1, 3, H, W]
                    ref_image = first_frame_pixel[:, 0]  # [B, 3, H, W]
                
                clip_fea, y_list = self.model.encode_i2v_conditions(
                    ref_image=ref_image,
                    image_or_video_shape=image_or_video_shape,
                    batch_size=batch_size
                )
                conditional_dict["clip_fea"] = clip_fea
                conditional_dict["y"] = y_list
                unconditional_dict["clip_fea"] = clip_fea
                unconditional_dict["y"] = y_list

        if self.is_main_process and self.step % 10 == 0:
            print(f"[train_one_step] step={self.step}, conditions encoded, starting generator forward...", flush=True)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        # 计算 loss 移动平均
        cur_loss = generator_loss.item()
        if self.loss_ema is None:
            self.loss_ema = cur_loss
        else:
            self.loss_ema = self.loss_ema_decay * self.loss_ema + (1 - self.loss_ema_decay) * cur_loss

        step_elapsed = time.time() - step_start_time

        if self.is_main_process and (self.step <= 5 or self.step % 10 == 0):
            gpu_mem_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            gpu_mem_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
            lr_current = self.generator_optimizer.param_groups[0]['lr']
            print(
                f"[step {self.step:>6d}] "
                f"loss={cur_loss:.6f} | loss_ema={self.loss_ema:.6f} | "
                f"grad_norm={generator_grad_norm.item():.4f} | "
                f"lr={lr_current:.2e} | "
                f"step_time={step_elapsed:.2f}s | "
                f"GPU_mem={gpu_mem_alloc:.1f}GB/{gpu_mem_reserved:.1f}GB",
                flush=True
            )

        wandb_loss_dict = {
            "generator_loss": cur_loss,
            "generator_grad_norm": generator_grad_norm.item(),
            "loss_ema": self.loss_ema,
            "step_time": step_elapsed,
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    def train(self):
        if self.is_main_process:
            print("=" * 70, flush=True)
            print("[train] 训练配置摘要:", flush=True)
            print(f"  trainer:              {self.config.trainer}", flush=True)
            print(f"  i2v:                  {getattr(self.config, 'i2v', False)}", flush=True)
            print(f"  num_frame_per_block:  {getattr(self.config, 'num_frame_per_block', 'N/A')}", flush=True)
            print(f"  teacher_forcing:      {getattr(self.config, 'teacher_forcing', False)}", flush=True)
            print(f"  lr:                   {self.config.lr}", flush=True)
            print(f"  batch_size:           {self.config.batch_size} (total: {getattr(self.config, 'total_batch_size', 'N/A')})", flush=True)
            print(f"  mixed_precision:      {self.config.mixed_precision}", flush=True)
            print(f"  dataset_size:         {len(self.dataset)}", flush=True)
            print(f"  save_interval:        {self.config.log_iters}", flush=True)
            print(f"  output_path:          {self.output_path}", flush=True)
            print(f"  world_size:           {dist.get_world_size()}", flush=True)
            model_name = getattr(self.config, 'model_kwargs', {}).get('model_name', 'N/A')
            print(f"  model_name:           {model_name}", flush=True)
            print("=" * 70, flush=True)
            print("[train] Starting training loop...", flush=True)

        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    iter_time = current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iter_time}, step=self.step)
                    self.previous_time = current_time

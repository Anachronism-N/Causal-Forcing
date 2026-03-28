import argparse
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.wan_wrapper import WanCLIPEncoder
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# I2V: 初始化 CLIP 编码器
clip_encoder = None
if args.i2v:
    clip_model_dir = getattr(config, "clip_model_dir", None)
    if clip_model_dir is None:
        # 从 model_kwargs 中推断模型名
        model_name = getattr(config, "model_kwargs", {}).get("model_name", "Wan2.1-I2V-14B-720P")
        clip_model_dir = f"wan_models/{model_name}"
    print(f"Loading CLIP encoder from {clip_model_dir}")
    clip_encoder = WanCLIPEncoder(model_dir=clip_model_dir)
    clip_encoder = clip_encoder.to(device=device, dtype=torch.bfloat16)
    clip_encoder.eval()


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt]
        num_noise_frames = args.num_output_frames - 1
        # 自动向上对齐：确保噪声帧数能被 num_frame_per_block 整除
        block_size = config.num_frame_per_block
        if num_noise_frames % block_size != 0:
            aligned_noise_frames = ((num_noise_frames + block_size - 1) // block_size) * block_size
            print(f"[I2V] num_noise_frames {num_noise_frames} 不能被 num_frame_per_block={block_size} 整除，"
                  f"自动对齐为 {aligned_noise_frames}（输出帧数: {aligned_noise_frames + 1}）")
            num_noise_frames = aligned_noise_frames
        sampled_noise = torch.randn(
            [1, num_noise_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        
        # I2V: 编码 CLIP 条件并构造 y
        clip_fea = None
        y_cond = None
        if clip_encoder is not None:
            # image for CLIP: [B, C, H, W], 值域 [-1, 1]
            clip_image = batch['image'].squeeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            clip_fea = clip_encoder(clip_image)  # [B, 257, 1280]
            
            # 构造 y: 对齐 Wan2.1 I2V 官方实现
            # y = concat(mask[4ch], image_latent[16ch]) = 20ch
            num_latent_frames = num_noise_frames + 1  # F_lat（包含初始帧），使用对齐后的帧数
            F = (num_latent_frames - 1) * 4 + 1  # 像素帧数
            lat_h, lat_w = 60, 104
            H_pixel, W_pixel = lat_h * 8, lat_w * 8
            
            # 参考图像 resize 到目标分辨率
            ref_img = batch['image'].squeeze(0).to(device=device, dtype=torch.bfloat16)  # [C, H, W]
            ref_resized = torch.nn.functional.interpolate(
                ref_img.unsqueeze(0), size=(H_pixel, W_pixel), mode='bicubic').squeeze(0)  # [3, H, W]
            
            # 构造视频帧: [3, F, H, W], 第一帧是参考图，其余填零
            video_frames = torch.zeros(3, F, H_pixel, W_pixel, device=device, dtype=torch.bfloat16)
            video_frames[:, 0] = ref_resized
            
            # VAE 编码
            img_latent = pipeline.vae.model.encode(
                video_frames.unsqueeze(0),
                [pipeline.vae.mean.to(device=device, dtype=torch.bfloat16),
                 (1.0 / pipeline.vae.std).to(device=device, dtype=torch.bfloat16)]
            ).to(dtype=torch.bfloat16).squeeze(0)  # [16, F_lat, lat_h, lat_w]
            
            # 构造 mask: 对齐官方实现 (4 channel)
            msk = torch.ones(1, F, lat_h, lat_w, device=device, dtype=torch.bfloat16)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]  # [4, F_lat, lat_h, lat_w]
            
            # y = concat(mask[4ch], latent[16ch]) = [20, F_lat, lat_h, lat_w]
            y_cond = [torch.concat([msk, img_latent])]  # list of [20, F_lat, H, W]
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        clip_fea=clip_fea if args.i2v else None,
        y=y_cond if args.i2v else None
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
    write_video(output_path, video[0], fps=16)

       

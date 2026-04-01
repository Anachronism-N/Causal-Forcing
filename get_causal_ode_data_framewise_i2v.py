"""
Stage 2 ODE 数据生成脚本 (I2V 版本)
用 Stage 1 训练得到的 causal I2V AR Diffusion 模型生成 ODE trajectory pairs

使用方法:
torchrun --nproc_per_node=8 get_causal_ode_data_framewise_i2v.py \
    --output_folder dataset/ODE6KCausal_framewise_i2v \
    --rawdata_path dataset/clean_data_i2v \
    --generator_ckpt outputs/stage1_i2v/checkpoint_model_XXXXXX/model.pt \
    --guidance_scale 6.0
"""
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanCLIPEncoder
from utils.scheduler import FlowMatchScheduler
from utils.distributed import launch_distributed_job

import torch.distributed as dist
import torch.nn.functional as TF_func
from tqdm import tqdm
import argparse
import torch
import math
import os
from utils.dataset import LatentLMDBDataset


def init_model(device, model_name="Wan2.1-Fun-1.3B-InP"):
    model = WanDiffusionWrapper(model_name=model_name, is_causal=True).to(device).to(torch.float32)
    model.model.num_frame_per_block = 3

    encoder = WanTextEncoder().to(device).to(torch.float32)

    vae = WanVAEWrapper().to(device).to(torch.float32)

    clip_encoder = WanCLIPEncoder(model_dir=f"wan_models/{model_name}").to(device).to(torch.float32)

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(text_prompts=[sample_neg_prompt])

    return model, encoder, vae, clip_encoder, scheduler, unconditional_dict


def encode_i2v_conditions(clip_encoder, vae, ref_image, image_or_video_shape, device, dtype=torch.float32):
    """编码 I2V 条件：CLIP 特征 + y (mask + image_latent)"""
    ref_image = ref_image.to(device=device, dtype=dtype)
    clip_fea = clip_encoder(ref_image)  # [B, 257, 1280]

    lat_h, lat_w = image_or_video_shape[3], image_or_video_shape[4]
    num_latent_frames = image_or_video_shape[1]
    F_pixel = (num_latent_frames - 1) * 4 + 1
    H_pixel = lat_h * 8
    W_pixel = lat_w * 8

    ref_resized = TF_func.interpolate(
        ref_image, size=(H_pixel, W_pixel), mode='bicubic')

    batch_size = ref_image.shape[0]
    y_list = []
    for b in range(batch_size):
        video_frames = torch.zeros(3, F_pixel, H_pixel, W_pixel, device=device, dtype=dtype)
        video_frames[:, 0] = ref_resized[b]

        img_latent = vae.model.encode(
            video_frames.unsqueeze(0),
            [vae.mean.to(device=device, dtype=dtype),
             (1.0 / vae.std).to(device=device, dtype=dtype)]
        ).float().squeeze(0)

        msk = torch.ones(1, F_pixel, lat_h, lat_w, device=device, dtype=dtype)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        y_i = torch.concat([msk, img_latent])
        y_list.append(y_i)

    return clip_fea, y_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--rawdata_path", type=str, required=True)
    parser.add_argument("--generator_ckpt", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--model_name", type=str, default="Wan2.1-Fun-1.3B-InP")

    args = parser.parse_args()

    launch_distributed_job()
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, vae, clip_encoder, scheduler, unconditional_dict = init_model(
        device=device, model_name=args.model_name)

    # 加载 Stage 1 checkpoint
    state_dict = torch.load(args.generator_ckpt, map_location="cpu")
    gen_sd = state_dict["generator"]
    fixed = {}
    for k, v in gen_sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "", 1)
        if k.startswith("model."):
            k = k.replace("model.", "", 1)
        fixed[k] = v
    state_dict = fixed
    model.model.load_state_dict(state_dict, strict=True)
    # 确保加载 checkpoint 后所有参数统一为 float32
    # （混合精度训练保存的 checkpoint 可能包含 float16 权重）
    model = model.to(torch.float32)

    dataset = LatentLMDBDataset(args.rawdata_path)

    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    total_steps = int(math.ceil(len(dataset) / dist.get_world_size()))
    image_or_video_shape = [1, 21, 16, 60, 104]

    for index in tqdm(range(total_steps), disable=(dist.get_rank() != 0)):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        sample = dataset[prompt_index]
        prompt = sample["prompts"]
        clean_latent = sample["clean_latent"].to(device).unsqueeze(0)

        # 编码文本条件
        conditional_dict = encoder(text_prompts=prompt)

        # I2V: 从 clean_latent 首帧解码获取参考图，编码 CLIP + y
        first_frame_latent = clean_latent[:, 0:1]
        first_frame_pixel = vae.decode_to_pixel(first_frame_latent).to(torch.float32)
        ref_image = first_frame_pixel[:, 0]  # [B, 3, H, W]

        clip_fea, y_list = encode_i2v_conditions(
            clip_encoder, vae, ref_image, image_or_video_shape, device)
        conditional_dict["clip_fea"] = clip_fea
        conditional_dict["y"] = y_list
        # I2V: 无条件分支也必须包含 clip_fea 和 y，否则 CFG 的无条件预测缺少图像条件
        unconditional_dict["clip_fea"] = clip_fea
        unconditional_dict["y"] = y_list

        # ODE sampling
        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device)

        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps, disable=(dist.get_rank() != 0))):
            timestep = t * torch.ones([1, 21], device=device, dtype=torch.float32)
            noisy_input.append(latents)

            f_cond, x0_pred_cond = model(
                latents, conditional_dict, timestep, clean_x=clean_latent)

            f_uncond, x0_pred_uncond = model(
                latents, unconditional_dict, timestep, clean_x=clean_latent)

            flow_pred = f_uncond + args.guidance_scale * (f_cond - f_uncond)

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)
        noisy_input.append(clean_latent)

        noisy_inputs = torch.stack(noisy_input, dim=1)
        # 选取关键时间步: [0, 12, 24, 36, 倒数第二, 最后(clean)]
        noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -2, -1]]

        stored_data = noisy_inputs

        torch.save(
            {prompt: stored_data.cpu().detach()},
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

    dist.barrier()


if __name__ == "__main__":
    main()

# 推理方案设计

## 1. 推理模式

### 1.1 单段 I2V 推理
与现有 CausalPipeline.generate_video 类似，增加 I2V 条件：
```python
video = pipeline.generate_video(
    ref_image=image,
    prompt="...",
    num_frames=81,
    height=480, width=832,
    num_causal_block=4,
)
```

### 1.2 流式自回归推理
```python
# 初始化
state = pipeline.init_stream(ref_image, prompt, config)

# 逐 chunk 生成
for i in range(max_chunks):
    chunk_latent = pipeline.generate_next_chunk(state)
    # 可选：增量 VAE decode 并输出
    chunk_pixels = vae.decode(chunk_latent)
    yield chunk_pixels

# 或一次性合并
all_latents = torch.cat(state.history_latents, dim=2)
video = vae.decode(all_latents)
```

## 2. CausalI2VPipeline 设计

### 2.1 __init__
```python
class CausalI2VPipeline:
    def __init__(
        self, 
        model,          # WanCausalI2VModel
        vae,            # WanVAE
        text_encoder,   # T5 encoder
        clip_model,     # CLIP visual model
        scheduler,      # DDPM/Flow scheduler
    ):
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.clip_model = clip_model
        self.scheduler = scheduler
```

### 2.2 encode_conditions
```python
def encode_conditions(self, ref_image, prompt, device):
    """编码所有条件，返回可复用的条件 dict"""
    # Text
    text_tokens = self.text_encoder.encode(prompt)
    null_context = self.text_encoder.encode("")
    
    # CLIP image
    clip_input = preprocess_for_clip(ref_image)
    clip_fea = self.clip_model.visual(clip_input)
    
    # VAE image latent
    image_latent = self.vae.encode(ref_image)  # [1, 16, 1, H, W]
    
    return {
        "text_tokens": text_tokens,
        "null_context": null_context,
        "clip_fea": clip_fea,
        "image_latent": image_latent,
    }
```

### 2.3 generate_video（单段，兼容现有接口）
```python
def generate_video(self, ref_image, prompt, num_frames, height, width, 
                   num_causal_block, cfg_scale=7.5, num_steps=50):
    """
    在现有 CausalPipeline.generate_video 基础上增加 I2V 条件
    """
    # 1. 编码条件
    conds = self.encode_conditions(ref_image, prompt, device)
    
    # 2. 计算 latent 尺寸
    T_lat = (num_frames - 1) // 4 + 1  # Wan2.1 VAE 的时间下采样
    H_lat = height // 8
    W_lat = width // 8
    
    # 3. 构造 image_latent repeat
    image_latent_full = conds["image_latent"].repeat(1, 1, T_lat, 1, 1)
    
    # 4. 自回归去噪循环（复用现有逻辑，增加 I2V 条件）
    # ... 与 CausalPipeline.generate_video 类似 ...
    # 每次调用模型时：
    #   model_input = cat([noisy_latent, image_latent_full], dim=1)
    #   model(model_input, t, context=text_tokens, clip_fea=clip_fea, ...)
    
    # 5. VAE decode
    video = self.vae.decode(clean_latent)
    return video
```

### 2.4 init_stream
```python
def init_stream(self, ref_image, prompt, config) -> StreamingState:
    conds = self.encode_conditions(ref_image, prompt, device)
    return StreamingState(
        text_tokens=conds["text_tokens"],
        clip_fea=conds["clip_fea"],
        image_latent=conds["image_latent"],
        null_context=conds["null_context"],
        history_latents=[],
        num_generated_chunks=0,
        latent_h=config.latent_h,
        latent_w=config.latent_w,
        chunk_len=config.chunk_len,
        max_history_chunks=config.max_history_chunks,
        overlap_frames=config.overlap_frames,
    )
```

### 2.5 generate_next_chunk
```python
def generate_next_chunk(self, state) -> Tensor:
    """
    核心流式生成函数
    """
    # 1. 取 history 窗口
    history = state.history_latents[-state.max_history_chunks:]
    n_hist = len(history)
    total_blocks = n_hist + 1
    
    # 2. 拼接 history (clean) + new noise
    if history:
        hist_cat = torch.cat(history, dim=2)  # [B, C, hist_T, H, W]
        total_T = hist_cat.shape[2] + state.chunk_len
    else:
        hist_cat = None
        total_T = state.chunk_len
    
    # 3. image_latent repeat
    img_lat = state.image_latent.repeat(1, 1, total_T, 1, 1)
    
    # 4. 初始化新 block 为随机噪声
    noise = torch.randn(1, 16, state.chunk_len, state.latent_h, state.latent_w)
    noisy = noise.clone()
    
    # 5. 去噪循环
    for t in self.scheduler.timesteps:
        # 拼接 history + noisy
        if hist_cat is not None:
            x = torch.cat([hist_cat, noisy], dim=2)
        else:
            x = noisy
        
        # channel concat image latent
        model_input = torch.cat([x, img_lat], dim=1)
        
        # CFG: 有条件 + 无条件
        noise_pred_cond = self.model(
            model_input, t, context=state.text_tokens, 
            clip_fea=state.clip_fea, num_causal_block=total_blocks
        )
        noise_pred_uncond = self.model(
            model_input, t, context=state.null_context,
            clip_fea=state.clip_fea,  # image 不 drop
            num_causal_block=total_blocks
        )
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 只取最后一个 block 的预测更新
        noise_pred_new = noise_pred[:, :, -state.chunk_len:]
        noisy = self.scheduler.step(noise_pred_new, t, noisy)
    
    # 6. 更新 state
    clean_chunk = noisy
    state.history_latents.append(clean_chunk)
    state.num_generated_chunks += 1
    
    return clean_chunk
```

## 3. CLI 入口

```python
# inference_i2v.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task", choices=["i2v", "i2v_streaming"], default="i2v")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    # 加载模型
    pipeline = build_i2v_pipeline(args)
    
    if args.task == "i2v":
        video = pipeline.generate_video(
            ref_image=load_image(args.ref_image),
            prompt=args.prompt,
            num_frames=args.num_frames,
            height=args.height, width=args.width,
        )
        save_video(video, "output.mp4")
    
    elif args.task == "i2v_streaming":
        state = pipeline.init_stream(
            ref_image=load_image(args.ref_image),
            prompt=args.prompt,
            config=stream_config,
        )
        for i in range(args.num_chunks):
            chunk = pipeline.generate_next_chunk(state)
            print(f"Generated chunk {i+1}/{args.num_chunks}")
        
        video = pipeline.merge_and_decode(state)
        save_video(video, "output_streaming.mp4")
```

## 4. merge_chunks 实现

```python
def merge_chunks(chunks, overlap_frames=0):
    """
    合并多个 chunk latent
    chunks: List[Tensor], each [B, C, T, H, W]
    """
    if not chunks:
        return None
    if overlap_frames == 0 or len(chunks) == 1:
        return torch.cat(chunks, dim=2)
    
    result = chunks[0]
    for chunk in chunks[1:]:
        alpha = torch.linspace(1, 0, overlap_frames, device=chunk.device)
        alpha = alpha.view(1, 1, -1, 1, 1)
        
        prev_overlap = result[:, :, -overlap_frames:]
        curr_overlap = chunk[:, :, :overlap_frames]
        blended = alpha * prev_overlap + (1 - alpha) * curr_overlap
        
        result = torch.cat([
            result[:, :, :-overlap_frames],
            blended,
            chunk[:, :, overlap_frames:]
        ], dim=2)
    
    return result
```

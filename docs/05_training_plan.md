# 训练方案设计

## 1. 训练阶段规划

### Phase 1: Causal I2V 训练
- **目标**: 在 causal-forcing 框架下训练 I2V 生成
- **数据**: ref_image (第一帧) + text + 完整视频
- **模型**: WanCausalI2VModel
- **训练方式**: 与现有 Causal-Forcing 一致，启用 causal block mask + pyramid noise
- **条件**: text + CLIP image feature + VAE image latent concat
- **Loss**: 标准 flow matching loss（分段噪声，causal mask）
- **初始化**: 从 T2V Causal-Forcing checkpoint 加载，新模块零初始化

### Phase 2: Streaming Continuation 训练（后期）
- **目标**: 训练模型基于历史 chunk 生成下一个 chunk
- **数据**: ref_image + history chunks (clean) + target chunk (noisy)
- **Loss**: 只在 target chunk 上计算

## 2. 训练循环修改

### 2.1 现有训练循环（train.py 核心）

```python
for batch in dataloader:
    # 1. VAE encode video
    video_latent = vae.encode(batch["video"])
    
    # 2. T5 encode text
    text_tokens = t5_encode(batch["text"])
    
    # 3. Sample timestep, add noise (pyramid noise)
    noise, timestep = pyramid_noise(video_latent, num_causal_block)
    noisy_latent = add_noise(video_latent, noise, timestep)
    
    # 4. Model forward
    noise_pred = model(noisy_latent, timestep, text_tokens, num_causal_block=N)
    
    # 5. Loss
    loss = flow_matching_loss(noise_pred, noise, timestep)
    loss.backward()
    optimizer.step()
```

### 2.2 I2V 训练循环修改

```python
for batch in dataloader:
    # 1. VAE encode video（不变）
    video_latent = vae.encode(batch["video"])
    
    # 2. T5 encode text（不变）
    text_tokens = t5_encode(batch["text"])
    
    # === 新增：I2V 条件编码 ===
    # 3. VAE encode 参考图像
    image_latent = vae.encode(batch["ref_image"])  # [B, 16, 1, H, W]
    image_latent = image_latent.repeat(1, 1, T, 1, 1)  # repeat to T
    
    # 4. CLIP encode 参考图像
    clip_fea = clip_model.visual(batch["clip_image"])  # [B, 257, 1280]
    
    # 5. Sample timestep, add noise（不变）
    noise, timestep = pyramid_noise(video_latent, num_causal_block)
    noisy_latent = add_noise(video_latent, noise, timestep)
    
    # === 新增：image latent concat ===
    # 6. 拼接 image latent
    model_input = torch.cat([noisy_latent, image_latent], dim=1)  # [B, 32, T, H, W]
    
    # === 修改：传入 clip_fea ===
    # 7. Model forward
    noise_pred = model(
        model_input, timestep, text_tokens, 
        clip_fea=clip_fea,           # 新增
        num_causal_block=N
    )
    
    # 8. Loss（不变）
    loss = flow_matching_loss(noise_pred, noise, timestep)
    loss.backward()
    optimizer.step()
```

## 3. Checkpoint 加载策略

### 3.1 从 T2V Causal-Forcing 加载

```python
def load_t2v_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # 处理 patch_embedding 通道扩展: 16ch → 32ch
    key = "patch_embedding.proj.weight"
    if key in state_dict:
        old_w = state_dict[key]  # [out_ch, 16, kT, kH, kW]
        new_w = torch.zeros(old_w.shape[0], 32, *old_w.shape[2:])
        new_w[:, :16] = old_w   # 前16通道复制旧权重
        # 后16通道零初始化 → 初始时 image_latent concat 通道不贡献
        state_dict[key] = new_w
    
    # 处理 patch_embedding bias（如果有）
    key_bias = "patch_embedding.proj.bias"
    # bias 不需要修改（输出维度不变）
    
    # 加载，允许缺失 img_emb
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")    # 预期: img_emb.*
    print(f"Unexpected keys: {unexpected}")
    
    return model
```

### 3.2 img_emb 初始化
```python
# 零初始化最后一层 → 初始时 image tokens 不干扰
nn.init.zeros_(model.img_emb[-1].weight)
nn.init.zeros_(model.img_emb[-1].bias)
```

## 4. 配置文件扩展

```yaml
# configs/wan_causal_i2v.yaml
task_type: "i2v"

# === 模型配置 ===
model:
  dim: 5120
  num_heads: 80
  num_layers: 40
  in_dim: 16                    # VAE latent channels（不变）
  model_in_dim: 32              # patch_embedding 输入通道（新增）
  clip_dim: 1280                # CLIP 输出维度（新增）

# === 条件配置 ===
condition:
  ref_image_source: "first_frame"
  p_drop_text: 0.1
  p_drop_image: 0.0             # 第一版不 drop image

# === CLIP 编码器 ===
clip:
  model_path: "/path/to/clip-vit-h-14"
  image_size: 224

# === 训练配置（保持与 T2V 一致） ===
training:
  learning_rate: 1e-5
  num_causal_block: 4
  # ... 其余不变

# === Streaming 配置 ===
streaming:
  chunk_size: 17                # 每个 chunk 的像素帧数
  max_history_chunks: 8
  overlap_frames: 0             # 第一版无重叠
```

## 5. 超参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| Learning Rate | 1e-5 | 与 T2V 训练一致 |
| Warm-up | 1000 steps | 新模块需要 warm-up |
| p_drop_text | 0.1 | CFG 文本 drop |
| p_drop_image | 0.0 | 第一版不 drop image |
| num_causal_block | 配置值 | 与 T2V 一致 |
| Gradient Accumulation | 按显存 | 至少 effective BS=8 |

## 6. 验证指标

- **训练 loss 曲线**：应单调下降并收敛
- **首帧重建质量**：生成视频首帧应与参考图像高度一致
- **视频连续性**：帧间无跳变
- **文本对齐**：视频内容应匹配文本描述

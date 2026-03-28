# 条件系统设计

## 1. 条件注入路径

### 1.1 文本条件（已有，保持不变）
- T5 encoder → `context` tensor `[B, text_seq_len, 4096]`
- 通过 cross-attention 注入每个 transformer block
- 在 `WanCausalModel.forward` 中通过 `e_context` 使用

### 1.2 图像条件 - 路径 A: CLIP → Cross-Attention
ref_image [B, 3, H, W]
→ CLIP visual encoder
→ clip_fea [B, 257, 1280] (256 patch tokens + 1 CLS token)
→ img_emb (MLP: 1280 → dim → dim)
→ image_tokens [B, 257, dim]
→ concat with context: [B, text_seq + 257, dim]
→ cross-attention 的 key/value

**关键参数**（对齐 Wan2.1 I2V 14B 配置）：
- `clip_dim = 1280`（CLIP-ViT-H/14 输出维度）
- `dim = 5120`（I2V-14B 模型维度）
- img_emb: `Linear(1280, 5120) → SiLU → Linear(5120, 5120)`

### 1.3 图像条件 - 路径 B: VAE Latent → Channel Concat
ref_image [B, 3, H_pixel, W_pixel]
→ resize to video resolution
→ VAE encode
→ image_latent [B, 16, 1, H_lat, W_lat]
→ repeat along time: [B, 16, T_lat, H_lat, W_lat]
→ concat with noisy_latent on channel dim
→ model_input [B, 32, T_lat, H_lat, W_lat]


**关键细节**（来自 Wan2.1 image2video.py）：
- VAE 编码后的 latent channels = 16
- concat 后 channels = 32
- `patch_embedding` 的 `in_channels` 需要从 16 改为 32
- image_latent 在时间维度上 repeat 到与 noisy_latent 相同的帧数

## 2. 条件在模型内部的流转

### WanCausalI2VModel.forward 中的处理：

```python
def forward(self, x, t, context, seq_len, clip_fea=None, num_causal_block=None):
    # x: [B, 32, T, H, W] — 已在外部 concat 好 image_latent
    # context: [B, text_seq, dim] — 文本 tokens
    # clip_fea: [B, 257, clip_dim] — CLIP 图像特征
    
    # 1. 图像 tokens 融合到 context
    if clip_fea is not None:
        image_tokens = self.img_emb(clip_fea)  # [B, 257, dim]
        context = torch.cat([context, image_tokens], dim=1)
    
    # 2. 后续与 WanCausalModel.forward 完全一致
    #    patch_embed → blocks (with causal attn + cross attn) → unpatch
```

## 3. 训练时的条件准备

### 在 train.py 的训练循环中：

```python
# 1. VAE 编码视频
video_latent = vae.encode(video)  # [B, 16, T, H, W]

# 2. VAE 编码参考图像
ref_image = video[:, :, 0:1, :, :]  # 取第一帧
image_latent = vae.encode(ref_image)  # [B, 16, 1, H, W]
image_latent = image_latent.repeat(1, 1, T, 1, 1)  # [B, 16, T, H, W]

# 3. CLIP 编码参考图像
clip_fea = clip_model.visual(ref_image_preprocessed)  # [B, 257, 1280]

# 4. 构造模型输入
noisy_latent = add_noise(video_latent, noise, timestep)  # [B, 16, T, H, W]
model_input = torch.cat([noisy_latent, image_latent], dim=1)  # [B, 32, T, H, W]

# 5. 调用模型
noise_pred = model(model_input, timestep, context=text_tokens, clip_fea=clip_fea, ...)
```

## 4. T2V 模式兼容

当 `task_type == "t2v"` 时：
- `clip_fea = None`，不计算图像条件
- `image_latent` 全零：`torch.zeros(B, 16, T, H, W)`
- `model_input = torch.cat([noisy_latent, zeros], dim=1)` — 后 16 通道全零
- 由于 `img_emb` 不参与、`patch_embedding` 后 16 通道零初始化，行为等价于原始 T2V

## 5. Classifier-Free Guidance 策略

I2V 模式下有三种条件可以 drop：
1. **Drop text only**: context = null_text_tokens, clip_fea = clip_fea
2. **Drop image only**: context = text_tokens, clip_fea = null_clip_fea, image_latent = zeros
3. **Drop both**: context = null_text_tokens, clip_fea = null_clip_fea, image_latent = zeros

CFG 公式（对齐 Wan2.1）：
noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)


第一版简化：只做 text drop（与 T2V 一致），image condition 始终保留。

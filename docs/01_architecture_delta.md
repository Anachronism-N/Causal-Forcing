# 架构差异分析

## 1. Wan2.1 T2V vs I2V 模型结构差异

### 1.1 关键对比

| 组件 | T2V (`WanModel`) | I2V (`WanI2VModel`) |
|------|-------------------|----------------------|
| 继承关系 | 基类 | 继承 `WanModel` |
| `patch_embedding` 输入通道 | `in_dim` (=16) | `in_dim * 2` (=32) |
| `img_emb` | 无 | `nn.Sequential(Linear, SiLU, Linear)` 将 CLIP dim 映射到 `dim` |
| cross-attn context | text tokens only | `torch.cat([context, img_emb], dim=1)` |
| model forward 输入 x 的通道 | 16 (noisy latent) | 32 (noisy latent || image_latent) |

### 1.2 WanI2VModel 新增代码（Wan2.1/wan/modules/model.py L698-L957）

```python
class WanI2VModel(WanModel):
    def __init__(self, ...):
        super().__init__(...)
        # 关键1: patch_embedding 输入通道翻倍
        self.patch_embedding = nn.Conv3d(in_dim * 2, dim, ...)
        # 关键2: 新增 img_emb MLP
        self.img_emb = nn.Sequential(
            nn.Linear(clip_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x, t, context, clip_fea=None, ...):
        # 关键3: image tokens concat 到 cross-attn context
        if clip_fea is not None:
            context = torch.cat([context, self.img_emb(clip_fea)], dim=1)
        # 关键4: x 已经是 32 通道（外部拼接好的）
        # 其余逻辑与 WanModel 相同
```

### 1.3 I2V Pipeline 中的条件准备（Wan2.1/wan/image2video.py）

```python
# 1. CLIP 编码参考图像
clip_encoder_out = self.clip.visual(ref_image_tensor)  # [B, 257, 1280]

# 2. VAE 编码参考图像
img_latent = self.vae.encode([ref_image_resized])  # [B, C, 1, H', W']

# 3. 将 img_latent 重复到所有时间帧
img_latent = img_latent.repeat(1, 1, T_latent, 1, 1)  # [B, C, T, H', W']

# 4. 拼接到 noisy latent 的通道维度
model_input = torch.cat([noisy_latent, img_latent], dim=1)  # [B, 2C, T, H', W']

# 5. 调用模型
noise_pred = model(model_input, t, context=text_tokens, clip_fea=clip_encoder_out)
```

## 2. Causal-Forcing 当前结构分析

### 2.1 核心文件

| 文件 | 核心内容 |
|------|---------|
| `models/wan_causal.py` | `WanCausalModel`：在 `WanModel` 基础上改造 attention 为 causal block |
| `models/attention.py` | `flash_attention_causal`：基于 `flash_attn_varlen_func` 实现 block-wise causal mask |
| `models/pipeline.py` | `CausalPipeline`：自回归推理循环，支持 next-block prediction |
| `models/noise.py` | `pyramid_noise`：分段噪声，不同 block 用不同时间步 |
| `data/dataset.py` | `VideoDataset`：视频数据加载 |
| `data/bucket.py` | `BucketSampler`：按 (H,W,T) 分桶 |
| `train.py` | 训练主循环 |
| `inference.py` | 推理入口 |
| `configs/wan_causal.yaml` | 训练/推理配置 |

### 2.2 WanCausalModel 相对 WanModel 的改造

1. **self.blocks 中的 self-attn 替换**：从标准 full attention → block causal attention
2. **`flash_attention_causal`**：通过 `cu_seqlens` 控制每个 block 的 attention 范围
3. **forward 参数新增 `num_causal_block`**：控制 causal block 数量
4. **forward 中 patch → token → attention → unpatch 的流程与 `WanModel` 基本一致**

### 2.3 CausalPipeline 自回归推理逻辑

```python
# pipeline.py: generate_video
for idx in range(num_causal_block):
    # 构造当前窗口：history blocks(clean) + new block(noisy)
    # 用标准 DDPM sampling 去噪 new block
    # 将去噪后的 new block 加入 history
```

## 3. 目标架构

### 3.1 WanCausalI2VModel（新模型类）
WanCausalI2VModel
├── 继承 WanCausalModel（获得 causal block attention + pyramid noise）
├── 新增 img_emb（对齐 WanI2VModel 的 MLP）
├── 修改 patch_embedding 输入通道（16 → 32）
├── forward 接收 clip_fea 参数
│ ├── img_emb(clip_fea) concat 到 cross-attn context
│ └── x 已是 32 通道（外部拼接 image_latent）
└── 其余逻辑继承自 WanCausalModel

### 3.2 CausalI2VPipeline（新 pipeline） 
CausalI2VPipeline
├── 继承/组合 CausalPipeline 的自回归循环
├── 新增 CLIP encoder、img_emb 条件编码
├── 新增 VAE image encoding → image_latent 拼接
├── 新增 StreamingState 管理
├── generate_video(ref_image, prompt, ...)
└── generate_next_chunk(state) for streaming


### 3.3 数据管道扩展 
VideoDatasetI2V
├── 继承 VideoDataset
├── 新增返回: ref_image_tensor (用于 CLIP), image_latent (用于 VAE concat)
└── 可配置: ref_image_source = "first_frame" | "separate_file"

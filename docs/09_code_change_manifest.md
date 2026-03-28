# 代码修改清单 (修订版 — 原地修改策略)

## 1. 设计原则

**不再新建模型/pipeline/trainer文件**，所有改动在原有文件上通过条件分支完成：
- `task_type` / `i2v` flag 区分 T2V 和 I2V 路径
- T2V 路径零改动、零风险回归
- I2V 改动量小（模型约15行，pipeline约50-80行），不值得继承或新建文件

## 2. 修改文件总览

| 文件路径 | 修改内容 | 优先级 | 影响范围 |
|----------|---------|--------|---------|
| `utils/wan_wrapper.py` | WanDiffusionWrapper 新增 CLIP 编码器 + 条件传递 | P0 | I2V 分支 |
| `wan/modules/causal_model.py` | CausalWanModel 已支持 I2V，无需修改（仅确认兼容） | - | 无 |
| `pipeline/self_forcing_training.py` | 训练 pipeline 传递 clip_fea + y 条件 | P0 | I2V 分支 |
| `pipeline/teacher_forcing_training.py` | 同上 | P0 | I2V 分支 |
| `pipeline/causal_inference.py` | few-step 推理 pipeline 传递 I2V 条件 | P0 | I2V 分支 |
| `pipeline/causal_diffusion_inference.py` | multi-step 推理 pipeline 传递 I2V 条件 | P0 | I2V 分支 |
| `model/base.py` | BaseModel 新增 CLIP 编码器初始化 + 条件编码 | P0 | I2V 分支 |
| `model/dmd.py` | DMD 训练中传递 I2V 条件 | P0 | I2V 分支 |
| `trainer/distillation.py` | Trainer 数据加载 + 条件编码 | P0 | I2V 分支 |
| `utils/dataset.py` | TextImagePairDataset 增强（适配训练） | P1 | 新增类 |
| `inference.py` | 推理入口增加 CLIP 条件编码 | P0 | I2V 分支 |
| `configs/default_config.yaml` | 新增 I2V 相关默认配置 | P0 | 新增字段 |

### 不修改的文件

| 文件路径 | 原因 |
|----------|------|
| `wan/modules/causal_model.py` | 已原生支持 `model_type='i2v'`（img_emb + i2v_cross_attn + clip_fea + y） |
| `wan/modules/model.py` | 原始 WanModel，已支持 I2V |
| `wan/modules/clip.py` | CLIP 模型实现，不需要修改 |
| `train.py` | 训练入口仅是调度器，实际逻辑在 trainer/ 中 |

## 3. 详细修改规格

### 3.1 utils/wan_wrapper.py

**改动点 1**: `WanDiffusionWrapper.__init__` 中，I2V 模型需要将 `model_type='i2v'` 传入 CausalWanModel

**改动点 2**: `WanDiffusionWrapper.forward` 中，新增 `clip_fea` 和 `y` 参数透传

**改动点 3**: 新增 `WanCLIPEncoder` 类（封装 CLIP 视觉编码器）

### 3.2 model/base.py

**改动点**: `BaseModel._initialize_models` 中，当 `args.i2v=True` 时：
- 创建 `WanCLIPEncoder` 实例
- 在 `_run_generator` 中编码图像条件

### 3.3 pipeline/self_forcing_training.py

**改动点**: `inference_with_trajectory` 新增 `clip_fea` + `y` 参数，传递给 generator

### 3.4 trainer/distillation.py

**改动点**: `fwdbwd_one_step` 中，I2V 路径新增 CLIP 编码 + y 构造

### 3.5 inference.py

**改动点**: I2V 路径新增 CLIP 模型加载 + 图像条件编码

### 3.6 pipeline/causal_inference.py & causal_diffusion_inference.py

**改动点**: `inference` 方法新增 `clip_fea` + `y` 参数透传给 generator

## 4. 实施顺序

```
Step 1: utils/wan_wrapper.py — 新增 WanCLIPEncoder + WanDiffusionWrapper I2V 支持
Step 2: model/base.py — BaseModel 初始化 CLIP + 条件编码
Step 3: pipeline/self_forcing_training.py — 训练 pipeline 传递 I2V 条件
Step 4: pipeline/teacher_forcing_training.py — 同上
Step 5: pipeline/causal_inference.py — 推理 pipeline 传递 I2V 条件
Step 6: pipeline/causal_diffusion_inference.py — 同上
Step 7: model/dmd.py — DMD 训练传递 I2V 条件
Step 8: trainer/distillation.py — Trainer 数据加载 + 条件编码
Step 9: inference.py — 推理入口添加 CLIP 编码
Step 10: 验证 checkpoint 兼容性
```

## 5. I2V 条件注入方式（对齐 Wan2.1 官方）

### CLIP 路径
```
参考图像 → CLIP ViT-H/14 (前31层, use_31_block=True) → [B, 257, 1280]
→ img_emb (MLPProj: LayerNorm → Linear → GELU → Linear → LayerNorm) → [B, 257, dim]
→ concat 到 text context 前面 → cross-attention
```

### VAE 路径 (image latent concat)
```
参考图像 → resize to (H, W) → VAE encode → image_latent [16, F_lat, H_lat, W_lat]
→ 构造 mask (第1帧=1, 其余=0) → y = concat(mask, image_latent) [32, F_lat, H_lat, W_lat]
→ concat 到 noisy latent (x) 的 channel 维度 → patch_embedding 输入 32ch
```

### 关键细节
- CausalWanModel 当 `model_type='i2v'` 时，`patch_embedding` 输入是 `in_dim=16` 但 `y` 通过 `torch.cat([x, y], dim=0)` 在 channel 维 concat，实际输入 32ch
- WanI2VCrossAttention 中 `context[:, :257]` 是 CLIP tokens，`context[:, 257:]` 是 text tokens
- CLIP visual 输出 257 tokens（1 CLS + 256 patches），使用前31层特征而非最终输出

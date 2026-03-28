# 已完成的代码修改记录

## 修改时间
2026-03-19

## 修改策略
采用**原地修改**策略，不新建模型/pipeline/trainer文件，所有改动通过条件分支完成。
I2V 条件（clip_fea + y）通过 conditional_dict 自然流动，无需修改训练 pipeline 核心逻辑。

---

## 1. utils/wan_wrapper.py

### 1.1 新增 import
```python
import torch.nn.functional as F
from wan.modules.clip import CLIPModel
```

### 1.2 新增 WanCLIPEncoder 类
- 封装 CLIP 视觉编码器，用于 I2V 条件注入
- 对齐 Wan2.1 官方实现：使用 xlm-roberta-large-ViT-H-14，前31层特征
- 输入：[B, C, H, W] 或 [B, C, 1, H, W]，值域 [-1, 1]
- 输出：[B, 257, 1280] (1 CLS + 256 patches)

### 1.3 WanDiffusionWrapper.forward 修改
- 从 conditional_dict 中提取 clip_fea 和 y
- 通过 `**i2v_kwargs` 透传给底层模型
- **安全检查**：只有当 `model.model_type == 'i2v'` 时才传递 clip_fea 和 y
  → 防止 T2V score models 收到 I2V 条件导致 channel 不匹配

---

## 2. model/base.py

### 2.1 新增 import
```python
from utils.wan_wrapper import WanCLIPEncoder
```

### 2.2 _initialize_models 修改
- 新增 `self.i2v` 标志（提前到 real/fake_name 计算前）
- **I2V 模式下默认模型名变更**：
  - `real_name` 默认 `"Wan2.1-I2V-14B-720P"` → **14B I2V 教师模型**
  - `fake_name` 默认 `"Wan2.1-Fun-1.3B-InP"` → **1.3B I2V 学生 critic**
- T2V 模式下保持原有默认值 `"Wan2.1-T2V-1.3B"`
- `real_score` 和 `fake_score` 以 `is_causal=False` 加载 → 使用 `WanModel`（非 causal）
  - 当模型目录的 `config.json` 包含 `model_type='i2v'` 时，`WanModel` 会自动创建 `img_emb` 和 `i2v_cross_attn`
  - `WanDiffusionWrapper.forward` 中的 `model_type == 'i2v'` 检查确保 I2V 条件正确传递
- CLIP 编码器从 `real_name` 目录加载（优先 14B 模型目录，也支持 `clip_model_dir` 覆盖）

---

## 3. trainer/distillation.py

### 3.1 新增 import
```python
import numpy as np
import torch.nn.functional as TF_func
```

### 3.2 __init__ 修改
- CLIP 编码器移到 GPU（不需要 FSDP，只做推理）

### 3.3 fwdbwd_one_step 修改（I2V 条件编码）
- I2V 模式下从 batch 获取参考图像
- CLIP 编码：ref_image → clip_encoder → [B, 257, 1280]
- y 构造（对齐官方 Wan2.1 I2V 实现）：
  - 参考图像 resize → VAE 编码 → [16, F_lat, lat_h, lat_w]
  - mask 构造：首帧=1，其余=0，repeat 4x → [4, F_lat, lat_h, lat_w]
  - y = concat(mask[4ch], latent[16ch]) = [20, F_lat, lat_h, lat_w]
  - 格式：list of [20, F_lat, H, W]（对齐 WanModel 输入格式）
- 将 clip_fea 和 y 注入 conditional_dict 和 unconditional_dict

---

## 4. inference.py

### 4.1 新增 import
```python
from utils.wan_wrapper import WanCLIPEncoder
```

### 4.2 CLIP 编码器初始化
- I2V 模式下加载 WanCLIPEncoder
- 自动从 config 推断 CLIP 模型目录

### 4.3 I2V 条件编码
- CLIP 编码参考图像
- y 构造：对齐官方实现（4ch mask + 16ch latent = 20ch）
- 传递 clip_fea 和 y 给 pipeline.inference()

---

## 5. pipeline/causal_inference.py

### 5.1 inference 方法签名扩展
- 新增 `clip_fea` 和 `y` 可选参数
- 注入到 conditional_dict 中

---

## 6. pipeline/causal_diffusion_inference.py

### 6.1 inference 方法签名扩展
- 新增 `clip_fea` 和 `y` 可选参数
- 注入到 conditional_dict 和 unconditional_dict 中

---

## 7. 不需要修改的文件

| 文件 | 原因 |
|------|------|
| `wan/modules/causal_model.py` | 已原生支持 I2V (model_type='i2v', img_emb, i2v_cross_attn, clip_fea, y) |
| `wan/modules/model.py` | WanModel 已支持 I2V |
| `pipeline/self_forcing_training.py` | I2V 条件通过 **conditional_dict 自动传递 |
| `pipeline/teacher_forcing_training.py` | 同上 |
| `model/dmd.py` | conditional_dict 自动包含 clip_fea 和 y |
| `utils/dataset.py` | TextImagePairDataset 已返回 image 字段 |

---

## 8. 关键技术细节

### y (image latent + mask) 构造方式
```
参考图像 [3, H, W]
  → resize to (lat_h*8, lat_w*8)
  → 构造视频序列 [3, F, H_pixel, W_pixel] (首帧=图像, 其余=0)
  → VAE encode → image_latent [16, F_lat, lat_h, lat_w]

mask:
  → [1, F, lat_h, lat_w], 首帧=1, 其余=0
  → 首帧 repeat 4x (VAE temporal compression)
  → reshape → [4, F_lat, lat_h, lat_w]

y = concat(mask, image_latent) → [20, F_lat, lat_h, lat_w]
对应 in_dim = 36 = 16(x) + 20(y)
```

### CLIP 特征路径
```
参考图像 [C, 1, H, W]
  → CLIPModel.visual (use_31_block=True)
  → [B, 257, 1280]
  → CausalWanModel.img_emb (MLPProj)
  → [B, 257, dim]
  → concat 到 text context 前面
  → i2v_cross_attention 中 context[:, :257] 是 CLIP tokens
```

### 条件传递链
```
trainer/distillation.fwdbwd_one_step
  → conditional_dict["clip_fea"] = clip_fea
  → conditional_dict["y"] = y_list
  → model/dmd.generator_loss(conditional_dict)
    → model/base._run_generator(conditional_dict)
      → _consistency_backward_simulation(**conditional_dict)
        → pipeline.inference_with_trajectory(**conditional_dict)
          → WanDiffusionWrapper.forward(conditional_dict=conditional_dict)
            → CausalWanModel.forward(clip_fea=..., y=...)
```

### 安全机制
- WanDiffusionWrapper 通过 `model.model_type == 'i2v'` 检查
  → T2V score models 不会收到 clip_fea 和 y
  → 避免 channel 不匹配错误

---

## 9. 配置要求

I2V 模式下需要在配置中设置：
```yaml
i2v: true
model_kwargs:
  model_name: "Wan2.1-Fun-1.3B-InP"  # 或 "Wan2.1-I2V-14B-720P"
clip_model_dir: "wan_models/Wan2.1-Fun-1.3B-InP"  # CLIP 模型路径
```

---

## 10. 验证结果 (2026-03-19)

### ✅ 1. CLIP 编码器加载
- **权重文件存在**：`wan_models/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (4.44 GB) ✓
- **Tokenizer 存在**：`wan_models/Wan2.1-Fun-1.3B-InP/xlm-roberta-large/` 目录及文件齐全 ✓
- **14B 模型同样具备**：`wan_models/Wan2.1-I2V-14B-720P/` 下同名文件也存在 ✓
- **CLIPModel 构造函数**：接受 `dtype, device, checkpoint_path, tokenizer_path`，与 `WanCLIPEncoder.__init__` 传入参数对齐 ✓
- **visual 方法输入格式**：list of `[C, T, H, W]`，经 `u.transpose(0,1)` 变为 `[T, C, H, W]`，与 `WanCLIPEncoder.forward` 中 `[img for img in images]` → list of `[C, 1, H, W]` 兼容 ✓
- **use_31_block=True**：VisionTransformer 返回前 31 层特征 `[B, 257, 1280]` ✓

### ✅ 2. VAE 编码
- **官方实现**对照确认：`image2video.py` 中 `self.vae.encode([video_tensor])[0]` 返回 `[16, F_lat, lat_h, lat_w]`
- **我们的实现**：`self.model.vae.model.encode(video_frames.unsqueeze(0), [mean, 1/std]).float().squeeze(0)` → `[16, F_lat, lat_h, lat_w]`
- 两者等价 ✓

### ✅ 3. y 维度对齐
- **mask 构造验证**（实际运行确认）：
  - F=5 → mask concat+reshape → `[4, 2, 2, 2]`，首帧全 1、次帧全 0 ✓
  - F=81 → mask 最终 `[4, 21, lat_h, lat_w]` ✓
- **y = concat(mask[4ch], latent[16ch]) = [20, F_lat, lat_h, lat_w]** ✓
- **in_dim = 36 = 16(x) + 20(y)** ✓
- **Fun-1.3B-InP checkpoint 实际确认**：`patch_embedding.weight` shape = `[1536, 36, 1, 2, 2]`，输入 channel = 36 ✓

### ✅ 4. KV cache 维度
- **1.3B 模型**：dim=1536, num_heads=12, head_dim=128
- **14B 模型**：dim=5120, num_heads=40, head_dim=128
- 两者 **head_dim 相同 (128)**，KV cache 格式一致，无兼容性问题 ✓

### ✅ 5. I2V 权重加载
- **Fun-1.3B-InP checkpoint 包含**：
  - `img_emb.proj.{0,1,3,4}.{weight,bias}` (8 keys) — MLPProj 权重 ✓
  - `blocks.N.cross_attn.{k_img,v_img,norm_k_img}.{weight,bias}` (158 keys) — i2v_cross_attn 权重 ✓
  - `patch_embedding.weight` shape = `[1536, 36, 1, 2, 2]` — in_dim=36 ✓
- **I2V-14B checkpoint 同样包含**：`img_emb` (8 keys) + cross_attn (608 keys) ✓
- **CausalWanModel.from_pretrained**：从 `config.json` 读取 `model_type='i2v'`, `in_dim=36` → 正确实例化含 `img_emb` 和 `i2v_cross_attn` 的模型 → `load_state_dict` 匹配 ✓

### ✅ 6. 条件传递链完整性
- `conditional_dict["clip_fea"]` 和 `conditional_dict["y"]` 在 trainer 中设置 → 传递到 `model.generator_loss` → `_run_generator` → `**conditional_dict` 展开 → pipeline → `WanDiffusionWrapper.forward` → `CausalWanModel._forward_train`/`_forward_inference` ✓
- T2V score models 的安全保护：`WanDiffusionWrapper` 检查 `model.model_type == 'i2v'` → T2V 模型不传 clip_fea/y ✓

### ⚠️ 7. 需注意事项

1. ~~Score models 使用 T2V 模型~~ **已修复**：`real_score` 和 `fake_score` 现在默认使用 I2V 模型（教师14B + 学生1.3B），与 generator 保持一致。
2. **不需要额外拉取模型权重**：所有必需的权重文件（CLIP、T5、VAE、DiT）在 `wan_models/Wan2.1-Fun-1.3B-InP/` 和 `wan_models/Wan2.1-I2V-14B-720P/` 中均已齐全。
3. **端到端测试**：代码逻辑和维度验证均通过，但仍需实际运行测试确认无运行时错误（如 GPU 内存、dtype 转换等）。
4. **显存注意**：14B 教师模型较大（~28GB BF16），需确保 FSDP 分片策略能在多卡上有效分摊。建议 `real_score_fsdp_wrap_strategy` 使用 `FULL_SHARD`。
5. **配置覆盖**：用户可通过 `real_name` 和 `fake_name` 配置项覆盖默认模型名，例如两个都用 1.3B I2V 进行轻量实验。

### 🔧 8. 发现的需修复问题

**暂未发现需要修复的问题。** 所有代码路径逻辑正确，维度对齐，权重文件齐全。

---

## 11. I2V Score Models 支持 (2026-03-19 更新)

### 需求
教师模型使用 14B I2V，学生模型使用 1.3B I2V。real_score 和 fake_score 都需要支持 I2V 条件输入。

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `model/base.py` | I2V 模式下 real_name 默认 "Wan2.1-I2V-14B-720P"，fake_name 默认 "Wan2.1-Fun-1.3B-InP" |

### 为什么只改了一个文件？

**因为现有代码架构已经天然支持 I2V score models：**

1. `WanModel`（非 causal）已原生支持 `model_type='i2v'`：
   - 当 config.json 中 `model_type='i2v'` 时，自动创建 `img_emb` 和 `i2v_cross_attn`
   - `_forward` 中 `assert clip_fea is not None and y is not None`
   - `x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]` 正确 concat
   - `context = torch.concat([context_clip, context], dim=1)` 正确融合 CLIP

2. `WanDiffusionWrapper.forward` 中 `model_type == 'i2v'` 检查：
   - 14B I2V 教师 → `model_type='i2v'` → 传递 clip_fea + y ✅
   - 1.3B I2V 学生 → `model_type='i2v'` → 传递 clip_fea + y ✅
   - T2V 模型（如果有） → `model_type='t2v'` → 跳过 clip_fea + y ✅

3. `conditional_dict` 和 `unconditional_dict` 中已包含 `clip_fea` 和 `y`

4. FSDP wrap 对 WanModel 同样有效，wrap_strategy 由配置控制

### 模型参数对比

| 参数 | Generator (1.3B I2V causal) | Real Score (14B I2V) | Fake Score (1.3B I2V) |
|------|---------------------------|---------------------|----------------------|
| 模型类 | CausalWanModel | WanModel | WanModel |
| model_type | i2v | i2v | i2v |
| dim | 1536 | 5120 | 1536 |
| in_dim | 36 | 36 | 36 |
| out_dim | 16 | 16 | 16 |
| num_heads | 12 | 40 | 12 |
| num_layers | 30 | 40 | 30 |
| head_dim | 128 | 128 | 128 |
| is_causal | True | False | False |
| requires_grad | True | False | True |
| 权重来源 | Wan2.1-Fun-1.3B-InP | Wan2.1-I2V-14B-720P | Wan2.1-Fun-1.3B-InP |

---

## 12. Stage 1-3 完整 I2V 训练流程支持 (2026-03-19 更新)

### 需求
我们没有现成的少步 causal I2V 模型，需要从 Stage 1 开始完整训练。

### 总体设计
在 BaseModel 中添加通用 `encode_i2v_conditions()` 方法，各 stage 的 model 和 trainer 通过条件分支复用。

### 代码审查修复 (第二轮)

| 问题 | 文件 | 修复内容 |
|------|------|----------|
| LatentLMDBDataset 无 image 字段 | `trainer/diffusion.py` | I2V 条件编码改为：优先 `batch["image"]`，fallback 到 clean_latent 首帧解码 |
| ODE trainer 有未使用变量 | `trainer/ode.py` | 移除 `first_frame_bchw` 等未使用变量 |
| DMD 使用 TextDataset 但 I2V 需要参考图 | `trainer/distillation.py` | I2V 模式改用 LatentLMDBDataset；参考图从 clean_latent 首帧解码 |
| DMD VAE 未在 I2V 模式加载到 GPU | `trainer/distillation.py` | 添加 `getattr(config, 'i2v', False)` 条件加载 VAE |
| Stage 1 配置缺少 `causal: true` | `configs/ar_diffusion_tf_framewise_i2v.yaml` | 添加 `causal: true` |
| 缺少 ODE 数据生成脚本 (I2V) | `get_causal_ode_data_framewise_i2v.py` | 新建：支持 CLIP+y 条件的 ODE trajectory 生成 |
| 缺少训练主脚本 | `scripts/train_i2v_all_stages.sh` | 新建：一键运行 Stage 1→2→3 全流程 |

### 修改文件列表

| 文件 | 阶段 | 修改内容 |
|------|------|----------|
| `model/base.py` | 通用 | 新增 `encode_i2v_conditions()` 通用方法（CLIP编码 + y构造） |
| `model/diffusion.py` | Stage 1 | `_initialize_models` 添加 I2V CLIP 编码器初始化 |
| `model/ode_regression.py` | Stage 2 ODE | `_initialize_models` 添加 I2V CLIP 编码器初始化 |
| `model/naive_consistency.py` | Stage 2 CD | `_initialize_models` 添加 I2V CLIP + VAE 初始化 |
| `trainer/diffusion.py` | Stage 1 | CLIP/VAE GPU 初始化 + I2V 条件编码（支持 image/latent fallback） |
| `trainer/ode.py` | Stage 2 ODE | CLIP/VAE GPU 初始化 + I2V 条件编码（从 ODE latent 解码首帧） |
| `trainer/naive_cd.py` | Stage 2 CD | CLIP/VAE GPU 初始化 + I2V 条件编码（从 clean_latent 解码首帧） |
| `trainer/distillation.py` | Stage 3 DMD | I2V 用 LatentLMDBDataset + VAE GPU 加载 + 参考图 fallback |

### 新增文件

| 文件 | 说明 |
|------|------|
| `configs/ar_diffusion_tf_framewise_i2v.yaml` | Stage 1 I2V AR Diffusion 配置 |
| `configs/causal_ode_framewise_i2v.yaml` | Stage 2 ODE I2V 配置 |
| `configs/causal_cd_framewise_i2v.yaml` | Stage 2 CD I2V 配置 |
| `configs/causal_forcing_dmd_framewise_i2v_local.yaml` | Stage 3 DMD I2V 配置（已更新） |
| `get_causal_ode_data_framewise_i2v.py` | Stage 2 ODE 数据生成脚本（I2V 版） |
| `scripts/train_i2v_all_stages.sh` | 一键训练脚本 |

### 训练流程总览

```
Stage 1: AR Diffusion Training
  ├─ 输入: dataset/clean_data_i2v (LatentLMDBDataset)
  ├─ 模型: Wan2.1-Fun-1.3B-InP → CausalWanModel (causal, I2V)
  ├─ 训练: Teacher Forcing + causal attention mask
  └─ 产出: 多步 causal AR Diffusion I2V 模型
       │
       ▼
Stage 2 Data: ODE Trajectory 生成
  ├─ 输入: dataset/clean_data_i2v + Stage 1 checkpoint
  ├─ 脚本: get_causal_ode_data_framewise_i2v.py
  └─ 产出: dataset/ODE6KCausal_framewise_i2v (trajectory pairs)
       │
       ▼
Stage 2: ODE / CD Distillation
  ├─ 输入: ODE trajectory 数据 (ODE) 或 clean_data_i2v (CD)
  ├─ 教师: Stage 1 产出的 causal I2V 模型
  ├─ 学生: CausalWanModel (目标: 少步)
  └─ 产出: 少步 (4步) causal I2V 模型
       │
       ▼
Stage 3: DMD Distillation
  ├─ 输入: dataset/clean_data_i2v (LatentLMDBDataset)
  ├─ 教师 (real_score): Wan2.1-I2V-14B-720P (双向, 14B)
  ├─ 学生 (generator): Stage 2 产出 (causal, 1.3B)
  ├─ Critic (fake_score): Wan2.1-Fun-1.3B-InP (双向, 1.3B)
  └─ 产出: 最终流式实时 I2V 模型
```

### 各 Stage 训练指令

**一键运行所有 Stage (推荐):**
```bash
bash scripts/train_i2v_all_stages.sh all 8
```

**分步运行:**

```bash
# Stage 1: AR Diffusion Training
bash scripts/train_i2v_all_stages.sh 1 8

# Stage 2: ODE 数据生成
bash scripts/train_i2v_all_stages.sh 2ode_data 8

# Stage 2: ODE Distillation
bash scripts/train_i2v_all_stages.sh 2ode 8

# Stage 3: DMD Distillation
bash scripts/train_i2v_all_stages.sh 3 8
```

**也可以手动运行:**

```bash
# Stage 1
torchrun --nproc_per_node=8 train.py \
    --config_path configs/ar_diffusion_tf_framewise_i2v.yaml \
    --logdir outputs/stage1_i2v --disable-wandb

# Stage 2 ODE Data
torchrun --nproc_per_node=8 get_causal_ode_data_framewise_i2v.py \
    --output_folder dataset/ODE6KCausal_framewise_i2v \
    --rawdata_path dataset/clean_data_i2v \
    --generator_ckpt outputs/stage1_i2v/checkpoint_model_XXXXXX/model.pt \
    --guidance_scale 6.0

# Stage 2 ODE Training
torchrun --nproc_per_node=8 train.py \
    --config_path configs/causal_ode_framewise_i2v.yaml \
    --logdir outputs/stage2_ode_i2v --disable-wandb

# Stage 3 DMD
torchrun --nproc_per_node=8 train.py \
    --config_path configs/causal_forcing_dmd_framewise_i2v_local.yaml \
    --logdir outputs/stage3_dmd_i2v --disable-wandb
```

### 各 Stage 数据需求

| Stage | 数据类型 | Dataset 类 | 数据路径 |
|-------|---------|-----------|---------|
| Stage 1 | clean latent + prompt | LatentLMDBDataset | `dataset/clean_data_i2v` |
| Stage 2 Data | clean latent + prompt | LatentLMDBDataset | `dataset/clean_data_i2v` (输入) |
| Stage 2 ODE | ODE trajectory + prompt | ODERegressionLMDBDataset | `dataset/ODE6KCausal_framewise_i2v` |
| Stage 2 CD | clean latent + prompt | LatentLMDBDataset | `dataset/clean_data_i2v` |
| Stage 3 DMD | clean latent + prompt | LatentLMDBDataset (I2V模式) | `dataset/clean_data_i2v` |

### 各 Stage I2V 条件来源

| Stage | 参考图像来源 | 说明 |
|-------|-------------|------|
| Stage 1 | `batch["image"]` → fallback clean_latent 首帧解码 | LatentLMDB 无 image 字段时自动解码 |
| Stage 2 ODE | ode_latent 中 clean latent 首帧解码 | ODE dataset 包含 latent |
| Stage 2 CD | clean_latent 首帧解码 | LatentLMDB dataset 包含 latent |
| Stage 3 DMD | `batch["image"]` → fallback clean_latent 首帧解码 | LatentLMDB 自动 fallback |

### 前置数据准备

1. **I2V 视频数据**：需要将视频数据预处理为 LMDB 格式（clean latent + prompt）
   - 格式参考：`utils/dataset.py` 中 `LatentLMDBDataset`
   - 存放路径：`dataset/clean_data_i2v/`
   - 每条数据包含：`prompts` (str) + `latents` (float16, shape `[1, 21, 16, 60, 104]`)

2. **模型权重**：确保以下目录存在
   - `wan_models/Wan2.1-Fun-1.3B-InP/` (1.3B I2V 模型)
   - `wan_models/Wan2.1-I2V-14B-720P/` (14B I2V 教师，Stage 3 需要)
   - `wan_models/Wan2.1-T2V-1.3B/` (T5 tokenizer)

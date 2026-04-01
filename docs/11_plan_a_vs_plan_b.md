# 方案 A vs 方案 B 对比文档

## 背景

I2V 推理时发现帧数不对齐问题（训练 21 帧 vs 推理 22 帧），有两种修复方案。

## 方案 A：`independent_first_frame=True`（原始 Causal-Forcing 设计）

**核心思路**：首帧独立去噪，与后续帧使用不同的 timestep。推理时传入 `initial_latent`。

### 配置

| 参数 | 值 |
|------|-----|
| `independent_first_frame` | `true` |
| `num_frame_per_block` | `3` |
| 总帧数 | `22`（`1 + 3*7 = 22`） |
| 分块方式 | `[1, 3, 3, 3, 3, 3, 3, 3]` = 22 帧 |
| 首帧处理 | 独立 timestep，独立去噪 |

### 文件对应

| 阶段 | 文件 |
|------|------|
| 阶段1 训练配置 | `configs/ar_diffusion_tf_framewise_i2v_plan_a.yaml` |
| 阶段1 & 阶段2 推理 | `inference.py`（方案A = 原始代码） |
| ODE 数据生成 | `get_causal_ode_data_framewise_i2v.py` |
| 阶段2 ODE 训练配置 | `configs/causal_ode_framewise_i2v_plan_a.yaml` |

### 推理逻辑

- 传入 `initial_latent`（编码后的首帧 latent）
- 噪声帧数 = `num_output_frames - 1` = 21 帧
- `y` 条件帧数 = 22 帧（包含 initial_latent 首帧）
- 推理管线中 `initial_latent` 作为 context 缓存到 KV cache
- `(22-1) / 3 = 7` ✅

### 注意事项

- **需要重新训练阶段1**（约 3k 步）
- 推理默认 `--num_output_frames 22`

---

## 方案 B：`independent_first_frame=False`

**核心思路**：所有帧统一处理，首帧在第一个 block 中与其他帧一起去噪。不传 `initial_latent`。

### 配置

| 参数 | 值 |
|------|-----|
| `independent_first_frame` | `false` |
| `num_frame_per_block` | `3` |
| 总帧数 | `21`（`3*7 = 21`） |
| 分块方式 | `[3, 3, 3, 3, 3, 3, 3]` = 21 帧 |
| 首帧处理 | 与第 2、3 帧共享 timestep |

### 文件对应

| 阶段 | 文件 |
|------|------|
| 阶段1 训练配置 | `configs/ar_diffusion_tf_framewise_i2v.yaml` |
| 阶段1 & 阶段2 推理 | `inference_plan_b.py` |
| ODE 数据生成 | `get_causal_ode_data_framewise_i2v.py` |
| 阶段2 ODE 训练配置 | `configs/causal_ode_framewise_i2v.yaml` |

### 推理逻辑

- **不传** `initial_latent`
- 噪声帧数 = `num_output_frames` = 21 帧
- `y` 条件帧数 = 21 帧
- 所有帧由模型从噪声去噪生成，首帧通过 `y` 条件（mask + image_latent）引导
- `21 / 3 = 7` ✅

### 注意事项

- **不需要重新训练阶段1**，使用现有权重
- 推理需指定 `--num_output_frames 21`

---

## ODE 阶段的关键区别

### `_prepare_generator_input` 中的首帧处理

两种方案在 ODE 训练中都有 `index[:, 0] = len(self.denoising_step_list) - 1`，即 I2V 首帧始终使用最小噪声水平的 latent 作为输入。

区别在于 `_get_timestep` 的行为：

- **方案 A**：首帧 timestep 独立于后续帧，后续帧按 block_size=3 分组，总 22 帧
- **方案 B**：首帧与第 2、3 帧共享 timestep，所有帧按 block_size=3 分组，总 21 帧

### ODE 数据生成

两种方案使用相同的数据生成脚本 `get_causal_ode_data_framewise_i2v.py`，`num_frame_per_block` 均为 3。

区别在于：
- **方案 A**：需设置 `independent_first_frame=True`，总帧数 22
- **方案 B**：`independent_first_frame=False`（默认），总帧数 21

---

## 修改记录

### 2026-03-28

1. **`model/ode_regression.py`**：注释掉 `__init__` 中重复创建 generator 的代码（问题3修复）
2. **`get_causal_ode_data_framewise_i2v.py`**：
   - 修复 `tqdm(total_steps)` → `tqdm(range(total_steps))`（问题1）
   - 简化 `unconditional_dict` 的 I2V 条件设置（问题2）
3. **文件整理**：
   - `inference.py`：方案 A 推理脚本（原始代码，传 `initial_latent`，默认 22 帧）
   - `inference_plan_b.py`：方案 B 推理脚本（不传 `initial_latent`，21 帧噪声）
   - `configs/ar_diffusion_tf_framewise_i2v_plan_a.yaml`：方案 A 阶段1训练配置
   - `configs/causal_ode_framewise_i2v_plan_a.yaml`：方案 A 阶段2 ODE 训练配置

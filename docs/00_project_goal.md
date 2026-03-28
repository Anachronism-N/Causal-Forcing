# I2V 流式自回归视频生成 - 项目目标

## 1. 项目概述

将 Causal-Forcing（基于 Wan2.1 T2V）扩展为支持 **Image-to-Video (I2V) 流式自回归视频生成** 的训练和推理框架。

## 2. 改造目标

### 2.1 条件扩展：T2V → I2V
- 新增参考图像作为生成条件
- 支持 CLIP 图像编码（cross-attention context 注入）
- 支持 VAE 图像 latent 注入（channel concat 到 noisy latent）
- 严格对齐 Wan2.1 官方 I2V 实现（`WanI2VModel`）

### 2.2 生成方式：支持流式自回归
- 基于 Causal-Forcing 已有的 block-wise causal attention 机制
- 每次生成一个 temporal chunk（对应一个 causal block）
- 支持按 chunk 逐段输出，实现流式生成

### 2.3 训练方式：chunk continuation training
- 训练时支持 参考图像 + 文本 + 历史 chunks（clean latent） + 目标 chunk（noisy）
- causal mask 跨 chunk 生效
- 支持 history corruption / scheduled sampling（后期）

## 3. 非目标（第一版不做）
- 无限长 KV cache 累积（第一版用简单重算 + 滑动窗口）
- Compressed memory token
- 多参考图像输入
- 视频编辑 / inpainting 任务

## 4. 里程碑

| 阶段 | 目标 | 验收标准 |
|------|------|----------|
| Phase 1 | 基础 I2V + causal 训练 | ref_image + text → 视频，启用 causal block mask 和 pyramid noise |
| Phase 2 | 流式自回归推理 | init → generate_chunk 循环，可生成多 chunk 视频 |
| Phase 3 | Streaming continuation 训练 | 训练时 history chunks 作为已知 blocks，只对 target chunk 计算 loss |
| Phase 4 | 优化与稳定 | overlap blend、显存优化、history curriculum |

## 5. 兼容性要求
- 保留 T2V 模式完全可用（通过 `task_type` 配置切换）
- 支持加载现有 T2V Causal-Forcing checkpoint（strict=False，新增模块随机/零初始化）
- 配置驱动切换 `t2v` / `i2v` / `i2v_streaming`

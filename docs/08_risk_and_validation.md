# 风险分析与验证计划

## 1. 技术风险

### 1.1 条件注入可能影响 Causal Attention 质量

**风险**: image tokens concat 到 cross-attn context 后，causal block mask 的设计可能与新增的 image tokens 交互异常。

**分析**: 
- Causal block mask 作用于 self-attention（空间+时间 tokens 之间）
- Cross-attention（query=spatial tokens, key/value=context tokens）不受 causal mask 影响
- 因此 image tokens 通过 cross-attn 注入，**不与 causal mask 冲突**

**验证方法**:
- 确认 `flash_attention_causal` 只作用于 self-attn，不影响 cross-attn
- 跑一个小 forward pass，确认梯度流正常

**结论**: **低风险**。Cross-attention 和 self-attention 是独立的路径。

### 1.2 Causal Block Mask 对 image_latent concat 通道的影响

**风险**: image_latent 在 channel dim concat 后通过 patch_embedding 映射到 token space，然后参与 self-attention。如果 image_latent 在所有时间帧上相同（因为 repeat），causal mask 可能导致不同 block 看到的 image 信号不一致。

**分析**:
- image_latent repeat 后，每个时间帧的 image 信息完全相同
- 通过 patch_embedding（Conv3d）后，image 信息已融入 token embedding
- Causal mask 只控制 attention 可见范围，不影响各 token 自身的 embedding
- 前面的 block 看不到后面 block 的 token，但每个 block 自己的 token 中已包含 image 信息
- 这与 Wan2.1 I2V 原始设计一致

**结论**: **低风险**。

### 1.3 Pyramid Noise 与 I2V 首帧一致性

**风险**: Causal-Forcing 的 pyramid noise 对不同 block 使用不同噪声等级。第一个 block（包含首帧）的噪声可能使首帧与参考图像不一致。

**分析**:
- 在推理时，image_latent 作为 channel concat 的条件始终存在
- 首帧的重建质量取决于模型学到的 image condition → first frame 映射
- Wan2.1 I2V 的原始设计就是这样工作的：noisy_latent 全程都有噪声，image_latent concat 提供参考
- Pyramid noise 中第一个 block 的噪声等级由调度器决定，不是特殊问题

**验证方法**:
- 训练初期检查首帧与参考图像的 SSIM/PSNR
- 必要时对第一个 block 使用更多去噪步数

**结论**: **中低风险**。是 I2V 任务的固有特性，不是 causal 改造引入的新问题。

### 1.4 流式推理时 History 重算的计算开销

**风险**: 每次生成新 chunk 时，需要将 history chunks 与新 chunk 一起输入模型。随着 history 增长，计算量线性增加。

**分析**:
- 当前设计: 每次 forward 将 history (clean) + new (noisy) 全部输入
- History chunks 数量受 `max_history_chunks` 限制（默认 8）
- 对于 14B 模型，8 个 history chunks 的 forward 开销约为完整视频的 2x（因为 attention 是 O(n²)）

**缓解措施**:
1. `max_history_chunks` 限制窗口大小
2. 后期可实现 KV cache（只对 history tokens 缓存 K/V，不重算）
3. 后期可实现 compressed memory（将远处 history 压缩为 summary tokens）

**结论**: **中风险**。第一版通过 max_history_chunks 限制可接受，后期需优化。

### 1.5 VAE 和 CLIP 额外显存

**风险**: 训练时需要同时加载 VAE、T5、CLIP 和 DiT 模型。

**分析**:
- VAE: 约 200M 参数 → ~0.8 GB (fp32) / ~0.4 GB (fp16)
- T5-XXL: 约 4.8B 参数 → 已在现有训练中加载
- CLIP-ViT-H/14: 约 630M 参数 → ~2.5 GB (fp32) / ~1.25 GB (fp16)
- 新增显存: ~1.25 GB (CLIP, fp16)

**缓解措施**:
1. CLIP 和 VAE 在编码后可 offload 到 CPU（推理模式下 no_grad）
2. 训练时 CLIP 和 VAE 冻结，不需存储梯度

**结论**: **低风险**。新增显存开销可控。

### 1.6 训练-推理不一致（Exposure Bias）

**风险**: Phase 1 训练时所有 block 都有噪声（pyramid noise），推理时 history blocks 是 clean 的。这种分布不一致可能导致推理质量下降。

**分析**:
- 这就是 Causal-Forcing 论文要解决的核心问题之一
- Causal-Forcing 的 causal mask + pyramid noise 设计本身就在缓解这个问题
- 进一步的改善需要 Phase 3 的 continuation training

**结论**: **中风险**。第一版可接受，Phase 3 会改善。

## 2. 验证计划

### 2.1 单元测试

| 测试项 | 验证内容 | 方法 |
|--------|---------|------|
| 模型构建 | WanCausalI2VModel 能正确构建 | 构建模型，打印参数统计 |
| Forward pass | 32ch 输入 + clip_fea 能正常前向 | 随机输入，检查输出 shape |
| Checkpoint 加载 | T2V ckpt 能正确加载到 I2V 模型 | 加载后检查 missing/unexpected keys |
| Causal mask | causal mask 与 I2V 条件不冲突 | 检查 attention 输出的梯度 |
| Loss 计算 | I2V 训练能产生有效 loss | 随机数据 forward+backward |

### 2.2 集成测试

| 测试项 | 验证内容 | 方法 |
|--------|---------|------|
| 训练循环 | I2V 训练能跑通完整一个 step | 小数据集 + 小模型 |
| 推理循环 | I2V 推理能生成视频 | 从随机/预训练 ckpt 推理 |
| 流式推理 | streaming 模式能逐 chunk 生成 | init_stream + generate_next_chunk × N |
| T2V 兼容 | I2V 模型加 clip_fea=None 能做 T2V | 运行现有 T2V 推理脚本 |

### 2.3 质量验证

| 阶段 | 指标 | 预期 |
|------|------|------|
| Phase 1 训练 10k steps | 训练 loss 下降 | loss 应稳定下降 |
| Phase 1 推理 | 首帧 vs 参考图像 SSIM | > 0.8 |
| Phase 1 推理 | 视频帧间一致性 | 无明显跳变 |
| Phase 2 流式 | chunk 边界平滑度 | 视觉连续，无明显断裂 |
| Phase 2 流式 | 多 chunk 语义一致性 | 动作连贯，不偏离 prompt |

## 3. 回滚方案

### 3.1 代码回滚
- 所有新增代码在独立文件中（`wan_causal_i2v.py`, `pipeline_i2v.py`, `inference_i2v.py`）
- 不修改现有 `wan_causal.py`, `pipeline.py`, `inference.py` 的核心逻辑
- 配置通过 `task_type` 切换，T2V 路径完全不受影响

### 3.2 训练回滚
- 如果 I2V 训练不收敛，可随时切回 T2V 训练（改配置即可）
- checkpoint 包含 model_type 标记，不会混淆

## 4. 已知限制（第一版）

1. **无 KV cache**: 每次 forward 重算所有 history tokens，效率非最优
2. **无 overlap blending**: chunk 边界可能有微小不连续
3. **无 image condition drop**: CFG 只 drop text，不 drop image
4. **固定 chunk size**: 所有 chunk 大小相同，不支持可变长度
5. **无 multi-scale**: 不支持不同分辨率的 streaming

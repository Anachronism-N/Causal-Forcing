# 流式自回归设计

## 1. 核心概念

### 1.1 Chunk 与 Causal Block 的关系
- Causal-Forcing 将时间帧分为 `num_causal_block` 个 block
- 每个 block 包含 `block_size` 个 latent 帧
- **一个 chunk = 一个 causal block**
- 自回归推理：每次生成一个新的 block（chunk），前面的 block 都是已知的

### 1.2 现有 CausalPipeline.generate_video 的自回归逻辑

```python
# pipeline.py 核心逻辑（已存在）
for idx in range(num_causal_block):
    if idx == 0:
        # 初始 block 从纯噪声开始
        pass
    else:
        # 历史 blocks 已经 clean
        # 只对当前 block 做去噪
        pass
    
    for step in denoising_steps:
        # 构造输入：前 idx 个 block 用 clean latent + 当前 block 用 noisy
        # 模型预测噪声
        # 只更新当前 block 的 latent
        pass
    
    # 当前 block 去噪完毕，加入 history
```

### 1.3 I2V 流式扩展的核心变化

在现有自回归循环基础上，需要：
1. **条件注入**：每次调用模型时传入 `clip_fea` 和 image_latent concat
2. **Streaming state**：管理编码好的条件和历史 chunks
3. **增量输出**：每生成一个 chunk 就可以输出

## 2. StreamingState 定义

```python
@dataclass
class StreamingState:
    """流式生成的状态对象"""
    # 条件（在 init 时编码，后续复用）
    text_tokens: Tensor           # [B, text_seq, dim]
    clip_fea: Tensor              # [B, 257, clip_dim]
    image_latent: Tensor          # [B, C, 1, H_lat, W_lat]  (未 repeat)
    null_context: Tensor          # [B, text_seq, dim] for CFG
    
    # 历史
    history_latents: List[Tensor] # 每个元素 [B, C, block_T, H_lat, W_lat]
    num_generated_chunks: int     # 已生成的 chunk 数
    
    # 空间配置
    latent_h: int                 # latent 高度
    latent_w: int                 # latent 宽度
    chunk_len: int                # 每个 chunk 的 latent 帧数
    
    # 推理配置
    max_history_chunks: int       # 最大历史窗口大小
    overlap_frames: int           # chunk 间重叠帧数（latent 空间）
```

## 3. 流式推理流程

### 3.1 初始化
```python
def init_stream(ref_image, prompt, config) -> StreamingState:
    # 1. T5 编码文本
    text_tokens = t5_encode(prompt)
    null_context = t5_encode("")
    
    # 2. CLIP 编码参考图像
    clip_fea = clip_model.visual(preprocess(ref_image))
    
    # 3. VAE 编码参考图像
    image_latent = vae.encode(ref_image)  # [B, C, 1, H, W]
    
    # 4. 初始化状态
    return StreamingState(
        text_tokens=text_tokens,
        clip_fea=clip_fea,
        image_latent=image_latent,
        null_context=null_context,
        history_latents=[],
        num_generated_chunks=0,
        ...
    )
```

### 3.2 生成下一个 Chunk
```python
def generate_next_chunk(state) -> Tensor:
    # 1. 确定当前窗口
    history = state.history_latents[-state.max_history_chunks:]
    num_history_blocks = len(history)
    total_blocks = num_history_blocks + 1  # +1 for new block
    
    # 2. 拼接 history latent + noise for new block
    if history:
        history_cat = torch.cat(history, dim=2)  # [B, C, sum_T, H, W]
    else:
        history_cat = None
    new_noise = torch.randn(B, C, chunk_len, H, W)
    
    # 3. 构造 image_latent concat
    total_T = (num_history_blocks + 1) * chunk_len
    image_latent_expanded = state.image_latent.repeat(1, 1, total_T, 1, 1)
    
    # 4. 去噪循环（只对最后一个 block）
    for step in denoising_steps:
        # 拼接 history(clean) + current(noisy)
        x = torch.cat([history_cat, noisy_current], dim=2) if history_cat else noisy_current
        model_input = torch.cat([x, image_latent_expanded], dim=1)  # channel concat
        
        noise_pred = model(
            model_input, timestep, 
            context=text_tokens, clip_fea=clip_fea,
            num_causal_block=total_blocks
        )
        
        # 只取最后一个 block 的预测来更新
        noise_pred_new = noise_pred[:, :, -chunk_len:]
        noisy_current = scheduler_step(noisy_current, noise_pred_new, step)
    
    # 5. 更新 state
    clean_chunk = noisy_current  # 去噪完成
    state.history_latents.append(clean_chunk)
    state.num_generated_chunks += 1
    
    # 6. 可选：窗口裁剪
    if len(state.history_latents) > state.max_history_chunks:
        state.history_latents = state.history_latents[-state.max_history_chunks:]
    
    return clean_chunk
```

## 4. 与训练的对齐

### 训练时的 Chunk Continuation（Phase 3）

训练时模拟推理场景：
输入: [history_chunks(clean) | target_chunk(noisy)]
↑ causal mask 可见 ↑ 要去噪的部分

Loss: 只在 target_chunk 上计算
这与 Causal-Forcing 已有的训练框架天然对齐：
- Causal-Forcing 的 pyramid noise 已经支持不同 block 不同噪声
- 只需将 history blocks 的噪声设为 0（clean），target block 的噪声正常
- Loss mask 只覆盖 target block

### Phase 1 训练（简化版）

Phase 1 不需要 explicit history：
- 整段视频作为输入
- 启用 causal block mask + pyramid noise
- 与现有 Causal-Forcing T2V 训练逻辑一致
- 唯一区别：增加了 I2V 条件注入

## 5. Overlap Blending

### 第一版：无重叠
- 直接拼接 chunks
- 简单但可能有边界不连续

### 第二版：线性混合
```python
def merge_chunks(chunks, overlap_frames):
    """
    chunks: List[Tensor], each [B, C, T, H, W]
    overlap_frames: int, latent 空间的重叠帧数
    """
    if overlap_frames == 0:
        return torch.cat(chunks, dim=2)
    
    result = chunks[0]
    for chunk in chunks[1:]:
        alpha = torch.linspace(1, 0, overlap_frames, device=chunk.device)
        alpha = alpha.view(1, 1, -1, 1, 1)
        
        overlap_prev = result[:, :, -overlap_frames:]
        overlap_curr = chunk[:, :, :overlap_frames]
        blended = alpha * overlap_prev + (1 - alpha) * overlap_curr
        
        result = torch.cat([
            result[:, :, :-overlap_frames],
            blended,
            chunk[:, :, overlap_frames:]
        ], dim=2)
    
    return result
```

## 6. 显存分析

每个 chunk latent: `[B, 16, chunk_T, H_lat, W_lat]`

以 480p（H_lat=60, W_lat=104）、chunk_T=4、B=1 为例：
- 单个 chunk: `16 × 4 × 60 × 104 × 4 bytes ≈ 6.4 MB` (float32)
- 8 个 history chunks ≈ 51 MB，完全可接受
- 瓶颈在模型 forward 时的 attention 显存，而非 history 存储

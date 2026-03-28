# Checkpoint 兼容性设计

## 1. 权重来源分析

### 1.1 T2V Causal-Forcing Checkpoint
- 模型类: `WanCausalModel`（继承自 `WanModel`）
- `patch_embedding` 输入通道: 16（标准 VAE latent channels）
- 无 `img_emb` 模块
- 包含完整 causal attention 训练权重
- 关键 module 列表:
  - `patch_embedding` (nn.Conv3d, in=16)
  - `text_embedding_padding` (nn.Parameter)
  - `time_embedding` (TimestepEmbedding)
  - `time_projection` (Timesteps)
  - `blocks` (nn.ModuleList of WanAttentionBlock，内含 causal attention)
  - `head` (Head)

### 1.2 Wan2.1 I2V 官方 Checkpoint
- 模型类: `WanI2VModel`（继承自 `WanModel`）
- `patch_embedding` 输入通道: 32（noisy_latent 16ch + image_latent 16ch）
- 有 `img_emb` 模块: `nn.Sequential(Linear(1280, dim), SiLU, Linear(dim, dim))`
- attention 是标准 full attention（非 causal block）
- 有预训练好的 image condition 权重

### 1.3 WanCausalI2VModel（目标模型）
- 在 WanCausalModel 基础上新增:
  - `patch_embedding` in_channels: 16 → 32
  - `img_emb`: nn.Sequential(Linear(1280, dim), SiLU, Linear(dim, dim))
- 其余结构与 WanCausalModel 完全一致

## 2. 加载策略

### 2.1 策略 A: 从 T2V Causal-Forcing 加载（推荐初始路径）

**优点**: 保留 causal attention 的训练成果，最核心的时序建模能力
**缺点**: 需要从头训练 image condition 相关模块

```python
def load_from_t2v_causal(model: WanCausalI2VModel, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    
    # === 处理 patch_embedding.proj.weight 通道扩展: [out, 16, kT, kH, kW] → [out, 32, kT, kH, kW] ===
    pe_key = "patch_embedding.proj.weight"
    if pe_key in state:
        old_w = state[pe_key]  # shape: [dim, 16, 1, 2, 2]
        new_w = torch.zeros(
            old_w.shape[0], 32, *old_w.shape[2:], 
            dtype=old_w.dtype
        )
        new_w[:, :16] = old_w  # 前16通道：复制旧权重
        # 后16通道：零初始化 → 初始时 image_latent concat 不贡献信号
        state[pe_key] = new_w
    
    # === 加载（允许 img_emb 缺失）===
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    # 预期 missing keys:
    #   img_emb.0.weight, img_emb.0.bias   (Linear: 1280 → dim)
    #   img_emb.2.weight, img_emb.2.bias   (Linear: dim → dim)
    assert all("img_emb" in k for k in missing), f"Unexpected missing keys: {missing}"
    
    # === img_emb 特殊初始化 ===
    # 第一层正常 kaiming 初始化
    nn.init.kaiming_normal_(model.img_emb[0].weight)
    nn.init.zeros_(model.img_emb[0].bias)
    # 最后一层零初始化 → 确保初始时 image tokens 全零，不干扰已训练的 text cross-attn
    nn.init.zeros_(model.img_emb[2].weight)
    nn.init.zeros_(model.img_emb[2].bias)
    
    return model
```

### 2.2 策略 B: 从 Wan2.1 I2V 官方加载

**优点**: 获得预训练好的 img_emb 和 32 通道 patch_embedding
**缺点**: attention 非 causal，需要 causal 能力重新学习

```python
def load_from_wan_i2v(model: WanCausalI2VModel, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    
    # WanI2VModel 与 WanCausalI2VModel 的 key 名称基本一致
    # （因为 WanCausalModel 的结构名沿用了 WanModel）
    # 直接加载，可能有少量 attention 相关 key 不匹配
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    print(f"Missing: {missing}")
    print(f"Unexpected: {unexpected}")
    
    return model
```

### 2.3 策略 C: 混合加载（高级，效果最佳）

从两个 checkpoint 分别取最优部分：

```python
def load_hybrid(model: WanCausalI2VModel, t2v_ckpt: str, i2v_ckpt: str):
    t2v_state = torch.load(t2v_ckpt, map_location="cpu")
    i2v_state = torch.load(i2v_ckpt, map_location="cpu")
    
    merged = {}
    
    for key in model.state_dict().keys():
        if key.startswith("img_emb."):
            # img_emb: 从 I2V 官方加载（已预训练）
            if key in i2v_state:
                merged[key] = i2v_state[key]
            # 若 i2v_state 无此 key，后续零初始化
        elif key == "patch_embedding.proj.weight":
            # patch_embedding: 前16ch 从 T2V，后16ch 从 I2V
            t2v_w = t2v_state[key]  # [out, 16, ...]
            i2v_w = i2v_state[key]  # [out, 32, ...]
            new_w = torch.zeros_like(i2v_w)
            new_w[:, :16] = t2v_w           # 前16ch: T2V causal 训练的权重
            new_w[:, 16:] = i2v_w[:, 16:]   # 后16ch: I2V 预训练的 image latent 通道权重
            merged[key] = new_w
        elif key == "patch_embedding.proj.bias":
            # bias 维度不变，从 T2V 加载
            if key in t2v_state:
                merged[key] = t2v_state[key]
        else:
            # 其余所有权重: 优先从 T2V causal 加载（保留 causal attention 训练成果）
            if key in t2v_state:
                merged[key] = t2v_state[key]
            elif key in i2v_state:
                merged[key] = i2v_state[key]
    
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing:
        print(f"Warning: Still missing keys: {missing}")
    
    return model
```

## 3. Key 映射对照表

| 模块 | WanCausalModel key | WanI2VModel key | WanCausalI2VModel key | 来源 |
|------|-------------------|-----------------|----------------------|------|
| patch_embed | `patch_embedding.proj.weight` [out,16,...] | `patch_embedding.proj.weight` [out,32,...] | `patch_embedding.proj.weight` [out,32,...] | 混合 |
| img_emb | — | `img_emb.0/2.weight/bias` | `img_emb.0/2.weight/bias` | I2V |
| text_embed_pad | `text_embedding_padding` | `text_embedding_padding` | `text_embedding_padding` | T2V |
| time_embed | `time_embedding.*` | `time_embedding.*` | `time_embedding.*` | T2V |
| blocks | `blocks.*.self_attn.*` | `blocks.*.self_attn.*` | `blocks.*.self_attn.*` | T2V |
| blocks | `blocks.*.cross_attn.*` | `blocks.*.cross_attn.*` | `blocks.*.cross_attn.*` | T2V |
| blocks | `blocks.*.ffn.*` | `blocks.*.ffn.*` | `blocks.*.ffn.*` | T2V |
| head | `head.*` | `head.*` | `head.*` | T2V |

## 4. 验证加载正确性

```python
def verify_loading(model, ref_state):
    """验证加载后的模型权重是否合理"""
    for name, param in model.named_parameters():
        if "img_emb.2" in name:
            # 最后一层应接近零（零初始化或 I2V 预训练值）
            print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
        elif "patch_embedding" in name and "weight" in name:
            # 前16ch 应有非零值，后16ch 取决于初始化策略
            w = param.data
            print(f"{name}: ch[:16] std={w[:,:16].std():.6f}, ch[16:] std={w[:,16:].std():.6f}")
```

## 5. 保存格式

训练时保存完整 state_dict，包含所有新增模块：

```python
def save_checkpoint(model, optimizer, step, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_type": "WanCausalI2VModel",  # 标记模型类型
    }, path)
```

## 6. 向后兼容

加载 I2V checkpoint 用于 T2V 推理（如果需要）：
- 将 `clip_fea` 传 None → img_emb 不执行
- 将 `image_latent` 传全零 → 后 16ch 输入为零
- 理论上行为与纯 T2V 模型等价（因零初始化）

# 数据管道设计

## 1. 现有数据管道分析

### 1.1 VideoDataset（data/dataset.py）

现有流程：
```python
class VideoDataset(Dataset):
    def __init__(self, data_dir, ...):
        # 扫描 video_dir 下的视频文件
        # 加载对应的 caption
    
    def __getitem__(self, idx):
        # 1. 读取视频帧
        # 2. resize/crop 到目标分辨率
        # 3. 读取 caption
        # 返回 {"video": tensor, "text": str}
```

### 1.2 BucketSampler（data/bucket.py）

现有流程：
- 按 (H, W, num_frames) 分桶
- 同一 batch 内视频具有相同分辨率和帧数

## 2. I2V 数据管道扩展

### 2.1 数据来源

**方案 A：视频第一帧作为参考图像**（推荐第一版）
- 无需额外数据
- 从视频中提取第一帧
- 保证参考图像与视频内容一致

**方案 B：独立参考图像**（后期扩展）
- 需要额外的 image-video pair 数据
- 支持更灵活的 I2V 场景

### 2.2 修改后的 __getitem__

```python
def __getitem__(self, idx):
    # === 现有 ===
    video = load_video(...)        # [T, C, H, W]
    text = load_caption(...)
    
    # === 新增（I2V 模式）===
    if self.task_type in ("i2v", "i2v_streaming"):
        # 提取参考图像（第一帧）
        ref_image = video[0]  # [C, H, W]
        
        # 用于 CLIP 编码的图像（需要 CLIP 预处理）
        clip_image = self.clip_preprocess(ref_image)  # [C, 224, 224]
    
    return {
        "video": video,          # [T, C, H, W]
        "text": text,
        "ref_image": ref_image,  # [C, H, W] (I2V only)
        "clip_image": clip_image # [C, 224, 224] (I2V only)
    }
```

### 2.3 Collate 和编码

在 train.py 中，batch 取出后：
```python
# VAE 编码（在训练循环中）
with torch.no_grad():
    video_latent = vae.encode(batch["video"])        # [B, 16, T, H, W]
    image_latent = vae.encode(batch["ref_image"])     # [B, 16, 1, H, W]
    clip_fea = clip_model.visual(batch["clip_image"]) # [B, 257, 1280]
```

**注意：VAE 和 CLIP 编码在训练循环中 on-the-fly 执行，不做预编码**
（原因：Causal-Forcing 目前就是这样做的，保持一致）

## 3. Bucket 策略

### 不变的部分
- 按 (H, W, T) 分桶
- 同 batch 内分辨率和帧数相同

### 新增约束
- I2V 模式下，参考图像分辨率与视频帧分辨率相同
- 不影响现有分桶逻辑

## 4. 数据增强

### 4.1 参考图像增强
- **空间变换**：与视频帧同步（random crop, flip）
- **颜色增强**：不做（避免参考图像和视频不一致）

### 4.2 Condition Drop（用于 CFG 训练）
- 以 `p_drop_text` 概率 drop 文本（替换为空字符串）
- 以 `p_drop_image` 概率 drop 图像（clip_fea 置零，image_latent 置零）
- 联合 drop 概率 = `p_drop_both`

## 5. 预编码方案（可选优化）

如果训练瓶颈在编码速度，可以考虑预编码：
预处理脚本
for each video:
video_latent = vae.encode(video)
image_latent = vae.encode(first_frame)
clip_fea = clip.visual(first_frame)
save(video_latent, image_latent, clip_fea)

训练时直接加载
dataset = LatentDataset(latent_dir)
第一版不做预编码，保持与现有代码一致。

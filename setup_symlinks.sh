#!/bin/bash
# 在模型权重复制完成后执行此脚本，将 wan_models 软链接指向更快的存储路径
# 使用方法: bash setup_symlinks.sh

set -e

FAST_STORAGE="/apdcephfs_gy2/share_302533218/cedricnie"
SLOW_STORAGE="/apdcephfs/wx_feature/home/cedricnie/LongLive/wan_models"
PROJECT_DIR="/apdcephfs/wx_feature/home/cedricnie/Causal-Forcing"

echo "=== 检查复制是否完成 ==="

# 检查关键文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        echo "错误: $1 不存在，复制可能尚未完成"
        return 1
    fi
    echo "  ✓ $(basename $1) 存在"
    return 0
}

ALL_OK=true

# 检查 Stage3 I2V 需要的关键模型文件
echo "检查 Wan2.1-Fun-1.3B-InP (generator + fake_score)..."
check_file "${FAST_STORAGE}/wan_models/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" || ALL_OK=false
check_file "${FAST_STORAGE}/wan_models/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" || ALL_OK=false
check_file "${FAST_STORAGE}/wan_models/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth" || ALL_OK=false

echo "检查 Wan2.1-I2V-14B-720P (real_score / teacher)..."
check_file "${FAST_STORAGE}/wan_models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors" || ALL_OK=false
check_file "${FAST_STORAGE}/wan_models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors" || ALL_OK=false
check_file "${FAST_STORAGE}/wan_models/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" || ALL_OK=false

echo "检查 Wan2.1-T2V-1.3B (text encoder + VAE)..."
check_file "${FAST_STORAGE}/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" || ALL_OK=false

echo "检查 Wan2.1-T2V-14B..."
check_file "${FAST_STORAGE}/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors" || ALL_OK=false
check_file "${FAST_STORAGE}/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors" || ALL_OK=false

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "部分文件尚未复制完成，请等待复制完成后再执行此脚本"
    echo "可以用以下命令检查复制进程: ps aux | grep 'cp -r' | grep -v grep"
    exit 1
fi

echo ""
echo "=== 所有关键文件已就绪 ==="

# 更新 wan_models 软链接
echo "更新 wan_models 软链接..."
rm -f "${PROJECT_DIR}/wan_models"
ln -s "${FAST_STORAGE}/wan_models" "${PROJECT_DIR}/wan_models"
echo "  wan_models -> ${FAST_STORAGE}/wan_models"

echo ""
echo "=== 软链接更新完成 ==="
echo "当前 wan_models 指向: $(readlink -f ${PROJECT_DIR}/wan_models)"
echo ""
echo "原始模型权重仍保留在: ${SLOW_STORAGE}"
echo "如需回退: ln -sf ${SLOW_STORAGE} ${PROJECT_DIR}/wan_models"

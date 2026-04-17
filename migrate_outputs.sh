#!/bin/bash
# 迁移 outputs 目录到快速存储
# 用法: bash migrate_outputs.sh

set -e

SRC="/apdcephfs/wx_feature/home/cedricnie/Causal-Forcing/outputs"
DST="/apdcephfs_gy2/share_302533218/cedricnie/outputs"

echo "=========================================="
echo "迁移 outputs 到快速存储"
echo "源: $SRC"
echo "目标: $DST"
echo "=========================================="

# 步骤0: 等待之前的mv进程完成
echo "[步骤0] 检查是否有正在进行的mv进程..."
while pgrep -f "mv.*stage.*apdcephfs_gy2" > /dev/null 2>&1; do
    echo "  等待mv进程完成..."
    sleep 10
done
echo "  无正在进行的mv进程"

# 步骤1: 确保目标目录存在
echo "[步骤1] 确保目标目录存在..."
mkdir -p "$DST"

# 步骤2: 将源目录中的真实目录（非软链接）移动到目标
echo "[步骤2] 迁移子目录到快速存储..."
cd "$SRC"
for item in */; do
    item="${item%/}"
    if [ -L "$item" ]; then
        echo "  跳过软链接: $item"
        # 如果是指向目标目录的软链接，删除它（后续整个outputs会变成软链接）
        continue
    fi
    if [ -d "$DST/$item" ]; then
        echo "  目标已存在，跳过: $item"
    else
        echo "  移动: $item ($(du -sh "$item" 2>/dev/null | cut -f1))"
        cp -a "$item" "$DST/"
        echo "  完成: $item"
    fi
done

# 步骤3: 重命名原目录为备份
echo "[步骤3] 重命名原outputs为outputs_old..."
cd /apdcephfs/wx_feature/home/cedricnie/Causal-Forcing
mv outputs outputs_old

# 步骤4: 创建软链接
echo "[步骤4] 创建软链接 outputs -> $DST"
ln -s "$DST" outputs

# 步骤5: 验证
echo "[步骤5] 验证..."
echo "  软链接: $(ls -la outputs)"
echo "  内容:"
ls -la outputs/
echo ""
echo "=========================================="
echo "迁移完成！"
echo "原数据备份在: outputs_old/"
echo "确认无误后可删除: rm -rf outputs_old/"
echo "=========================================="

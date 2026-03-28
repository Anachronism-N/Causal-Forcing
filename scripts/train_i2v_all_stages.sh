#!/bin/bash
# =============================================================================
# Stage 1-3 完整 I2V 训练流程
# 
# 前置条件：
#   1. I2V 视频数据已预处理为 LMDB 格式 (clean latent + prompt)
#      存放在 dataset/clean_data_i2v/
#   2. 模型权重已下载到 wan_models/ 目录:
#      - wan_models/Wan2.1-Fun-1.3B-InP/ (1.3B I2V 模型)
#      - wan_models/Wan2.1-I2V-14B-720P/ (14B I2V 教师模型, Stage 3 需要)
#   3. 环境已安装所有依赖 (torch, torchrun, lmdb 等)
#
# 使用方法：
#   bash scripts/train_i2v_all_stages.sh [STAGE] [NUM_GPUS]
#   STAGE: 1, 2ode, 2cd, 2ode_data, 3, all (默认 all)
#   NUM_GPUS: GPU 数量 (默认 8)
# =============================================================================

set -e

STAGE=${1:-all}
NUM_GPUS=${2:-8}

# 输出目录
STAGE1_OUTPUT="outputs/stage1_i2v"
STAGE2_ODE_DATA="dataset/ODE6KCausal_framewise_i2v"
STAGE2_ODE_OUTPUT="outputs/stage2_ode_i2v"
STAGE2_CD_OUTPUT="outputs/stage2_cd_i2v"
STAGE3_OUTPUT="outputs/stage3_dmd_i2v"

# 数据路径
I2V_DATA="/apdcephfs/wx_feature/home/cedricnie/data/lmdb_1219"

echo "============================================"
echo "  Causal-Forcing I2V 训练流程"
echo "  Stage: ${STAGE}"
echo "  GPUs: ${NUM_GPUS}"
echo "============================================"

# ---------- Stage 1: AR Diffusion Training ----------
run_stage1() {
    echo ""
    echo "========== Stage 1: AR Diffusion Training (I2V) =========="
    echo "输入数据: ${I2V_DATA}"
    echo "输出目录: ${STAGE1_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config_path configs/ar_diffusion_tf_framewise_i2v.yaml \
        --logdir ${STAGE1_OUTPUT} \
        --disable-wandb

    echo "Stage 1 完成！checkpoint 保存在 ${STAGE1_OUTPUT}/"
}

# ---------- Stage 2 ODE Data Generation ----------
run_stage2_ode_data() {
    echo ""
    echo "========== Stage 2: ODE Trajectory 数据生成 (I2V) =========="
    
    # 自动查找最新的 Stage 1 checkpoint
    STAGE1_CKPT=$(ls -d ${STAGE1_OUTPUT}/checkpoint_model_*/model.pt 2>/dev/null | sort | tail -1)
    if [ -z "${STAGE1_CKPT}" ]; then
        echo "错误: 未找到 Stage 1 checkpoint，请先运行 Stage 1"
        exit 1
    fi
    echo "使用 Stage 1 checkpoint: ${STAGE1_CKPT}"
    echo "输出目录: ${STAGE2_ODE_DATA}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} get_causal_ode_data_framewise_i2v.py \
        --output_folder ${STAGE2_ODE_DATA} \
        --rawdata_path ${I2V_DATA} \
        --generator_ckpt ${STAGE1_CKPT} \
        --guidance_scale 6.0 \
        --model_name Wan2.1-Fun-1.3B-InP

    echo "ODE 数据生成完成！数据保存在 ${STAGE2_ODE_DATA}/"
}

# ---------- Stage 2 Option A: ODE Distillation ----------
run_stage2_ode() {
    echo ""
    echo "========== Stage 2: ODE Distillation (I2V) =========="
    
    STAGE1_CKPT=$(ls -d ${STAGE1_OUTPUT}/checkpoint_model_*/model.pt 2>/dev/null | sort | tail -1)
    if [ -z "${STAGE1_CKPT}" ]; then
        echo "错误: 未找到 Stage 1 checkpoint，请先运行 Stage 1"
        exit 1
    fi
    echo "使用 generator_ckpt: ${STAGE1_CKPT}"
    echo "ODE 数据: ${STAGE2_ODE_DATA}"
    echo "输出目录: ${STAGE2_ODE_OUTPUT}"
    echo ""

    # 动态覆盖 generator_ckpt 和 data_path
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config_path configs/causal_ode_framewise_i2v.yaml \
        --logdir ${STAGE2_ODE_OUTPUT} \
        --disable-wandb

    echo "Stage 2 ODE 完成！checkpoint 保存在 ${STAGE2_ODE_OUTPUT}/"
}

# ---------- Stage 2 Option B: Consistency Distillation ----------
run_stage2_cd() {
    echo ""
    echo "========== Stage 2: Consistency Distillation (I2V) =========="
    
    STAGE1_CKPT=$(ls -d ${STAGE1_OUTPUT}/checkpoint_model_*/model.pt 2>/dev/null | sort | tail -1)
    if [ -z "${STAGE1_CKPT}" ]; then
        echo "错误: 未找到 Stage 1 checkpoint，请先运行 Stage 1"
        exit 1
    fi
    echo "使用 generator_ckpt: ${STAGE1_CKPT}"
    echo "输出目录: ${STAGE2_CD_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config_path configs/causal_cd_framewise_i2v.yaml \
        --logdir ${STAGE2_CD_OUTPUT} \
        --disable-wandb

    echo "Stage 2 CD 完成！checkpoint 保存在 ${STAGE2_CD_OUTPUT}/"
}

# ---------- Stage 3: DMD Distillation ----------
run_stage3() {
    echo ""
    echo "========== Stage 3: DMD Distillation (I2V) =========="
    
    # 优先查找 ODE checkpoint，否则查找 CD checkpoint
    STAGE2_CKPT=$(ls -d ${STAGE2_ODE_OUTPUT}/checkpoint_model_*/model.pt 2>/dev/null | sort | tail -1)
    if [ -z "${STAGE2_CKPT}" ]; then
        STAGE2_CKPT=$(ls -d ${STAGE2_CD_OUTPUT}/checkpoint_model_*/model.pt 2>/dev/null | sort | tail -1)
    fi
    if [ -z "${STAGE2_CKPT}" ]; then
        echo "错误: 未找到 Stage 2 checkpoint，请先运行 Stage 2"
        exit 1
    fi
    echo "使用 generator_ckpt: ${STAGE2_CKPT}"
    echo "教师模型: Wan2.1-I2V-14B-720P (14B)"
    echo "学生模型: Wan2.1-Fun-1.3B-InP (1.3B)"
    echo "输出目录: ${STAGE3_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config_path configs/causal_forcing_dmd_framewise_i2v_local.yaml \
        --logdir ${STAGE3_OUTPUT} \
        --disable-wandb

    echo "Stage 3 DMD 完成！最终模型保存在 ${STAGE3_OUTPUT}/"
}

# ---------- 执行逻辑 ----------
case ${STAGE} in
    1)
        run_stage1
        ;;
    2ode_data)
        run_stage2_ode_data
        ;;
    2ode)
        run_stage2_ode
        ;;
    2cd)
        run_stage2_cd
        ;;
    3)
        run_stage3
        ;;
    all)
        run_stage1
        run_stage2_ode_data
        run_stage2_ode
        run_stage3
        ;;
    *)
        echo "未知 Stage: ${STAGE}"
        echo "可选值: 1, 2ode_data, 2ode, 2cd, 3, all"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  训练完成！"
echo "============================================"

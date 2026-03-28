#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-configs/causal_forcing_dmd_framewise_i2v_local.yaml}"
LOGDIR="${LOGDIR:-/apdcephfs/wx_feature/home/cedricnie/Causal-Forcing/outputs/causal_forcing_dmd_framewise_i2v}"
WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-/apdcephfs/wx_feature/home/cedricnie/Causal-Forcing/outputs/wandb}"
MASTER_PORT="${MASTER_PORT:-29511}"
LOG_FILE="$LOGDIR/train_$(date +%F_%H-%M-%S).log"

mkdir -p "$LOGDIR" "$WANDB_SAVE_DIR" checkpoints

source /apdcephfs/wx_feature/home/cedricnie/activate_conda.sh longlive

cd "$PROJECT_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "WANDB_SAVE_DIR=$WANDB_SAVE_DIR"
echo "MASTER_PORT=$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LOG_FILE=$LOG_FILE"

if [ ! -e "wan_models" ]; then
    ln -s /apdcephfs/wx_feature/home/cedricnie/LongLive/wan_models ./wan_models
fi

if [ ! -f "checkpoints/framewise/causal_ode.pt" ]; then
    echo "Downloading framewise causal_ode.pt ..."
    huggingface-cli download zhuhz22/Causal-Forcing framewise/causal_ode.pt --local-dir checkpoints
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTHONUNBUFFERED=1

torchrun \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$WANDB_SAVE_DIR" \
  --disable-wandb 2>&1 | tee -a "$LOG_FILE"

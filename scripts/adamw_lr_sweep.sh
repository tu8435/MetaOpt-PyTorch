#!/bin/bash
# MetaOpt LR sweep with AdamW base optimizer
# H = 15 , HH = 10
# Sweeps base_lr ∈ {1e-5 , 3e-5 , 5e-5}   (lr_gpc fixed)

set -euo pipefail

# export HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set by the SLURM wrapper}

MODEL_ID="facebook/opt-125m"
TASK="sst2"
DEVICE="cuda"
STEPS=200
SUBSET=128
EPOCHS=0.25
H=15
HH=10
LR_GPC=1e-4          # keep ≤ 5e-6 for SGD and ≤ 1e-3 for AdamW
MAX_NORM=1.0

BASE_LR_LIST=(1e-4 5e-4 1e-3)

for BASE_LR in "${BASE_LR_LIST[@]}"; do
  echo "─────────────────────────────────────────────────────────────"
  echo "▶ MetaOpt-AdamW  base_lr=${BASE_LR}  lr_gpc=${LR_GPC}"
  echo "─────────────────────────────────────────────────────────────"

  python run.py \
    --hf_token "$HF_TOKEN" \
    --model_id "$MODEL_ID" \
    --task "$TASK" \
    --device "$DEVICE" \
    --steps "$STEPS" \
    --subset "$SUBSET" \
    --num_epochs "$EPOCHS" \
    --H "$H" \
    --HH "$HH" \
    --m_method scalar \
    --base_lr "$BASE_LR" \
    --lr_gpc "$LR_GPC" \
    --weight_decay 0.0 \
    --freeze_gpc_params false \
    --fake_the_dynamics false \
    --base_optimizer_cls AdamW \
    --base_optimizer_kwargs "{\"lr\":${BASE_LR},\"betas\":[0.9,0.99]}" \
    --gpc_optimizer_cls AdamW \
    --gpc_optimizer_kwargs "{\"lr\":${LR_GPC},\"betas\":[0.9,0.99]}" \
    --max_norm "$MAX_NORM" \
    --cache_dir "$HF_HOME" \
    --out_dir "./results/${TASK}_H${H}_LR${BASE_LR}_LG${LR_GPC}"
done
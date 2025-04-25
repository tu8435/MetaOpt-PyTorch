#!/bin/bash
# MetaOpt LR sweep with AdamW base optimizer
# H = 15 , HH = 10
# Sweeps base_lr ∈ {1e-5 , 3e-5 , 5e-5}   (lr_gpc fixed)

set -e

export HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set by the SLURM wrapper}

MODEL_ID="facebook/opt-125m"
TASK="sst2"
DEVICE="cuda"
STEPS=2000
SUBSET=128
EPOCHS=2
H=15
HH=10
LR_GPC=5e-6          # keep low per spec
MAX_NORM=1.0

BASE_LR_LIST=(1e-5 3e-5 5e-5)

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
    --max_norm "$MAX_NORM" \
    --cache_dir "$HF_HOME" \
    --out_dir "./results/{task}_H{H}_LR{base_lr}_LG{lr_gpc}"
done
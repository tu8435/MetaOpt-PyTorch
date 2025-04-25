#!/bin/bash
# ──────────────────────────────────────────────────────────────
# MetaOpt LR sweep with SGD base optimizer
#  H = 15 , HH = 10
#  LR pairs: (base_lr , lr_gpc)
# ──────────────────────────────────────────────────────────────

set -e
export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running.}

MODEL_ID="facebook/opt-125m"
TASK="sst2"
DEVICE="cuda"
STEPS=2000
SUBSET=128
EPOCHS=2
H=15
HH=10

# (base_lr  lr_gpc) pairs
# Already did "1e-3 5e-6"
pairs=(
  "1e-3 1e-7"
  "1e-4 5e-6"
  "1e-3 1e-7"
)

for pair in "${pairs[@]}"; do
  read base_lr lr_gpc <<< "$pair"

  echo "─────────────────────────────────────────────────────────────"
  echo "▶  MetaOpt-SGD  base_lr=${base_lr}  lr_gpc=${lr_gpc}"
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
    --base_lr "$base_lr" \
    --lr_gpc "$lr_gpc" \
    --weight_decay 0.0 \
    --freeze_gpc_params false \
    --fake_the_dynamics false \
    --base_optimizer_cls SGD \
    --base_optimizer_kwargs "{\"lr\":${base_lr}}" \
    --max_norm 1.0 \
    --cache_dir /scratch/gpfs/tu8435/model_cache/ 
done
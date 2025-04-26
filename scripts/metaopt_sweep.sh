#!/bin/bash
# MetaOpt selective run: AdamW and SGD (fixed settings)

set -euo pipefail

MODEL_ID="facebook/opt-125m"
TASK="sst2"
DEVICE="cuda"
STEPS=200
SUBSET=128
EPOCHS=0.25
H=15
HH=10
MAX_NORM=1.0

# 1) AdamW run: base AdamW with default LR (0.001), gpc lr 1e-4
echo "─────────────────────────────────────────────────────────────"
echo "▶ MetaOpt-AdamW  base_lr=default (0.001)  lr_gpc=1e-4"
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
  --weight_decay 0.0 \
  --freeze_gpc_params false \
  --fake_the_dynamics false \
  --base_optimizer_cls AdamW \
  --base_optimizer_kwargs "{\"lr\":0.001,\"betas\":[0.9,0.99]}" \
  --gpc_optimizer_cls AdamW \
  --gpc_optimizer_kwargs "{\"lr\":1e-4,\"betas\":[0.9,0.99]}" \
  --max_norm "$MAX_NORM" \
  --cache_dir "$HF_HOME" \
  --out_dir "./results/${TASK}_H${H}_AdamW_LG1e-4"

# 2) SGD run: base SGD with default LR (0.001), gpc lr 1e-6
echo "─────────────────────────────────────────────────────────────"
echo "▶ MetaOpt-SGD  base_lr=default (0.001)  lr_gpc=1e-6"
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
  --weight_decay 0.0 \
  --freeze_gpc_params false \
  --fake_the_dynamics false \
  --base_optimizer_cls SGD \
  --base_optimizer_kwargs "{\"lr\":0.001,\"momentum\":0.0}" \
  --gpc_optimizer_cls SGD \
  --gpc_optimizer_kwargs "{\"lr\":1e-6,\"momentum\":0.0}" \
  --max_norm "$MAX_NORM" \
  --cache_dir "$HF_HOME" \
  --out_dir "./results/${TASK}_H${H}_SGD_LG1e-6"
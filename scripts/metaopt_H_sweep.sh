#!/bin/bash
# MetaOpt H sweep: SGD-SGD (fixed settings)
set -euo pipefail

MODEL_ID="facebook/opt-125m"
TASK="sst2"
DEVICE="cuda"
STEPS=2000
EPOCHS=2
MAX_NORM=1.0
BASE_LR=1e-3          # base LR for both optimizers
LR_GPC=1e-6           # lr for the meta-optimizer (gpc)
HF_CACHE="$HF_HOME"   # adjust if you keep models elsewhere

###############################################################################
# 3) Sweep H: 5, 10, 15, 20   (fix HH=10, subset=256)
###############################################################################
HH=10
SUBSET=256
for H in 5 10 15 20; do
  echo "─────────────────────────────────────────────────────────────"
  echo "▶ SGD-SGD | H=${H} | HH=${HH} | subset=${SUBSET}"
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
    --lr_gpc "$LR_GPC" \
    --base_lr "$BASE_LR" \
    --freeze_gpc_params false \
    --fake_the_dynamics false \
    --base_optimizer_cls SGD \
    --base_optimizer_kwargs "{\"lr\":${BASE_LR},\"momentum\":0.0}" \
    --gpc_optimizer_cls SGD \
    --gpc_optimizer_kwargs "{\"lr\":${LR_GPC},\"momentum\":0.0}" \
    --max_norm "$MAX_NORM" \
    --cache_dir "$HF_CACHE" \
    --out_dir "./results/SST2_subset${SUBSET}_H${H}_HH${HH}_SGD-SGD"
done
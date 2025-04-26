#!/bin/bash
export PYTHONPATH="/scratch/gpfs/tu8435/COS484/MetaOpt-PyTorch"

python run.py \
  --hf_token $HF_TOKEN \
  --model_id facebook/opt-125m \
  --task sst2 \
  --H 15 \
  --HH 10 \
  --m_method scalar \
  --base_lr 1e-3 \
  --weight_decay 0.0 \
  --freeze_gpc_params false \
  --fake_the_dynamics false \
  --lr_gpc 1e-4 \ # Not tuned, simple an order of magnitude smaller than base_lr to avoid exploding gradients
  --base_optimizer_cls AdamW \
  --base_optimizer_kwargs '{"lr":1e-3,"betas":[0.9,0.99]}' \
  --gpc_optimizer_cls AdamW \
  --gpc_optimizer_kwargs "{\"lr\":${LR_GPC},\"betas\":[0.9,0.99]}"
  --max_norm 1.0 \
  --steps 250 \
  --subset 128 \
  --num_epochs 0.2 \
  --cache_dir /scratch/gpfs/tu8435/model_cache/ \
  --out_dir "./results/{task}_H{H}_LR{base_lr}_LG{lr_gpc}"
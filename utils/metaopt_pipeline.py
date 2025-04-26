"""
An assorment of utility functions for experiments
"""

# ----------------------------------------------------------------
# ----------------------- META-OPT PIPELINE ----------------------
# ----------------------------------------------------------------

# ───────────────────────── imports & helpers ────────────────────
import os
import sys
import gc
import copy
import torch
import evaluate
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    # ---- If you want to do standard hugging face Trainer-based baselines:
    Adafactor
)

from datasets import load_dataset
from huggingface_hub import login


# ---- Optimizers
import torch.optim as optim
from torch.optim import AdamW
from utils.adan import Adan
from utils.metaoptimizer import MetaOpt
from utils.metatrainer import MetaTrainer

# ---- LoRA/PEFT
from peft import LoraConfig, get_peft_model, TaskType

# ---- Utils
from utils.utils import load_task, build_base_model, compute_metrics_fn


# ---- Pipeline (Top --> Down)
# utils/metaopt_pipeline.py  (or wherever the original lives)
# -----------------------------------------------------------------
def metaopt_pipeline(
        task: str,
        steps: int = 2000,
        subset: int = 256,
        num_epochs: int = 3,
        device: str = "cuda",
        base_lr: float = 1e-3,
        lr_gpc: float = 1e-6,
        cache_dir: str = os.getenv("HF_HOME", "./hf_cache"),
        metaopt_overrides: dict = None,
        model_id_or_path: str = None,
        output_dir: str = None,
):
    """
    Run the two-phase Meta-Opt workflow and return eval metrics.

    Parameters
    ----------
    task : {"sst2","squad","copa"} TODO: actually implement for squad or copa
    steps : int
        # of GPC training steps in phase-1.
    subset : int
        Dataset subset size used for phase-1.
    num_epochs : int
        Finetuning epochs in phase-2.
    device : {"cuda","cpu","mps"}
    base_lr : float
        Inner optimizer LR (also fed into lambda t: base_lr).
    lr_gpc : float
        Learning-rate for the GPC parameters.
    cache_dir : str | None
        HF cache dir for offline runs.  Forwarded to tokenizer,
        `load_dataset`, and `from_pretrained`.
    metaopt_overrides : dict | None
        Extra kwargs forwarded **verbatim** into every MetaOpt(...)
        constructor call (phase-1 *and* phase-2).
    model_id_or_path : str | None
        HF model repo or local path. Falls back to global `model_id`.
    """
    if metaopt_overrides is None:
        raise ValueError("must pass in metaopt_overrides dict")
    if model_id_or_path is None:
        raise ValueError("must pass in model_id_or_path")
    if output_dir is None:
        print("WARNING: output_dir not set, using default:\n'../results/{task}_metaopt_lr{base_lr}'")
        output_dir = f"../results/{task}_metaopt_lr{base_lr}"

    # 0) Define output directories
    if output_dir is None:
        raise ValueError("output_dir must be passed")
    phase1_dir = os.path.join(output_dir, "phase_1")
    phase2_dir = os.path.join(output_dir, "phase_2")

    # 1) tokenise / load data for that task
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, cache_dir=cache_dir, local_files_only=True) # Assuming compute node doesn't have access to internet)
    # TODO: Figure out how to get the correct pad_token for the model
    tokenizer.pad_token = tokenizer.eos_token # OPT uses <eos> as pad

    train_ds, val_ds, test_ds, _, _ = load_task(
        task=task, tokenizer=tokenizer,  cache_dir=cache_dir
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2) phase-1: meta-train GPC on deterministic subset
    # NOTE: 
    #   - This optimizer is passed into phase-2
    meta_opt = meta_train_gpc(
        task              = task,
        model_id_or_path  = model_id_or_path,
        train_dataset     = train_ds,
        val_dataset       = val_ds,
        steps             = steps,
        subset_size       = subset,
        device            = device,
        base_lr           = base_lr,
        lr_gpc            = lr_gpc,
        data_collator     = data_collator,
        tokenizer         = tokenizer,
        cache_dir         = cache_dir,
        metaopt_overrides = metaopt_overrides,
        compute_metrics_fn= compute_metrics_fn(task),
        output_dir        = phase1_dir,
    )

    # 3) phase-2: freeze GPC and fine-tune on full dataset
    metrics = run_meta_finetuning(
        task             = task,
        meta_optimizer   = meta_opt,
        train_dataset    = train_ds,
        val_dataset      = val_ds,
        test_dataset     = test_ds,
        output_dir       = phase2_dir,
        num_epochs       = num_epochs,
        device           = device,
        base_lr          = base_lr,
        cache_dir        = cache_dir,
        metaopt_overrides= metaopt_overrides,
        data_collator    = data_collator,
        tokenizer        = tokenizer,
        compute_metrics_fn= compute_metrics_fn(task),
        model_id_or_path = model_id_or_path,
    )
    return metrics

# ---------------------------------------------------------
# META PHASE 1  – train GPC on small deterministic subset
# ---------------------------------------------------------
def meta_train_gpc(
        task:str,
        train_dataset, 
        val_dataset,
        model_id_or_path:str = "facebook/opt-125m",
        steps:int = 2000, subset_size:int = 256,
        device:str = "cuda",
        base_lr: float = 1e-3, # TODO: remove redundancy
        lr_gpc: float = 1e-6,  # TODO: remove redundancy
        cache_dir: str = os.getenv("HF_HOME", "./hf_cache"),
        metaopt_overrides: dict = None,
        data_collator: DataCollatorWithPadding = None,
        tokenizer: AutoTokenizer = None,
        compute_metrics_fn: callable = None,
        output_dir: str = None,
):

    print(f"\n=== Meta-train GPC ({task}) for {steps} steps, base_lr={base_lr}, lr_gpc={lr_gpc} ===")

    # 1) fresh model + task‑specific LoRA
    base_model, lora_tt, _, _ = build_base_model(task=task, 
                                           model_id=model_id_or_path, 
                                           cache_dir=cache_dir,
                                           device=device)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj",
                        "gate_proj","down_proj","up_proj","o_proj"],
        task_type=lora_tt,
    )
    base_model = get_peft_model(base_model, lora_cfg)

    # 2) deterministic subset
    small_ds  = train_dataset.select(range(subset_size))
    batch_sz  = subset_size
    steps_per_epoch    = subset_size // batch_sz
    gradient_acc_steps = subset_size // 4 # 32 x 4 = 128 --> subset size
    """
    Comments about gradient_acc_steps, batch_sz, and subset_size:
    - We need gradient_acc_steps x batch_sz = subset_size so that we 
      can train on the entire subset in one optimization step in the
      deterministic setting.
    - Smaller gradient_acc_steps will lead to faster optimization steps
      but will necessitate a larger batch size. Experimentally, the 
      memory footprint of meta-opt seems to scale linearly with batch
      size so BEWARE of increasing the batch size too much.
    """
    num_epochs         = steps // steps_per_epoch

    args_phase1 = TrainingArguments(
        output_dir          = output_dir,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = gradient_acc_steps,
        learning_rate      = base_lr, # TODO: remove redundancy
        num_train_epochs   = num_epochs,
        eval_strategy      = "steps",
        eval_steps         = 50,
        max_grad_norm      = metaopt_overrides.get("max_norm", 1.0), # <— use passed max_norm
        logging_steps      = 50,
        fp16=False, bf16=False, gradient_checkpointing=False,
        report_to=["tensorboard"],
        lr_scheduler_type='constant',
    )

    meta_opt = MetaOpt(
        model               = base_model,
        H                   = metaopt_overrides.get("H", 15), # <— use passed H
        HH                  = metaopt_overrides.get("HH", 10), # <— use passed HH
        m_method            = metaopt_overrides.get("m_method", "scalar"), # <— use passed m_method
        base_lr             = lambda t: metaopt_overrides.get("base_lr", base_lr),     # <— use passed base_lr
        weight_decay        = metaopt_overrides.get("weight_decay", 0.0), # <— use passed weight_decay
        freeze_gpc_params   = metaopt_overrides.get("freeze_gpc_params", False), # <— use passed freeze_gpc_params
        fake_the_dynamics   = metaopt_overrides.get("fake_the_dynamics", False), # <— use passed fake_the_dynamics
        lr_gpc              = metaopt_overrides.get("lr_gpc", lr_gpc), # <— use passed lr_gpc
        device              = device,
        base_optimizer_cls  = metaopt_overrides.get("base_optimizer_cls", AdamW),
        base_optimizer_kwargs= metaopt_overrides.get("base_optimizer_kwargs", {"lr": 1e-3, "betas": [0.9, 0.99]}), # <— also use base_lr here
        gpc_optimizer_cls   = metaopt_overrides.get("gpc_optimizer_cls", AdamW),
        gpc_optimizer_kwargs= metaopt_overrides.get("gpc_optimizer_kwargs", {"lr": 1e-6, "betas": [0.9, 0.99]}), # <— also use base_lr here
        max_norm            = metaopt_overrides.get("max_norm", 1.0), # <— use passed max_norm
    )

    trainer = MetaTrainer(
        model              = base_model,
        meta_optimizer_params = {},          # not needed; we pass opt directly
        args               = args_phase1,
        train_dataset      = small_ds,
        eval_dataset       = val_dataset,
        data_collator      = data_collator,
        tokenizer          = tokenizer,
        compute_metrics    = compute_metrics_fn,
        optimizers         = (meta_opt, None)
    )

    # TODO: Possibly integrate libraries like higher meant for higher order derivative calculations to prevent errors with gradient checkpointing
    print(f"Sanity Check from trainer optimizer Phase 1: {[(trainer.optimizer.lr_gpc, trainer.optimizer.gpc_optimizer_kwargs, trainer.optimizer.gpc_optimizer_cls,), (trainer.optimizer.base_lr, trainer.optimizer.base_optimizer_kwargs, trainer.optimizer.base_optimizer_cls), (trainer.optimizer.freeze_gpc_params, trainer.optimizer.fake_the_dynamics, trainer.optimizer.H, trainer.optimizer.HH),]}")
    with sdpa_kernel(SDPBackend.MATH):
        trainer.train()

    print("✔ GPC trained - returning MetaOpt (with params on GPU)")
    return meta_opt               # contains trained gpc_params

# ---------------------------------------------------------
# META PHASE 2  – freeze GPC and fine‑tune full dataset
# ---------------------------------------------------------
def run_meta_finetuning(
        task:str,
        meta_optimizer:MetaOpt,
        train_dataset, val_dataset, test_dataset,
        num_epochs:float = 2,
        device: str = "cuda",
        model_id_or_path:str = "facebook/opt-125m",
        base_lr: float = 1e-3,
        cache_dir: str = os.getenv("HF_HOME", "./hf_cache"),
        metaopt_overrides: dict = None,
        data_collator: DataCollatorWithPadding = None,
        tokenizer: AutoTokenizer = None,
        compute_metrics_fn: callable = None,
        output_dir: str = None):

    print(f"\n=== Meta-finetune ({task}) with frozen GPC, base_lr={base_lr} ===")

    # 1) brand‑new model + LoRA identical to phase 1
    base_model, lora_tt, _, _ = build_base_model(task=task, 
                                           model_id=model_id_or_path, 
                                           cache_dir=cache_dir,
                                           device=device)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj",
                        "gate_proj","down_proj","up_proj","o_proj"],
        task_type=lora_tt,
    )
    new_model = get_peft_model(base_model, lora_cfg).to(device)

    # 2) new MetaOpt that re‑uses trained gpc_params but sets freeze=True
    frozen_meta_opt = MetaOpt(
        model              = new_model,
        H                  = metaopt_overrides.get("H", 15), # <— use passed H
        HH                 = metaopt_overrides.get("HH", 10), # <— use passed HH
        m_method           = metaopt_overrides.get("m_method", "scalar"), # <— use passed m_method
        base_lr            = lambda t: metaopt_overrides.get("base_lr", base_lr),     # <— use passed base_lr
        weight_decay       = metaopt_overrides.get("weight_decay", 0.0), # <— use passed weight_decay
        freeze_gpc_params  = True, # <— freeze GPC params here ALWAYS
        fake_the_dynamics  = metaopt_overrides.get("fake_the_dynamics", False), # <— use passed fake_the_dynamics
        lr_gpc             = 0.0,                  # ignored when frozen
        device             = device,
        base_optimizer_cls = metaopt_overrides.get("base_optimizer_cls", AdamW),
        base_optimizer_kwargs= metaopt_overrides.get("base_optimizer_kwargs", {"lr": 1e-3, "betas": [0.9, 0.99]}), # <— also use base_lr here
        gpc_optimizer_cls  = metaopt_overrides.get("gpc_optimizer_cls", AdamW), # ignored when frozen
        gpc_optimizer_kwargs= metaopt_overrides.get("gpc_optimizer_kwargs", {"lr": 1e-6, "betas": [0.9, 0.99]}), # <— irrelevant, optimizer is frozen
        max_norm           = metaopt_overrides.get("max_norm", 1.0), # <— use passed max_norm
    )
    # copy GPC weights
    with torch.no_grad():
        frozen_meta_opt.gpc_params.copy_(meta_optimizer.gpc_params.data)

    batch_size = 4 * (1 + 3 * (task == "squad")) # 4x speedup for squad (larger dataset)

    args_phase2 = TrainingArguments(
        output_dir          = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = 4,
        learning_rate      = base_lr,
        num_train_epochs   = num_epochs,
        eval_strategy      = "steps",
        eval_steps         = 50,
        logging_steps      = 50,
        max_grad_norm      = metaopt_overrides.get("max_norm", 1.0), # <— use passed max_norm
        fp16=False, bf16=False, gradient_checkpointing=False,
        save_steps         = 1000,
        report_to=["tensorboard"],
        lr_scheduler_type='constant',
    )

    trainer = MetaTrainer(
        model            = new_model,
        meta_optimizer_params = {},          # not needed; we pass opt directly
        args             = args_phase2,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        data_collator    = data_collator,
        tokenizer        = tokenizer,
        compute_metrics  = compute_metrics_fn,
        optimizers       = (frozen_meta_opt, None) # <— frozen optimizer
    )

    print(f"Sanity Check from trainer optimizer Phase 2: {[(trainer.optimizer.lr_gpc, trainer.optimizer.gpc_optimizer_kwargs, trainer.optimizer.gpc_optimizer_cls,), (trainer.optimizer.base_lr, trainer.optimizer.base_optimizer_kwargs, trainer.optimizer.base_optimizer_cls), (trainer.optimizer.freeze_gpc_params, trainer.optimizer.fake_the_dynamics, trainer.optimizer.H, trainer.optimizer.HH),]}")
    trainer.train()
    metrics = trainer.evaluate(val_dataset)
    print(f"[Meta-Frozen GPC]  {metrics}")
    return metrics











# ----------------------------------------------------------------
# ----------------------------------------------------------------
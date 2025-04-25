"""
A collection of utility functions for experiment pipelines.

These functions are designed to work with the Hugging Face Transformers
library and the Datasets library.
"""

# ──────────────────────────── imports ─────────────────────────────
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

# ---- LoRA/PEFT
from peft import LoraConfig, get_peft_model, TaskType

# ---- Evaluation Metrics
import evaluate
ACCURACY_METRIC = evaluate.load("accuracy")


def compute_metrics_fn(task:str):
    """
    Returns a callable compatible with HF Trainer.compute_metrics
    """
    if task in {"sst2"}:                # simple accuracy
        def _acc(pred):
            logits, labels = pred
            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits)
            preds = logits.argmax(-1)
            return ACCURACY_METRIC.compute(predictions=preds,
                                           references=labels)
        return _acc
    else:
        raise ValueError(task, f"{task} not supported for compute_metrics_fn")

# ---- Universal Dataset Loader
def load_task(task:str, 
              tokenizer, 
              max_len=128, 
              cache_dir=os.getenv("HF_HOME", "./hf_cache")
              ):
    """
    Returns train/val/test HF Datasets already tokenised & formatted
    task ∈ {"sst2","squad","copa"}
    """
    if task=="sst2":
        ds      = load_dataset("glue","sst2", cache_dir=cache_dir) #FIX  
        num_lab = 2
        def tok(ex): return tokenizer(ex["sentence"],
                    truncation=True,padding="max_length",max_length=max_len)
        ds = ds.map(tok, batched=True)
        cols=["label"]
    else:
      raise ValueError(task)
    
    # keep what Trainer / metric needs
    keep_cols = set(cols) | {"label", "labels", "answers", "id"}   
    ds = ds.remove_columns([c for c in ds["train"].column_names
                            if c not in keep_cols
                            and c not in tokenizer.model_input_names])

    # Check if 'test' split exists, if not, use 'validation'
    # Use validation as fallback for test
    test_ds = ds.get("test", ds["validation"]) 

    return ds["train"], ds["validation"], test_ds, num_lab, list(keep_cols)

# ---- Universal Model Loader
def build_base_model(task: str, 
                     model_id: str = "facebook/opt-125m", 
                     tokenizer: AutoTokenizer=None,
                     data_collator: DataCollatorWithPadding=None,
                     cache_dir: str = os.getenv("HF_HOME", "./hf_cache"), 
                     device: str ="cuda"):
    """
    Returns (model, lora_task_type, data_collator, num_labels)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        # TODO: Figure out how to get the correct pad_token for the model
        tokenizer.pad_token = tokenizer.eos_token # OPT uses <eos> as pad

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # TODO: Support for other tasks
    if task == "squad":
        ModelCls  = AutoModelForQuestionAnswering
        lora_type = TaskType.QUESTION_ANS
        num_labels = None # not needed for Suqad
    elif task == "copa":
        ModelCls  = AutoModelForSequenceClassification # decoder only models dont support MC, treat as SEQ class
        lora_type = TaskType.SEQ_CLS
        num_labels = 2
    else: # "sst2"
        ModelCls  = AutoModelForSequenceClassification
        lora_type = TaskType.SEQ_CLS
        num_labels = 2

    if num_labels is None:
        model = ModelCls.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True) 
    else:
      model = ModelCls.from_pretrained(model_id,
                                      num_labels=num_labels,
                                      cache_dir=cache_dir,
                                      local_files_only=True)
    model.config.use_cache     = False # save as much memory as possible, can be set to True if we have an abundance of memory
    model.config.pad_token_id  = tokenizer.pad_token_id
    model.to(device)

    return model, lora_type, data_collator, num_labels
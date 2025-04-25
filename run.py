"""
This is the primary entry point for the MetaOpt-PyTorch package experiments.
It's implementation is meant to be heavily modular such that each 
instance of run.py performs a one-off experiment. Feel free to
create bash scripts to run multiple experiments in parallel.

As it stands, run.py runs a fine-tuning job of the SST2 dataset
on a pretrained model and then evaluates its performance.

TODO:
- Augment to retrieve model/tokenizer/dataset from cache
- Add support for other datasets
- Add support for other models
- Add support for other optimizers
- Add support for other tasks
- Add support for other metrics

Example:
--------
python run.py \
    python run.py \
        --hf_token  $HF_TOKEN \
        --model_id  facebook/opt-125m \
        --task      sst2 \
        --H         15 \
        --HH        10 \
        --m_method  scalar \
        --base_lr   1e-3 \
        --weight_decay 0.0 \
        --freeze_gpc_params false \
        --fake_the_dynamics false \
        --lr_gpc    5e-6 \
        --base_optimizer_cls Adam \
        --base_optimizer_kwargs '{"lr":1e-3,"betas":[0.9,0.99]}' \
        --max_norm  1.0 \
"""

# ───────────────────────── imports & helpers ───────────────────────
import argparse, os, json, time, warnings
from typing import Any, Dict

def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "y", "t"}

def json_or_none(v: str) -> Dict[str, Any]:
    return {} if v == "" else json.loads(v)

# ─────────────────────────── CLI spec ──────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Meta-Opt single-run CLI")

    # authentication / pipeline infra
    p.add_argument("--hf_token",   required=True, help="HuggingFace token")
    p.add_argument("--model_id",   default="facebook/opt-125m")
    p.add_argument("--task",       choices=["sst2","squad","copa"], default="sst2")

    # phase-1 / phase-2 high-level knobs
    p.add_argument("--steps",      type=int,   default=2000, help="GPC train steps")
    p.add_argument("--subset",     type=int,   default=128,  help="subset size for phase-1")
    p.add_argument("--num_epochs", type=float, default=2,    help="finetune epochs")
    p.add_argument("--device",     choices=["cuda","cpu","mps"], default="cuda")

    # ─────────────────── MetaOpt Parameters ────────────────────
    p.add_argument("--H",                        type=int,   default=15)
    p.add_argument("--HH",                       type=int,   default=10)
    p.add_argument("--m_method",                 default="scalar")
    p.add_argument("--base_lr",                  type=float, default=1e-3,
                   help="base_lr(t) will be a constant lambda returning this value")
    p.add_argument("--weight_decay",             type=float, default=0.0)
    p.add_argument("--freeze_gpc_params",        type=str2bool, default=False)
    p.add_argument("--fake_the_dynamics",        type=str2bool, default=False)
    p.add_argument("--lr_gpc",                   type=float, default=5e-6)
    p.add_argument("--base_optimizer_cls",
                   choices=["Adam","SGD","AdamW","RMSprop","Adan","Adafactor"],
                   default="AdamW")
    p.add_argument("--base_optimizer_kwargs",
                   type=json_or_none,
                   default='{"lr":1e-3,"betas":[0.9,0.99]}',
                   help='JSON string, e.g. \'{"lr":1e-3,"betas":[0.9,0.99]}\'')
    p.add_argument("--gpc_optimizer_cls",
                   choices=["Adam","SGD","AdamW","RMSprop","Adan","Adafactor"],
                   default="AdamW")
    p.add_argument("--gpc_optimizer_kwargs",
                   type=json_or_none,
                   default='{"lr":1e-3,"betas":[0.9,0.99]}',
                   help='JSON string, e.g. \'{"lr":1e-3,"betas":[0.9,0.99]}\'')

    p.add_argument("--max_norm",                 type=float, default=1.0)

    # Cache directory and output directory
    p.add_argument("--cache_dir",  default=os.getenv("HF_HOME", "./hf_cache"),
    help="Path to pre-downloaded HF cache (models + datasets)")

    p.add_argument("--out_dir",    default="./results")
    return p.parse_args()

# ──────────────────────────── main ─────────────────────────────────
def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    # import corresponding pipeline for experiment
    from utils.metaopt_pipeline import metaopt_pipeline

    # login once for private model access
    from huggingface_hub import login
    login(args.hf_token)

    # map string name -> actual torch optimizer class
    from torch import optim
    from utils.adan import Adan
    from transformers import Adafactor
    OPT_LOOKUP = {
        "Adam":      optim.Adam,
        "AdamW":     optim.AdamW,
        "SGD":       optim.SGD,
        "RMSprop":   optim.RMSprop,
        "Adan":      Adan,
    }
    base_opt_cls = OPT_LOOKUP[args.base_optimizer_cls]
    gpc_opt_cls  = OPT_LOOKUP[args.gpc_optimizer_cls]

    # Format output directory
    run_name = f"MetaOpt{args.base_optimizer_cls}_{args.H}_HH{args.HH}_LR{args.base_lr}_LG{args.lr_gpc}_GPCSteps{args.steps}_"
    args.out_dir = os.path.join("results", f"{args.task}", run_name)

    # ----------------------------------------------------------------
    # call pipeline with  MetaOpt params
    # ----------------------------------------------------------------
    metrics = metaopt_pipeline(
        task        = args.task,
        steps       = args.steps,
        subset      = args.subset,
        num_epochs  = args.num_epochs,
        device      = args.device,
        base_lr     = args.base_lr,
        lr_gpc      = args.lr_gpc,
        cache_dir   = args.cache_dir,
        model_id_or_path= args.model_id,
        output_dir = args.out_dir,
        
        # below are *forwarded* into the MetaOpt constructor
        metaopt_overrides = dict(
            H                   = args.H,
            HH                  = args.HH,
            m_method            = args.m_method,
            base_lr             = lambda t: args.base_lr,
            weight_decay        = args.weight_decay,
            freeze_gpc_params   = args.freeze_gpc_params,
            fake_the_dynamics   = args.fake_the_dynamics,
            lr_gpc              = args.lr_gpc,
            device              = args.device,
            base_optimizer_cls  = base_opt_cls,
            base_optimizer_kwargs = args.base_optimizer_kwargs,
            gpc_optimizer_cls   = gpc_opt_cls,
            gpc_optimizer_kwargs = args.gpc_optimizer_kwargs,
            max_norm            = args.max_norm,
        )
    )

    # ---------------- store results ---------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{run_name}_{int(time.time())}.json")

    import json
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== FINAL METRICS ===")
    for k,v in metrics.items(): print(f"{k:20}: {v:.4f}")
    print(f"\nSaved → {out_file}")

# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
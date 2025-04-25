import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFace models and tokenizers to a cache dir")
    parser.add_argument("--hf_token", required=True, help="Your HuggingFace token (e.g. 'hf_abc123')")
    parser.add_argument("--cache_dir", default="./hf_cache", help="Where to save models/tokenizers")
    return parser.parse_args()

def main():
    args = parse_args()

    model_names = [
        "facebook/opt-125m",
        # Add more here if needed
    ]

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Downloading models and tokenizer to: {args.cache_dir}")

    for model_name in model_names:
        folder = model_name.replace("/", "--")
        save_path = os.path.join(args.cache_dir, folder)
        os.makedirs(save_path, exist_ok=True)

        try:
            print(f"Downloading model: {model_name}")
            if "roberta" in model_name.lower():
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=args.cache_dir,
                    token=args.hf_token,
                    force_download=True
                )
            elif "t5" in model_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=args.cache_dir,
                    token=args.hf_token,
                    force_download=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=args.cache_dir,
                    token=args.hf_token,
                    force_download=True
                )
            model.save_pretrained(save_path, from_pt=True)
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")

        try:
            print(f"Downloading tokenizer for: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=args.cache_dir,
                token=args.hf_token
            )
            tokenizer.save_pretrained(save_path, from_pt=True)
        except Exception as e:
            print(f"Error downloading tokenizer {model_name}: {e}")

    print("All downloads complete.")

if __name__ == "__main__":
    main()
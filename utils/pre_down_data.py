import os
import argparse
import datasets

def pre_download_datasets(cache_dir, hf_token=None, datasets_to_download=None):
    os.makedirs(cache_dir, exist_ok=True)

    if datasets_to_download is None:
        datasets_to_download = ['xsum', 'squad', 'pubmed', 'english', 'german', 'writing']

    key_map = {
        "xsum": "document",
        "squad": "context"
    }

    for ds in datasets_to_download:
        print(f"Pre-downloading dataset: {ds}")
        try:
                dset = datasets.load_dataset(
                    ds,
                    split="train",
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    token=hf_token if hf_token else None
                )
                key = key_map.get(ds)
                data = dset[key] if key else dset
                print(f"HuggingFace dataset '{ds}' pre-downloaded. Size: {len(data)}")
        except Exception as e:
            print(f"Error pre-downloading '{ds}': {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-download HuggingFace or custom datasets to cache")
    parser.add_argument("--cache_dir", default="./hf_cache", help="Where to store cached datasets")
    parser.add_argument("--hf_token", default=None, help="Optional HuggingFace token")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to download (e.g. sst2 squad)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pre_download_datasets(args.cache_dir, hf_token=args.hf_token, datasets_to_download=args.datasets)

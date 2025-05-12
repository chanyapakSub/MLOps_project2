import os
import json
import argparse

def get_dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to model folder")
    parser.add_argument("--metrics_file", default="log/metrics_eval_before_compress.json")
    args = parser.parse_args()

    size_mb = get_dir_size_mb(args.model_dir)
    print(f"Model size in {args.model_dir}: {size_mb} MB")

    if os.path.exists(args.metrics_file):
        with open(args.metrics_file) as f:
            data = json.load(f)
    else:
        data = {}

    data.setdefault("public", {})["model_size_mb"] = size_mb

    with open(args.metrics_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {args.metrics_file} with model size.")

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

def extract_selected_samples(selected_csv, json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read all target new_ids
    df = pd.read_csv(selected_csv, dtype=str)
    selected_ids = set(df["new_id"])
    print(f"[✓] Total samples to extract: {len(selected_ids)}")

    # 2. Traverse all chunk_*.json files and match by new_id
    collected = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json") or not fname.startswith("chunk_"):
            continue

        path = os.path.join(json_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)

        for item in data:
            if item.get("new_id") in selected_ids:
                collected.append(item)

    print(f"[✔] Actual number of samples extracted: {len(collected)}")

    # 3. Save two versions
    with open(os.path.join(output_dir, "selected_data_with_newid.json"), "w") as f1:
        json.dump(collected, f1, indent=2)
    print(f"[💾] Saved with new_id: selected_data_with_newid.json")

    for item in collected:
        item.pop("new_id", None)

    with open(os.path.join(output_dir, "selected_data_no_newid.json"), "w") as f2:
        json.dump(collected, f2, indent=2)
    print(f"[💾] Saved without new_id: selected_data_no_newid.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map selected samples (by new_id) back to original JSON data")
    parser.add_argument("--selected_csv", required=True, help="CSV file with selected sample list containing new_id")
    parser.add_argument("--json_dir", required=True, help="Directory containing original chunk_*.json files (with new_id)")
    parser.add_argument("--output_dir", required=True, help="Output directory for new JSON files")
    args = parser.parse_args()

    extract_selected_samples(args.selected_csv, args.json_dir, args.output_dir)

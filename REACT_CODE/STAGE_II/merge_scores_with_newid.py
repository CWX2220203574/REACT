import os
import json
import torch
import argparse
import pandas as pd

def parse_chunk_range(file_name):
    """
    Extract (start_chunk, end_chunk) from a file name like 'chunk_1_20_cosine.pt'.
    """
    base = os.path.splitext(file_name)[0]
    parts = base.split('_')
    return int(parts[1]), int(parts[2])

def load_new_ids(json_dir, start, end):
    """
    Load all 'new_id' values from chunk_{start}.json to chunk_{end}.json.
    """
    new_ids = []
    for i in range(start, end + 1):
        file_path = os.path.join(json_dir, f"chunk_{i}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for sample in data:
                if "new_id" not in sample:
                    raise ValueError(f"Missing 'new_id' field: {file_path}")
                new_ids.append(sample["new_id"])
    return new_ids

def merge_scores_to_csv(pt_dir, json_dir, output_csv):
    """
    Merge cosine similarity scores (.pt) with corresponding 'new_id' from JSON,
    and output to a CSV file.
    """
    rows = []

    for file in sorted(os.listdir(pt_dir)):
        if not file.endswith(".pt") or "_cosine.pt" not in file:
            continue

        pt_path = os.path.join(pt_dir, file)
        start_idx, end_idx = parse_chunk_range(file)

        print(f"[→] Processing: {file}  Corresponding JSON: chunk_{start_idx} to chunk_{end_idx}")

        new_ids = load_new_ids(json_dir, start_idx, end_idx)
        scores = torch.load(pt_path).squeeze().tolist()

        if len(new_ids) != len(scores):
            raise ValueError(f"{file}: Number of new_ids {len(new_ids)} does not match number of scores {len(scores)}")

        rows.extend(zip(new_ids, scores))

    df = pd.DataFrame(rows, columns=["new_id", "score"])
    df.to_csv(output_csv, index=False)
    print(f"[✔] CSV saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align .pt scores with 'new_id' fields from JSON and output a CSV")
    parser.add_argument("--pt_dir", required=True, help="Directory containing .pt score files")
    parser.add_argument("--json_dir", required=True, help="Directory containing JSON files with 'new_id' field")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file")

    args = parser.parse_args()
    merge_scores_to_csv(args.pt_dir, args.json_dir, args.output_csv)

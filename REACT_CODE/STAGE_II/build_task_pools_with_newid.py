import os
import argparse
import pandas as pd

def build_task_pools(root_dir, intersection_csv, top_percent):
    """
    Build task-specific data pools by removing intersection samples based on new_id.
    
    Parameters:
    - root_dir: directory containing topX_percent.csv files per task
    - intersection_csv: path to intersection_keys.csv containing global new_ids
    - top_percent: float value specifying the top percentage threshold
    """
    percent_int = int(top_percent * 100)
    input_file = f"top{percent_int}_percent.csv"
    output_file = f"task_pool_{percent_int}.csv"

    # Load intersection new_id set
    df_inter = pd.read_csv(intersection_csv, dtype=str)
    if "new_id" not in df_inter.columns:
        raise ValueError("❌ 'new_id' column missing in intersection_keys.csv")
    inter_ids = set(df_inter["new_id"])

    for task in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, task)
        top_path = os.path.join(task_dir, input_file)
        output_path = os.path.join(task_dir, output_file)

        if not os.path.isdir(task_dir) or not os.path.exists(top_path):
            print(f"[✘] Skipped: {top_path} not found")
            continue

        df = pd.read_csv(top_path, dtype=str)
        if "new_id" not in df.columns:
            print(f"[✘] Skipped: 'new_id' column missing in {top_path}")
            continue

        original_count = len(df)
        df = df.drop_duplicates(subset="new_id")
        df_filtered = df[~df["new_id"].isin(inter_ids)]

        df_filtered.to_csv(output_path, index=False)
        print(f"[✔] {task}: original {original_count} → after removing intersection: {len(df_filtered)} → saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build task-specific pools by removing global intersection (based on new_id)")
    parser.add_argument("--root_dir", required=True, help="Root directory containing topX_percent.csv files for each task")
    parser.add_argument("--intersection_csv", required=True, help="Path to intersection_keys.csv listing intersected new_ids")
    parser.add_argument("--top_percent", type=float, default=0.2, help="Top percentage to use (default: 0.2)")

    args = parser.parse_args()

    build_task_pools(args.root_dir, args.intersection_csv, args.top_percent)

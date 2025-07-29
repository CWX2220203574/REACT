import os
import argparse
import pandas as pd

def compute_intersection_keys(root_dir, top_percent):
    """
    Compute the intersection of 'new_id's across multiple tasks' top X% files.

    Parameters:
    - root_dir: directory containing subfolders for each task
    - top_percent: percentage (e.g., 0.2 for top 20%) used to find corresponding CSVs

    Returns:
    - List of new_ids that are common to all tasks
    """
    percent_int = int(top_percent * 100)
    file_name = f"top{percent_int}_percent.csv"

    dfs = []
    for subdir in sorted(os.listdir(root_dir)):
        csv_path = os.path.join(root_dir, subdir, file_name)
        if os.path.isdir(os.path.join(root_dir, subdir)) and os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype=str)
            if "new_id" not in df.columns:
                print(f"[⚠] Skipped: {csv_path} is missing 'new_id' column")
                continue
            df = df.drop_duplicates(subset="new_id")
            dfs.append(df)
            print(f"[✔] Loaded: {csv_path} ({len(df)} unique new_id entries)")
        else:
            print(f"[✘] Skipped: {csv_path} not found")

    if len(dfs) < 2:
        raise ValueError("❌ At least two tasks are required to compute intersection")

    intersect_ids = set(dfs[0]["new_id"])
    for df in dfs[1:]:
        intersect_ids &= set(df["new_id"])

    print(f"[→] Found {len(intersect_ids)} samples in the intersection of all tasks' top {percent_int}%")
    return sorted(intersect_ids)

def save_intersection_ids(intersect_ids, out_path):
    """
    Save the intersection new_ids into a CSV file.

    Parameters:
    - intersect_ids: list of new_ids
    - out_path: path to save CSV file
    """
    df = pd.DataFrame({"new_id": intersect_ids})
    df.to_csv(out_path, index=False)
    print(f"[💾] Saved {len(df)} intersected new_ids → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute intersection of samples (based on new_id) across tasks")
    parser.add_argument("--root_dir", required=True, help="Directory containing topX_percent.csv for each task")
    parser.add_argument("--out_keys", required=True, help="Path to save output CSV containing intersected new_ids")
    parser.add_argument("--top_percent", type=float, default=0.2, help="Top percentage to use (default: 0.2)")

    args = parser.parse_args()

    intersect_ids = compute_intersection_keys(args.root_dir, args.top_percent)
    save_intersection_ids(intersect_ids, args.out_keys)

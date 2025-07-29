import os
import argparse
import pandas as pd
from math import floor
import argparse
import os

def select_generalist_samples(weight_csv, subset_ratio, intersection_csv, task_root, output_dir, use_weights):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load task weights
    if use_weights:
        weights_df = pd.read_csv(weight_csv)
        task_weights = dict(zip(weights_df["task_name"], weights_df["weight"]))
    else:
        task_weights = {task: 0.1 for task in os.listdir(task_root)}  # If not using weights, assign default 0.1

    # 2. Get global sample size
    total_samples = None
    for task in os.listdir(task_root):
        all_scores_path = os.path.join(task_root, task, "all_scores.csv")
        if os.path.exists(all_scores_path):
            total_samples = len(pd.read_csv(all_scores_path))
            break
    if total_samples is None:
        raise ValueError("No all_scores.csv found to determine global sample size.")

    # 3. Load intersection samples
    inter_df = pd.read_csv(intersection_csv, dtype=str)
    inter_df["source_task"] = "intersection"
    inter_df["score"] = 1.0  # Assign maximum score to intersection samples
    inter_ids = set(inter_df["new_id"])
    inter_count = len(inter_ids)

    print(f"[✓] Total samples: {total_samples}, Subset ratio: {subset_ratio}, Required intersection samples: {inter_count}")

    target_total = round(total_samples * subset_ratio)
    remaining_quota = target_total - inter_count
    print(f"[✓] Final generalist target size: {target_total}, Remaining quota: {remaining_quota}")

    # 4. Precise quota allocation: float quota + residual compensation
    quota_info = []
    for task in sorted(task_weights):
        raw = remaining_quota * task_weights[task]
        q = floor(raw)
        quota_info.append((task, q, raw - q))

    quota_info.sort(key=lambda x: x[2], reverse=True)
    total_allocated = sum([q[1] for q in quota_info])
    leftover = remaining_quota - total_allocated
    for i in range(leftover):
        task, q, frac = quota_info[i]
        quota_info[i] = (task, q + 1, frac)

    task_quota = {task: q for task, q, _ in quota_info}

    # 5. Multi-round round-robin selection
    percent_int = int(subset_ratio * 100)
    output_file = f"task_pool_{percent_int}.csv"

    task_pools = {}
    for task in sorted(task_quota):
        task_dir = os.path.join(task_root, task)
        pool_path = os.path.join(task_dir, output_file)
        if not os.path.exists(pool_path):
            print(f"[✘] Skipped: {pool_path} does not exist")
            continue

        df = pd.read_csv(pool_path, dtype=str)
        if "score" not in df.columns or "new_id" not in df.columns:
            print(f"[✘] Skipped: {pool_path} missing new_id or score column")
            continue

        df["score"] = df["score"].astype(float)
        df = df.sort_values(by="score", ascending=False).drop_duplicates(subset="new_id")
        task_pools[task] = df.reset_index(drop=True)

    selected_ids = set(inter_ids)  # Selected samples (initially intersection)
    task_selected_rows = {task: [] for task in task_quota}  # Selected samples per task
    task_remaining = task_quota.copy()  # Remaining quota

    # Multi-round allocation
    print(f"[🔁] Starting multi-round allocation...")
    round_count = 0
    while sum(task_remaining.values()) > 0:
        round_count += 1
        for task in sorted(task_pools):
            if task_remaining[task] == 0:
                continue
            pool_df = task_pools[task]
            while not pool_df.empty:
                top_row = pool_df.iloc[0]
                nid = top_row["new_id"]
                pool_df = pool_df.iloc[1:]  # Drop current row
                if nid in selected_ids:
                    continue
                # Select this sample
                selected_ids.add(nid)
                top_row["source_task"] = task
                task_selected_rows[task].append(top_row[["new_id", "score", "source_task"]])
                task_remaining[task] -= 1
                # Remove this sample from all task pools
                for t in task_pools:
                    task_pools[t] = task_pools[t][task_pools[t]["new_id"] != nid].reset_index(drop=True)
                break  # Move to next task
        print(f"[🔄] Round {round_count} completed. Remaining quota: {sum(task_remaining.values())}")

    # 6. Merge final generalist set
    inter_final = inter_df[["new_id", "score", "source_task"]]
    selected_rows = [pd.DataFrame(rows) for rows in task_selected_rows.values()]
    final_df = pd.concat([inter_final] + selected_rows, ignore_index=True)
    print(f"[✅] Final generalist set size: {len(final_df)} / Target size: {target_total}")

    final_path = os.path.join(output_dir, "selected_data.csv")
    final_df.to_csv(final_path, index=False)
    print(f"[💾] Saved to: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a generalist subset by sampling from tasks using weights")
    parser.add_argument("--weight_csv", required=True, help="CSV file containing task weights")
    parser.add_argument("--subset_ratio", type=float, required=True, help="Proportion of generalist subset (e.g., 0.2)")
    parser.add_argument("--intersection_csv", required=True, help="CSV file of intersection samples (must contain new_id column)")
    parser.add_argument("--task_root", required=True, help="Root directory containing all task folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for selected subset")
    parser.add_argument("--use_weights", type=int, choices=[0, 1], default=1, help="Whether to use task weights: 1 (yes) or 0 (no)")

    args = parser.parse_args()

    use_weights = bool(args.use_weights)

    select_generalist_samples(
        args.weight_csv,
        args.subset_ratio,
        args.intersection_csv,
        args.task_root,
        args.output_dir,
        args.use_weights
    )

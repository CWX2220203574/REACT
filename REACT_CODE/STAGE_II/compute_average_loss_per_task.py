import os
import argparse
import pandas as pd

def compute_avg_loss(root_dir, output_csv):
    """
    Compute the average loss for each task folder under the root directory
    and save results as a CSV file.
    """
    rows = []

    for subdir in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, subdir)
        loss_path = os.path.join(task_dir, "losses.csv")
        if not os.path.isdir(task_dir) or not os.path.exists(loss_path):
            continue

        try:
            df = pd.read_csv(loss_path)
            avg_loss = df["loss"].astype(float).mean()

            # Extract task name, e.g., dev_gqa_conversations_60 → gqa
            task_name = subdir.replace("dev_", "").replace("_conversations_60", "")
            rows.append((task_name, avg_loss))
            print(f"[✔] {task_name}: average loss = {avg_loss:.4f}")
        except Exception as e:
            print(f"[✘] Failed to read {loss_path}: {e}")

    df_out = pd.DataFrame(rows, columns=["task_name", "avg_loss"])
    df_out.to_csv(output_csv, index=False)
    print(f"[💾] Saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average loss for each task")
    parser.add_argument("--root_dir", required=True, help="Root path containing task subdirectories")
    parser.add_argument("--output_csv", required=True, help="Path to output average loss CSV file")
    args = parser.parse_args()
    compute_avg_loss(args.root_dir, args.output_csv)

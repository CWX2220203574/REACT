import os
import argparse
import subprocess

def run_batch_merge(pt_root_dir, json_dir):
    """
    Batch run merge_scores_with_newid.py for each task folder under the root directory.
    """
    for task in sorted(os.listdir(pt_root_dir)):
        task_dir = os.path.join(pt_root_dir, task)
        if not os.path.isdir(task_dir):
            continue

        output_csv = os.path.join(task_dir, "all_scores.csv")
        print(f"[→] Processing task: {task}")

        cmd = [
            "python", "merge_scores_with_newid.py",
            "--pt_dir", task_dir,
            "--json_dir", json_dir,
            "--output_csv", output_csv
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run merge_scores_with_newid.py")
    parser.add_argument("--pt_root_dir", required=True, help="Root directory containing pt subfolders for each task")
    parser.add_argument("--json_dir", required=True, help="Directory containing JSON files with new_id fields")

    args = parser.parse_args()
    run_batch_merge(args.pt_root_dir, args.json_dir)

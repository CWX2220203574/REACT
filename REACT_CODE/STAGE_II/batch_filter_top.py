import os
import argparse
import subprocess

def run_batch_filter(score_root_dir, top_percent=0.2):
    percent_int = int(top_percent * 100)  # Convert 0.2 to 20 for naming
    for task in sorted(os.listdir(score_root_dir)):
        task_dir = os.path.join(score_root_dir, task)
        if not os.path.isdir(task_dir):
            continue

        input_csv = os.path.join(task_dir, "all_scores.csv")
        # output_csv = os.path.join(task_dir, "top20_percent.csv")
        output_csv = os.path.join(task_dir, f"top{percent_int}_percent.csv")  # Dynamic output name

        if not os.path.exists(input_csv):
            print(f"[Skipped] File not found: {input_csv}")
            continue

        print(f"[→] Filtering task: {task} top {int(top_percent * 100)}%")

        cmd = [
            "python", "filter_top_percent.py",
            "--input_csv", input_csv,
            "--output_csv", output_csv,
            "--top_percent", str(top_percent)
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch filter top N% scoring samples for each task")
    parser.add_argument("--score_root_dir", required=True, help="Root directory containing all_scores.csv for multiple tasks")
    parser.add_argument("--top_percent", type=float, default=0.2, help="Top percentage to select, default is 0.2")

    args = parser.parse_args()
    run_batch_filter(args.score_root_dir, args.top_percent)

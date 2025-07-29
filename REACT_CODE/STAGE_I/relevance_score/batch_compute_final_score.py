import os
import argparse
import subprocess

def extract_task_name(sim_dir_name):
    return sim_dir_name.replace("output_cosine_scores_mean_", "")

def run_all(similarity_root, weight_root, output_root):
    """
    Iterate over all similarity directories, match corresponding weight files,
    and execute compute_final_score.py for each task.
    """
    for subdir in os.listdir(similarity_root):
        sim_dir = os.path.join(similarity_root, subdir)
        if not os.path.isdir(sim_dir):
            continue

        task = extract_task_name(subdir)

        # Construct weight file path
        weight_csv = os.path.join(weight_root, f"dev_{task}_conversations", "weighted_loss_linear.csv")
        if not os.path.isfile(weight_csv):
            print(f"[Skipped] Weight file not found: {weight_csv}")
            continue

        # Construct output directory
        out_dir = os.path.join(output_root, task)
        os.makedirs(out_dir, exist_ok=True)

        # Construct command
        cmd = [
            "python", "compute_final_score.py",
            "--similarity_dir", sim_dir,
            "--weight_path", weight_csv,
            "--output_dir", out_dir,
            "--save_txt"
        ]

        print(f"[→] Running task: {task}")
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run compute_final_score.py")
    parser.add_argument("--similarity_root", required=True, help="Root directory containing similarity result folders")
    parser.add_argument("--weight_root", required=True, help="Root directory containing validation weights")
    parser.add_argument("--output_root", required=True, help="Root directory to save final scores")

    args = parser.parse_args()
    run_all(args.similarity_root, args.weight_root, args.output_root)

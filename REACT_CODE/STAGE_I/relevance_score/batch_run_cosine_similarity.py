import os
import argparse
import subprocess

def find_val_feature_paths(root_dir, suffix="_all_normalized.pt"):
    """
    Traverse the root directory to find all validation feature files ending with the given suffix.

    Returns:
        List of tuples: (task_name, validation_feature_path)
    """
    tasks = []
    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(sub_path):
            continue
        for file in os.listdir(sub_path):
            if file.endswith(suffix):
                val_path = os.path.join(sub_path, file)
                task_name = subdir.replace("dev_", "").replace("_conversations_60", "")
                tasks.append((task_name, val_path))
    return tasks

def run_all(train_chunks_dir, root_dir, output_root_dir):
    """
    Run cosine similarity computation for all validation tasks found under the root directory.

    Parameters:
        train_chunks_dir: Directory containing training representation chunks.
        root_dir: Root directory containing validation representations.
        output_root_dir: Base path to store similarity result outputs.
    """
    tasks = find_val_feature_paths(root_dir)

    for task_name, val_path in tasks:
        output_dir = os.path.join(output_root_dir, f"output_cosine_scores_mean_{task_name}")
        cmd = [
            "python", "cosine_similarity_matrix_reps.py",
            "--train_chunks_dir", train_chunks_dir,
            "--val_features_path", val_path,
            "--output_dir", output_dir,
            "--save_txt"
        ]
        print(f"[→] Running: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run cosine_similarity_matrix_reps.py")
    parser.add_argument("--train_chunks_dir", required=True, help="Directory containing training feature chunks")
    parser.add_argument("--root_dir", required=True, help="Root directory of validation features (e.g., output_dev_mean)")
    parser.add_argument("--output_root_dir", required=True, help="Base directory to store cosine similarity outputs (e.g., output_cosine_scores_mean)")

    args = parser.parse_args()
    run_all(args.train_chunks_dir, args.root_dir, args.output_root_dir)

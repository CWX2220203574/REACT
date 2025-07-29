import os
import argparse
import pandas as pd
import numpy as np


def compute_loss_weights(input_file: str, output_file: str, method: str = "linear"):
    """
    Compute sample weights based on validation loss and write them to output file.

    Parameters:
        input_file: path to CSV file containing 'id' and 'loss'
        output_file: output CSV path including columns id, loss, weight
        method: weighting strategy: 'exp' (default), 'linear'
    """
    df = pd.read_csv(input_file)

    if "loss" not in df.columns or "id" not in df.columns:
        raise ValueError("Input file must contain both 'id' and 'loss' columns.")

    loss = df["loss"].values

    if method == "exp":
        weights = np.exp(loss)
    elif method == "linear":
        weights = loss
    else:
        raise ValueError(f"Unsupported weight calculation method: {method}")

    weights = weights / weights.sum()
    df["weight"] = weights

    df.to_csv(output_file, index=False)
    print(f"Weights saved to: {output_file}")


def process_all_folders(root_dir: str):
    """
    Traverse all subfolders under the given root directory,
    and apply compute_loss_weights to losses.csv in each subfolder.
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        input_file = os.path.join(subdir_path, "losses.csv")
        output_file = os.path.join(subdir_path, "weighted_loss_linear.csv")

        if os.path.exists(input_file):
            try:
                compute_loss_weights(input_file, output_file)
            except Exception as e:
                print(f"[✘] Error processing {input_file}: {e}")
        else:
            print(f"[!] Skipped: {input_file} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compute weights from losses.csv in subfolders")
    parser.add_argument("--root_dir", "-r", required=True, help="Path to root directory containing subfolders")
    parser.add_argument("--method", required=False, help="Weighting method")
    args = parser.parse_args()
    process_all_folders(args.root_dir)

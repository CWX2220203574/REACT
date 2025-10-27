import os
import argparse
import torch
import pandas as pd

def load_weights(weight_csv_path):
    """
    Load sample weights from a CSV file and return as a float32 tensor.
    """
    df = pd.read_csv(weight_csv_path)
    weights = df['weight'].values
    return torch.tensor(weights, dtype=torch.float32)

def compute_and_save_scores(similarity_dir, weight_path, output_dir, save_txt=False):
    """
    Compute weighted scores by applying validation weights to cosine similarity matrices.

    Parameters:
    - similarity_dir: directory containing .pt files of cosine similarity matrices
    - weight_path: path to CSV file containing weights (e.g., weighted_loss_exp.csv)
    - output_dir: directory to save final weighted scores
    - save_txt: whether to also save scores in .txt format
    """
    os.makedirs(output_dir, exist_ok=True)

    weights = load_weights(weight_path)       # shape: [N_val]
    weights = weights.unsqueeze(1)            # shape: [N_val, 1]

    for file_name in os.listdir(similarity_dir):
        if not file_name.endswith(".pt"):
            continue

        sim_path = os.path.join(similarity_dir, file_name)
        sim_matrix = torch.load(sim_path)     # shape: [num_train, N_val]

        if sim_matrix.shape[1] != weights.shape[0]:
            raise ValueError(
                f"Mismatch: {file_name} has {sim_matrix.shape[1]} validation samples, "
                f"but weight file has {weights.shape[0]}"
            )

        weights = weights.to(sim_matrix.device)
        weighted_score = torch.matmul(sim_matrix, weights)  # shape: [num_train, 1]

        # Save as .pt
        output_pt_path = os.path.join(output_dir, file_name)
        torch.save(weighted_score, output_pt_path)
        print(f"[✔] Saved: {output_pt_path}")

        # Optionally save as .txt
        if save_txt:
            score_txt_path = os.path.splitext(output_pt_path)[0] + ".txt"
            with open(score_txt_path, "w") as f:
                for val in weighted_score.cpu().numpy():
                    f.write(f"{val[0]:.8f}\n")
            print(f"Saved TXT: {score_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute training sample scores by weighting cosine similarity with validation weights."
    )
    parser.add_argument("--similarity_dir", required=True, help="Directory containing cosine similarity .pt files")
    parser.add_argument("--weight_path", required=True, help="Path to weighted_loss_exp.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save weighted score outputs")
    parser.add_argument("--save_txt", action="store_true", help="Also save results as .txt files")

    args = parser.parse_args()

    compute_and_save_scores(
        similarity_dir=args.similarity_dir,
        weight_path=args.weight_path,
        output_dir=args.output_dir,
        save_txt=args.save_txt
    )

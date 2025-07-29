import pandas as pd
import numpy as np
import argparse

def compute_loss_weights(input_file: str, output_file: str, method: str = "exp", inverse: bool = False):
    """
    Compute task weights based on average validation loss, and save to output file.

    Parameters:
        input_file: Path to CSV file containing 'id' and 'avg_loss' columns
        output_file: Output CSV file path containing id, avg_loss, and computed weight
        method: Weighting method: 'exp', 'linear' (default), or 'avg_first_weight'
        inverse: If True, assign lower weights to higher loss values
    """
    df = pd.read_csv(input_file)

    loss = df["avg_loss"].values

    if method == "exp":
        if inverse:
            weights = np.exp(-loss)
        else:
            weights = np.exp(loss)
    elif method == "linear":
        if inverse:
            epsilon = 1e-10  # Prevent division by zero
            weights = 1 / (loss + epsilon)
        else:
            weights = loss
    elif method == "avg_first_weight":  # Weight based on distance to average
        median_loss = np.average(loss)
        epsilon = 1e-10  # Prevent division by zero
        weights = 1 / (np.abs(loss - median_loss) + epsilon)  # Closer to average gets higher weight
    else:
        raise ValueError(f"Unsupported weighting method: {method}")

    weights = weights / weights.sum()
    df["weight"] = weights

    df.to_csv(output_file, index=False)
    print(f"Weights saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute task-level weights based on average validation loss")
    parser.add_argument("--input", "-i", required=True, help="Path to input avg_loss.csv file")
    parser.add_argument("--output", "-o", required=True, help="Path to output weighted.csv file")

    parser.add_argument("--method", "-m", default="linear", choices=["exp", "linear", "avg_first_weight"],
                        help="Weighting method to use (default: linear)")
    parser.add_argument("--inverse", "-v", action="store_true", help="Use inverse weighting (higher loss = lower weight)")

    args = parser.parse_args()

    compute_loss_weights(args.input, args.output, args.method, args.inverse)

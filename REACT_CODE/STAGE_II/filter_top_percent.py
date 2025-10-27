import argparse
import pandas as pd

def filter_top_percent(input_csv, output_csv, top_percent=0.2):
    # Read the CSV
    df = pd.read_csv(input_csv)

    # Sort by score (from high to low)
    df_sorted = df.sort_values(by="score", ascending=False)

    # Calculate the top N rows
    top_n = int(len(df_sorted) * top_percent)
    df_top = df_sorted.head(top_n)

    # Save the result
    df_top.to_csv(output_csv, index=False)
    print(f"Saved the top {int(top_percent * 100)}% of the data to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter the top percentage of data samples based on score")
    parser.add_argument("--input_csv", required=True, help="Path to the input all_scores.csv file")
    parser.add_argument("--output_csv", required=True, help="Path to the output top_k_percent.csv file")
    parser.add_argument("--top_percent", type=float, default=0.2, help="The percentage to keep (default 0.2)")

    args = parser.parse_args()
    filter_top_percent(args.input_csv, args.output_csv, args.top_percent)

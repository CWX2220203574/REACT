import argparse
import pandas as pd

def filter_top_percent(input_csv, output_csv, top_percent=0.2):
    # 读取 CSV
    df = pd.read_csv(input_csv)

    # 按照 score 排序（从高到低）
    df_sorted = df.sort_values(by="score", ascending=False)

    # 计算前 N 条
    top_n = int(len(df_sorted) * top_percent)
    df_top = df_sorted.head(top_n)

    # 保存结果
    df_top.to_csv(output_csv, index=False)
    print(f"[✔] 已保存前 {int(top_percent * 100)}% 数据到: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="筛选得分前若干百分比的数据样本")
    parser.add_argument("--input_csv", required=True, help="输入的 all_scores.csv 文件路径")
    parser.add_argument("--output_csv", required=True, help="输出的 top_k_percent.csv 文件路径")
    parser.add_argument("--top_percent", type=float, default=0.2, help="保留的前百分比（默认 0.2）")

    args = parser.parse_args()
    filter_top_percent(args.input_csv, args.output_csv, args.top_percent)

import os
import json
import argparse
import re

def add_new_ids_to_chunks(input_dir, output_dir):
    """
    Add unique 'new_id' fields to each sample in chunk_*.json files.

    Parameters:
    - input_dir: directory containing original chunk_*.json files
    - output_dir: directory to save updated JSON files with new_id
    """
    os.makedirs(output_dir, exist_ok=True)

    global_index = 1  # Global index starting value

    def extract_chunk_number(fname):
        match = re.match(r"chunk_(\d+)\.json", fname)
        return int(match.group(1)) if match else float('inf')

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".json") and f.startswith("chunk_")]
    all_files = sorted(all_files, key=extract_chunk_number)

    # Traverse all chunk_*.json files
    for fname in all_files:
        if not fname.endswith(".json") or not fname.startswith("chunk_"):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        with open(input_path, 'r') as f:
            data = json.load(f)

        # Get current chunk number, e.g., chunk_23 → chunk23
        chunk_num = fname.replace("chunk_", "").replace(".json", "")
        chunk_tag = f"chunk{int(chunk_num)}"

        for item in data:
            new_id = f"{chunk_tag}_{global_index:07d}"  # 7-digit zero-padded global ID
            item["new_id"] = new_id
            global_index += 1

        with open(output_path, 'w') as f_out:
            json.dump(data, f_out, indent=2)

        print(f"[✔] {fname} processed, {len(data)} records written → {output_path}")

    print(f"[✅] All done. Total new_ids generated: {global_index - 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add unique 'new_id' field to chunk_*.json files")
    parser.add_argument("--input_dir", required=True, help="Directory containing original JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to save JSON files with new_id")
    args = parser.parse_args()

    add_new_ids_to_chunks(args.input_dir, args.output_dir)

import torch
import os
import argparse
import torch.nn.functional as F

def calculate_cosine_similarity(train_features: torch.Tensor, val_features: torch.Tensor):
    """
    Compute cosine similarity matrix of shape [N_train, N_val].
    """
    return torch.matmul(train_features, val_features.T)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load validation features
    try:
        val_feats = torch.load(args.val_features_path, map_location=device)
        if not torch.is_tensor(val_feats):
            val_feats = torch.tensor(val_feats)
        val_feats = val_feats.float().to(device)
    except Exception as e:
        print(f"[Error] Failed to load validation features: {e}")
        return

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Traverse all training chunks
    chunk_files = sorted([
        f for f in os.listdir(args.train_chunks_dir) if f.endswith(".pt")
    ])

    for chunk_file in chunk_files:
        print(f"Processing {chunk_file}...")
        chunk_path = os.path.join(args.train_chunks_dir, chunk_file)

        try:
            train_feats = torch.load(chunk_path, map_location=device)
            if not torch.is_tensor(train_feats):
                train_feats = torch.tensor(train_feats)
            train_feats = train_feats.float().to(device)
        except Exception as e:
            print(f"[Error] Failed to load training features from {chunk_file}: {e}")
            continue

        # Compute cosine similarity
        cosine_sim = calculate_cosine_similarity(train_feats, val_feats)
        print(f"Cosine similarity shape: {cosine_sim.shape}")

        # Save .pt file
        save_path = os.path.join(args.output_dir, chunk_file.replace('.pt', '_cosine.pt'))
        torch.save(cosine_sim, save_path)
        print(f"[✔] Similarity matrix saved to: {save_path}")

        # Optionally: save as .txt (human-readable)
        if args.save_txt:
            txt_path = os.path.join(args.output_dir, chunk_file.replace('.pt', '_cosine.txt'))
            with open(txt_path, 'w') as f:
                f.write(f"Cosine similarity shape: {cosine_sim.shape}\n\n")
                f.write(str(cosine_sim.cpu()))
            print(f"Text format saved to: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cosine similarity for feature representations.")
    parser.add_argument("--train_chunks_dir", type=str, required=True, help="Directory of training feature chunks")
    parser.add_argument("--val_features_path", type=str, required=True, help="Validation feature representation (.pt file)")
    parser.add_argument("--output_dir", type=str, default="cosine_outputs", help="Directory to save similarity outputs")
    parser.add_argument("--save_txt", action="store_true", help="Whether to also save similarity matrix as .txt")
    args = parser.parse_args()
    main(args)

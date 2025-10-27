#!/usr/bin/env python
"""
This script loads a LLaVA(-LoRA) model, iterates over a chunked dataset,
computes forward loss, extracts final-layer hidden representations (mean pooled),
and saves them along with per-sample loss values. No gradients or backpropagation is involved.
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from llava.model.builder import load_pretrained_model
from utils import LazySupervisedDataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import datetime
from tqdm import tqdm
from trak.projectors import BasicProjector, ProjectionType

import pandas as pd
from torch.nn.functional import normalize

@dataclass
class Args:
    train_file: str
    model_path: str
    # image_folder: str
    output_path: str
    torch_dtype: str = "bfloat16"
    max_length: int = 2048
    save_path: str = "saved_grads.pt"
    task: Optional[str] = field(default=None, metadata={"help": "Specify the task from mmvet. One of variables of task and train_file must be specified"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


def save_detailed_model_parameters(model, output_file="detailed_model_parameters.txt"):
    """
    Save all model parameters including tensor values, shapes, and flags to a file.

    Parameters:
    - model: the PyTorch model
    - output_file: path to write parameter dump
    """
    with open(output_file, 'w') as f:
        f.write(f"Detailed Model Parameters Dump - {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: Parameter containing:\n")
            f.write(str(param.data))
            f.write(f"\nShape: {tuple(param.shape)}")
            f.write(f"\nRequires grad: {param.requires_grad}")
            f.write(f"\nDevice: {param.device}")
            f.write(f"\nDtype: {param.dtype}\n")
            f.write("\n" + "-" * 80 + "\n")


def initialize_model(args):
    """
    Load a pretrained LLaVA or LLaVA-LoRA model and tokenizer.

    Parameters:
    - args: input args from HfArgumentParser

    Returns:
    - tokenizer, model, image_processor
    """
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    print("Loading model from:", args.model_path)

    if 'lora' in args.model_path:
        if '13b' in args.model_path:
            model_base = "./checkpoints/vicuna-13b-v1.5"
            model_name = "llava-v1.5-13b-lora"
        else:  
            model_base = "/data2/cwx/icons/checkpoints/vicuna-7b-v1.5"
            model_name = "llava-v1.5-7b-lora"
    else:
        model_base = None
        model_name = "llava-v1.5-13b" if '13b' in args.model_path else "llava-v1.5-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base=model_base,  
        model_name=model_name,
        load_8bit=False,
        load_4bit=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if 'lora' in args.model_path:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_path)
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.to(device)
        print(f"LoRA model loaded successfully and moved to {device}")
        save_detailed_model_parameters(model, "lora_model_parameters.txt")
        lora_params = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
        print(f"\nFound {len(lora_params)} LoRA parameters")
        print("\nSample LoRA parameters:")
        for param in lora_params[:5]:
            print(f"- {param}")

    print("Model loaded successfully")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        print("Token embeddings resized")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return tokenizer, model, image_processor


def load_raw_dataset(data_args, tokenizer, train_files, sample_percentage=None, seed=None):
    """
    Load LazySupervisedDataset from JSON/JSONL file.

    Parameters:
    - data_args: DataArguments
    - tokenizer: tokenizer instance
    - train_files: input file path
    """
    print(f"Train files received: {train_files}")
    
    if train_files is None:
        raise ValueError("train_files cannot be None. Please provide a valid data path.")
    
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_files,
        data_args=data_args
    )
    
    return dataset


def get_dataset(data_args, files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    """
    Wrap dataset with truncation logic, returning PyTorch-compatible samples.
    """
    raw_datasets = load_raw_dataset(data_args, tokenizer, files, sample_percentage=sample_percentage, seed=seed)
    
    class TruncatedDataset:
        def __init__(self, dataset, max_length):
            self.dataset = dataset
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item.get("image", None)
            
            if len(item["input_ids"]) > self.max_length:
                item["input_ids"] = item["input_ids"][:self.max_length]
                if "labels" in item:
                    item["labels"] = item["labels"][:self.max_length]
            
            if not isinstance(item["input_ids"], torch.Tensor):
                item["input_ids"] = torch.tensor(item["input_ids"])
            if "labels" in item and not isinstance(item["labels"], torch.Tensor):
                item["labels"] = torch.tensor(item["labels"])
            
            if image is not None:
                item["image"] = image
            
            return item
    
    return TruncatedDataset(raw_datasets, max_seq_length)


def merge_reps_chunks(save_dir, chunk_name, prefix="reps", normalize_variants=(False, True)):
    """
    Merge reps-*.pt files into chunk_name_all_normalized.pt and chunk_name_all_unnormalized.pt

    Parameters:
    - save_dir: directory containing reps-*.pt files
    - chunk_name: base name for merged output
    - prefix: chunk file prefix, default "reps"
    - normalize_variants: whether to save both normalized and unnormalized versions
    """
    files = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pt")]
    files.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))  # sort by chunk index
    
    for normalize_flag in normalize_variants:
        merged = []
        for f in files:
            path = os.path.join(save_dir, f)
            data = torch.load(path)
            if normalize_flag:
                data = normalize(data, dim=1)
            merged.append(data)

        if merged:
            merged_tensor = torch.cat(merged, dim=0)
            suffix = "normalized" if normalize_flag else "unnormalized"
            output_file = os.path.join(save_dir, f"{chunk_name}_all_{suffix}.pt")
            torch.save(merged_tensor, output_file)
            print(f"Saved {output_file} | Shape: {merged_tensor.shape}")
            
def build_custom_collate_fn(tokenizer):
    """
    Build a custom collate function that preserves sample 'id' and applies padding.
    """
    def custom_collate_fn(batch):
        id_list = [item.pop("id") for item in batch]

        base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
        batch_dict = base_collator(batch)

        batch_dict["id"] = id_list
        return batch_dict
    return custom_collate_fn


def get_dataloader(dataset, tokenizer, batch_size=1):
    """
    Return a DataLoader with batch size = 1, using custom collate function.
    """
    custom_collate = build_custom_collate_fn(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)
    print(f"There are {len(dataset)} examples in the dataset")
    return dataloader



def main():
    """
    Main routine: Load model and data, extract forward hidden states, save reps and losses.
    """
    parser = HfArgumentParser((Args, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()

    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    tokenizer, model, image_processor = initialize_model(args)
    data_args.image_processor = image_processor
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    print("\nLoading dataset...")
    dataset = get_dataset(data_args, args.train_file, tokenizer, args.max_length)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)

    print(f"Dataset loaded from {'task' if args.task else 'file'}:", args.task or args.train_file)
    print("Number of samples in the dataset:", len(dataset))

    print("\nRunning forward pass and saving hidden representations...")

    # Prepare output directory
    chunk_name = os.path.splitext(os.path.basename(args.train_file))[0]
    save_dir = os.path.join(args.output_path, f"train_{chunk_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Containers for saving
    all_reps = []
    all_ids = []
    all_losses = []
    chunk_size = 128
    chunk_counter = 0

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for idx, batch in enumerate(tqdm(dataloader)):
        model.zero_grad()
        torch.cuda.empty_cache()
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch["attention_mask"].to(model_device)
        labels = batch["labels"].to(model_device)
        sample_id = batch["id"][0]

        images = batch.get("image", None)
        with torch.no_grad():
            if images is not None:
                images = images.to(dtype=model_dtype, device=model_device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True
                )

        loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]  # shape: (B, T, D)
        cls_repr = last_hidden_state.mean(dim=1).cpu()  # mean-pool across tokens

        all_ids.append(sample_id)
        all_losses.append(loss.item())
        all_reps.append(cls_repr)

        if len(all_reps) == chunk_size:
            print(f"Processing chunk {chunk_counter}")
            print(f"Processing chunk of size {len(all_reps)}")
            chunk_tensor = torch.cat(all_reps, dim=0)
            print(chunk_tensor.shape)
            torch.save(chunk_tensor, os.path.join(save_dir, f"reps-{chunk_counter}.pt"))
            chunk_counter += 1
            all_reps = []

        model.zero_grad(set_to_none=True)

    # Save final incomplete chunk
    if len(all_reps) > 0:
        print(f"Processing chunk {chunk_counter}")
        print(f"Processing chunk of size {len(all_reps)}")
        chunk_tensor = torch.cat(all_reps, dim=0)
        torch.save(chunk_tensor, os.path.join(save_dir, f"reps-{chunk_counter}.pt"))

    # Save loss values
    loss_df = pd.DataFrame({"id": all_ids, "loss": all_losses})
    loss_df.to_csv(os.path.join(save_dir, "losses.csv"), index=False)
    print(f"Saved all representations and loss data to {save_dir}")

    # Merge reps
    merge_reps_chunks(save_dir, chunk_name)


if __name__ == "__main__":
    main()

"""Example script demonstrating Sheared LLaMA pruning for LLaMA 3.1 8B with distributed training on A100s"""

import os
import json
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from LLMPruner.pruner.sheared_llama_pruner import ShearedLLaMAPruner
import wandb
from tqdm import tqdm
import gc

# Enhanced memory management settings for A100
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help="Path to base model")
    parser.add_argument("--output_dir", type=str, default="prune_log/llama31_sheared",
                      help="Directory to save pruned model")
    parser.add_argument("--target_layer_sparsity", type=float, default=0.2,
                      help="Target sparsity for layer pruning")
    parser.add_argument("--target_head_sparsity", type=float, default=0.3,
                      help="Target sparsity for attention head pruning")
    parser.add_argument("--target_hidden_sparsity", type=float, default=0.1,
                      help="Target sparsity for hidden dimension pruning")
    parser.add_argument("--target_intermediate_sparsity", type=float, default=0.4,
                      help="Target sparsity for intermediate dimension pruning")
    parser.add_argument("--num_epochs", type=int, default=2,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=64,
                      help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for hard concrete distribution")
    parser.add_argument("--lambda_init", type=float, default=0.1,
                      help="Initial value for Lagrange multipliers")
    parser.add_argument("--wandb_project", type=str, default="llama31_sheared",
                      help="Weights & Biases project name")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing")
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")
    return parser.parse_args()

def setup_distributed():
    """Initialize distributed training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def create_dataloader(tokenizer, args, is_distributed=False):
    """Create dataloader using 100K examples from WikiText-103"""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Loading WikiText-103 (100K examples)...")
    
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:100]")
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Dataset size: {len(dataset)} examples")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Tokenizing dataset...")
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    sampler = DistributedSampler(tokenized_dataset) if is_distributed else None
    
    total_batch_size = args.batch_size * torch.cuda.device_count() if is_distributed else args.batch_size
    num_batches = len(tokenized_dataset) // total_batch_size
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Total batches per epoch: {num_batches}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {total_batch_size * args.gradient_accumulation_steps}")
    
    return DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=default_data_collator,
        pin_memory=True,
        drop_last=True,
        num_workers=2
    ), num_batches

def save_metrics(model, output_dir):
    """Save model architecture metrics"""
    metrics = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": {
            "num_layers": len(model.module.layers if hasattr(model, 'module') else model.layers),
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads
        }
    }
    
    if int(os.environ.get("LOCAL_RANK", -1)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

def main():
    args = parse_args()
    
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision('high')
    
    if local_rank == 0:
        wandb.init(project=args.wandb_project)
        print("\n=== Starting Sheared LLaMA Training ===")
        print(f"GPUs: {torch.cuda.device_count()}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * torch.cuda.device_count()}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Using A100 optimizations")
    
    if local_rank == 0:
        print(f"\nLoading model from {args.base_model}")
    
    clear_memory()
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=f"cuda:{local_rank}",
        max_memory={f"cuda:{local_rank}": "70GiB"}
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    if local_rank == 0:
        print("\nInitializing Sheared LLaMA pruner")
        print(f"Target sparsities:")
        print(f"  Layer: {args.target_layer_sparsity}")
        print(f"  Head: {args.target_head_sparsity}")
        print(f"  Hidden: {args.target_hidden_sparsity}")
        print(f"  Intermediate: {args.target_intermediate_sparsity}")
    
    pruner = ShearedLLaMAPruner(
        model,
        target_layer_sparsity=args.target_layer_sparsity,
        target_head_sparsity=args.target_head_sparsity,
        target_hidden_sparsity=args.target_hidden_sparsity,
        target_intermediate_sparsity=args.target_intermediate_sparsity,
        temperature=args.temperature,
        lambda_init=args.lambda_init
    )
    
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        static_graph=True,
        gradient_as_bucket_view=True
    )
    
    if local_rank == 0:
        print("\nCreating dataloader")
    dataloader, num_batches = create_dataloader(tokenizer, args, is_distributed=True)
    
    trainable_params = []
    trainable_params.extend(model.parameters())
    for mask in pruner.masks.values():
        trainable_params.append(mask.log_alpha)
    trainable_params.extend([
        pruner.optimizer.lambda_layer,
        pruner.optimizer.lambda_head,
        pruner.optimizer.lambda_hidden,
        pruner.optimizer.lambda_int
    ])
    
    # Setup optimizer with only fused=True (removed foreach)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        eps=1e-8,
        fused=True,
        betas=(0.9, 0.95)
    )
    
    if local_rank == 0:
        print("\nStarting training")
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        dataloader.sampler.set_epoch(epoch)
        optimizer.zero_grad()
        
        if local_rank == 0:
            pbar = tqdm(total=num_batches, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(dataloader):
            clear_memory()
            
            batch = {k: v.to(f"cuda:{local_rank}", non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"]
                )
                
                loss = pruner.compute_loss(outputs.loss)
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                pruner.update_lagrange_multipliers()
                clear_memory()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            if local_rank == 0:
                pbar.update(1)
                if batch_idx % 10 == 0:
                    binary_masks = pruner.get_pruning_mask()
                    sparsity_metrics = {
                        "layer_sparsity": 1 - binary_masks['layer'].float().mean().item(),
                        "head_sparsity": 1 - binary_masks['head'].float().mean().item(),
                        "hidden_sparsity": 1 - binary_masks['hidden'].float().mean().item(),
                        "intermediate_sparsity": 1 - binary_masks['intermediate'].float().mean().item()
                    }
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                        'layer_sparsity': f'{sparsity_metrics["layer_sparsity"]:.2%}',
                        'head_sparsity': f'{sparsity_metrics["head_sparsity"]:.2%}'
                    })
                    
                    wandb.log({
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        **sparsity_metrics,
                        "epoch": epoch,
                        "batch": batch_idx
                    })
        
        if local_rank == 0:
            pbar.close()
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch} complete, Average Loss: {avg_loss:.4f}")
            clear_memory()
    
    if local_rank == 0:
        print("\nApplying final pruning")
        clear_memory()
        pruner.apply_pruning()
        
        print(f"Saving model to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'model': model.module if hasattr(model, 'module') else model,
            'tokenizer': tokenizer,
            'config': model.config,
            'args': args
        }, os.path.join(args.output_dir, "pytorch_model.bin"))
        
        save_metrics(model, args.output_dir)
        
        print("\nTraining complete! Model saved with metrics.")
        wandb.finish()
    
    destroy_process_group()

if __name__ == "__main__":
    main()

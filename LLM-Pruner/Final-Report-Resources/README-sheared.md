# Sheared LLaMA Implementation for LLM-Pruner

This document describes the implementation of the Sheared LLaMA pruning approach adapted for LLaMA 3.1 8B, as described in ["Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"](https://arxiv.org/pdf/2310.06694v2).

## Table of Contents
- [Installation](#installation)
- [Pruning LLaMA 3.1 8B](#pruning-llama-31-8b)
- [Post-Training](#post-training)
- [Evaluation](#evaluation)
- [Model Architecture Analysis](#model-architecture-analysis)

## Installation

1. Install base requirements:
```bash
pip install -r requirements.txt
```

2. Additional requirements for distributed training:
```bash
pip install deepspeed accelerate
```

## Pruning LLaMA 3.1 8B

### 1. Pruning to 3B Parameters

```bash
python examples/sheared_llama_example.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir prune_log/llama31_3b_sheared \
    --target_layer_sparsity 0.2 \
    --target_head_sparsity 0.3 \
    --target_hidden_sparsity 0.1 \
    --target_intermediate_sparsity 0.4 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --temperature 1.0 \
    --lambda_init 0.1 \
    --wandb_project llama31_sheared
```

### 2. Pruning to 2B Parameters

```bash
python examples/sheared_llama_example.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir prune_log/llama31_2b_sheared \
    --target_layer_sparsity 0.3 \
    --target_head_sparsity 0.4 \
    --target_hidden_sparsity 0.2 \
    --target_intermediate_sparsity 0.5 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --temperature 1.0 \
    --lambda_init 0.1 \
    --wandb_project llama31_sheared
```

The pruning process uses our Sheared LLaMA implementation which includes:
- Structured pruning masks with hard concrete distributions
- GQA-aware head pruning
- Constrained optimization with Lagrange multipliers
- Dynamic batch loading

The pruned models will be saved in `prune_log/llama31_3b_sheared/` and `prune_log/llama31_2b_sheared/` respectively.

## Post-Training

### 1. Training with Alpaca (50K samples)

For 3B model:
```bash
CUDA_VISIBLE_DEVICES=0 python post_training.py \
    --prune_model prune_log/llama31_3b_sheared/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --output_dir tune_log/llama31_3b_sheared_alpaca \
    --wandb_project llama31_sheared_tune
```

For 2B model:
```bash
CUDA_VISIBLE_DEVICES=0 python post_training.py \
    --prune_model prune_log/llama31_2b_sheared/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --output_dir tune_log/llama31_2b_sheared_alpaca \
    --wandb_project llama31_sheared_tune
```

### 2. Training with LaMini-instruction (2.59M samples)

For distributed training of 3B model:
```bash
deepspeed --include=localhost:0,1,2,3 post_training.py \
    --prune_model prune_log/llama31_3b_sheared/pytorch_model.bin \
    --data_path MBZUAI/LaMini-instruction \
    --lora_r 8 \
    --num_epochs 3 \
    --output_dir tune_log/llama31_3b_sheared_lamini \
    --extra_val_dataset wikitext2,ptb \
    --wandb_project llama31_sheared_lamini \
    --learning_rate 5e-5 \
    --cache_dataset
```

For 2B model:
```bash
deepspeed --include=localhost:0,1,2,3 post_training.py \
    --prune_model prune_log/llama31_2b_sheared/pytorch_model.bin \
    --data_path MBZUAI/LaMini-instruction \
    --lora_r 8 \
    --num_epochs 3 \
    --output_dir tune_log/llama31_2b_sheared_lamini \
    --extra_val_dataset wikitext2,ptb \
    --wandb_project llama31_sheared_lamini \
    --learning_rate 5e-5 \
    --cache_dataset
```

## Evaluation

### 1. Prepare Evaluation Files

For Alpaca-trained models:
```bash
# For 3B model
cd tune_log/llama31_3b_sheared_alpaca
export epoch=2000  # or your chosen epoch
cp adapter_config.json checkpoint-$epoch/
mv checkpoint-$epoch/pytorch_model.bin checkpoint-$epoch/adapter_model.bin

# For 2B model
cd tune_log/llama31_2b_sheared_alpaca
export epoch=2000
cp adapter_config.json checkpoint-$epoch/
mv checkpoint-$epoch/pytorch_model.bin checkpoint-$epoch/adapter_model.bin
```

Similar steps for LaMini-trained models:
```bash
# For 3B model
cd tune_log/llama31_3b_sheared_lamini
export epoch=2000
cp adapter_config.json checkpoint-$epoch/
mv checkpoint-$epoch/pytorch_model.bin checkpoint-$epoch/adapter_model.bin

# For 2B model
cd tune_log/llama31_2b_sheared_lamini
export epoch=2000
cp adapter_config.json checkpoint-$epoch/
mv checkpoint-$epoch/pytorch_model.bin checkpoint-$epoch/adapter_model.bin
```

### 2. Run Evaluation

For 3B Alpaca model:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    tune_log/llama31_3b_sheared_alpaca \
    prune_log/llama31_3b_sheared/pytorch_model.bin \
    2000
```

For 2B Alpaca model:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    tune_log/llama31_2b_sheared_alpaca \
    prune_log/llama31_2b_sheared/pytorch_model.bin \
    2000
```

Similar commands for LaMini-trained models, just replace the tune_log paths.

## Model Architecture Analysis

### 1. Save Model Metrics

To analyze and save metrics for each model:
```bash
# For 3B model
python test_speedup.py \
    --model_type pruneLLM \
    --ckpt prune_log/llama31_3b_sheared/pytorch_model.bin \
    --save_metrics metrics_3b.json

# For 2B model
python test_speedup.py \
    --model_type pruneLLM \
    --ckpt prune_log/llama31_2b_sheared/pytorch_model.bin \
    --save_metrics metrics_2b.json
```

This will save detailed metrics including:
- Parameter count
- Memory usage
- Inference latency
- Model architecture details:
  - Number of layers (after layer pruning)
  - Number of attention heads per layer (after head pruning)
  - Hidden dimensions (after hidden dim pruning)
  - Intermediate dimensions (after FFN pruning)
- GQA configuration details
- Throughput measurements

### 2. Compare with Original Model

```bash
python test_speedup.py \
    --model_type pretrain \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --save_metrics metrics_original.json
```

The metrics files can be used to create comparison tables and analyze the efficiency gains from pruning.

## Expected Results

| Model | Parameters | Memory | Latency | Avg. Score (Alpaca) | Avg. Score (LaMini) |
|-------|------------|---------|----------|-------------------|-------------------|
| LLaMA 3.1 8B | 8B | ~16GB | 1x | Baseline | Baseline |
| Sheared 3B | 3B | ~6GB | ~0.4x | TBD | TBD |
| Sheared 2B | 2B | ~4GB | ~0.3x | TBD | TBD |

The evaluation should include:
1. Language modeling metrics:
   - Perplexity on WikiText-103
   - LAMBADA accuracy
2. Downstream task performance:
   - GLUE benchmark scores
   - SQuAD v1.1/v2.0 F1 scores
3. Efficiency metrics:
   - Inference latency
   - Memory usage
   - Throughput (tokens/second)
4. Architecture analysis:
   - Layer distribution
   - Head importance patterns
   - Hidden/intermediate dimension pruning patterns

Note: Fill in the actual scores after running evaluations. Compare against similar-sized models to demonstrate the effectiveness of the Sheared LLaMA approach.

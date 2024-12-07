# Sheared LLaMA Implementation Instructions

This document provides detailed instructions for implementing structured pruning for LLaMA 3.1 8B using the Sheared LLaMA approach.

## Project Structure

New files added to the LLM-Pruner repository:

```
LLMPruner/
├── pruner/
│   ├── sheared_llama_pruner.py  # Main pruning implementation
│   ├── masks.py                 # Pruning mask implementations
│   └── optimization.py          # Optimization and loss functions
├── sheared_llama_example.py # Training script
|── prune_log # contains pruned model
|── tune_log # contains post-trained pruned model
├── README-sheared.md           # Implementation details
└── hardware_setup.md           # Hardware configuration
```

## Hardware Requirements

- 2x NVIDIA A100 GPUs (80GB each)
- CUDA 11.8+
- 100GB+ storage
- 64GB+ RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/horseee/LLM-Pruner.git
cd LLM-Pruner
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install base requirements:
```bash
pip install --upgrade pip
pip install -r requirement.txt
```

4. Install additional dependencies:
```bash
pip install torch==2.1.1 torchvision torchaudio
pip install transformers datasets accelerate
pip install deepspeed wandb
pip install huggingface_hub
```

5. Login to Hugging Face (required for LLaMA model access):
```bash
huggingface-cli login
```

## Environment Setup

1. Set environment variables for optimal performance:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

2. Configure CUDA settings in your script:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## Running the Training

1. Basic training command:
```bash
torchrun --nproc_per_node=2 sheared_llama_example.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir prune_log/llama31_3b_sheared \
    --target_layer_sparsity 0.2 \
    --target_head_sparsity 0.3 \
    --target_hidden_sparsity 0.1 \
    --target_intermediate_sparsity 0.4 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 128 \
    --gradient_checkpointing \
    --wandb_project llama31_sheared
```

2. With environment variables:
```bash
NCCL_TIMEOUT=1800 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --nproc_per_node=2 sheared_llama_example.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir prune_log/llama31_3b_sheared \
    --target_layer_sparsity 0.2 \
    --target_head_sparsity 0.3 \
    --target_hidden_sparsity 0.1 \
    --target_intermediate_sparsity 0.4 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 128 \
    --gradient_checkpointing \
    --wandb_project llama31_sheared
```

## Command Line Arguments

- `--base_model`: Path to base LLaMA model
- `--output_dir`: Directory to save pruned model
- `--target_layer_sparsity`: Target sparsity for layer pruning (0.0-1.0)
- `--target_head_sparsity`: Target sparsity for attention head pruning (0.0-1.0)
- `--target_hidden_sparsity`: Target sparsity for hidden dimension pruning (0.0-1.0)
- `--target_intermediate_sparsity`: Target sparsity for intermediate dimension pruning (0.0-1.0)
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization
- `--batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Number of gradient accumulation steps
- `--max_length`: Maximum sequence length
- `--gradient_checkpointing`: Enable gradient checkpointing (recommended for large models)
- `--wandb_project`: Weights & Biases project name for logging

## Output Structure

The training will create the following outputs:

```
prune_log/
└── llama31_3b_sheared/
    ├── pytorch_model.bin  # Pruned model weights
    ├── metrics.json      # Model architecture metrics
    └── config.json       # Model configuration
```

## Monitoring Training

1. GPU Usage:
```bash
watch -n 1 nvidia-smi
```

2. Memory Usage:
```bash
nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu --format=csv -l 1
```

3. Training Progress:
- View real-time metrics on Weights & Biases dashboard
- Monitor terminal output for loss and sparsity metrics

## Troubleshooting

1. Out of Memory (OOM) Errors:
- Reduce batch size
- Increase gradient accumulation steps
- Enable gradient checkpointing
- Reduce sequence length

2. NCCL Errors:
- Increase NCCL timeout
- Enable async error handling
- Check GPU connectivity

3. Training Issues:
- Check GPU utilization
- Monitor memory usage
- Verify data loading

## References

1. Sheared LLaMA Paper:
   - "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"
   - https://arxiv.org/abs/2310.06694

2. LLM-Pruner Repository:
   - https://github.com/horseee/LLM-Pruner

3. Hardware Setup:
   - See hardware_setup.md for detailed A100 configuration

## Notes

- The training script uses distributed data parallel (DDP) for multi-GPU training
- Gradient checkpointing is enabled by default to manage memory usage
- The script includes automatic mixed precision (AMP) for efficient training
- Memory management is optimized for A100 GPUs
- The pruning process is structured and maintains model architecture constraints

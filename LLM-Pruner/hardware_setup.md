# Hardware Setup and Optimizations for Sheared LLaMA

This document details the hardware configuration and optimizations used for implementing Sheared LLaMA pruning on LLaMA 3.1 8B.

## Hardware Configuration

### GPU Setup
- **GPUs**: 2x NVIDIA A100 (80GB each)
- **Memory per GPU**: 80GB HBM2e
- **GPU Interconnect**: NVLink
- **Storage**: 100GB SSD
- **CUDA Version**: 11.8
- **PyTorch Version**: 2.1.1

### Memory Requirements
- Base model (LLaMA 3.1 8B): ~16GB
- Training overhead: ~40GB
- Gradient storage: ~20GB
- Optimizer states: ~4GB

## Software Optimizations

### 1. Memory Management
```python
# Memory allocator settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Enable tensor cores
torch.set_float32_matmul_precision('high')
```

### 2. Training Configuration
```python
{
    "batch_size": {
        "per_gpu": 4,
        "gradient_accumulation": 8,
        "effective_total": 64  # 4 * 2 GPUs * 8 accumulation
    },
    "sequence_length": 128,
    "precision": "bfloat16",
    "optimizer": {
        "type": "AdamW",
        "fused": True,
        "betas": [0.9, 0.95]
    }
}
```

### 3. A100-Specific Features
1. Tensor Core Operations
   - Automatic mixed precision (bfloat16)
   - Optimized matrix multiplications

2. Memory Bandwidth Utilization
   - Fused optimizer operations
   - Efficient memory access patterns
   - Gradient checkpointing

3. Multi-GPU Efficiency
   - NVLink communication
   - Distributed data parallel
   - Efficient gradient synchronization

## Setup Instructions

1. Create RunPod Instance:
```bash
# Select configuration:
- Template: PyTorch 2.1.1
- GPUs: 2x A100 (80GB)
- Volume Size: 50GB
```

2. Environment Setup:
```bash
# Clone repository
git clone https://github.com/horseee/LLM-Pruner.git
cd LLM-Pruner

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
pip install deepspeed accelerate wandb huggingface_hub
```

3. Hugging Face Setup:
```bash
# Login to Hugging Face
huggingface-cli login
# Request access to meta-llama/Llama-3.1-8B-Instruct
```

4. Training Launch:
```bash
# Run distributed training
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

## Performance Monitoring

### 1. Memory Usage
Monitor GPU memory usage:
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Memory stats during training
nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu --format=csv -l 1
```

### 2. Training Metrics
Track via Weights & Biases:
- Loss curves
- Sparsity levels
- Memory utilization
- GPU utilization
- Training throughput

### 3. Expected Performance
- Training time: ~2-3 hours per epoch
- Memory usage: 65-75GB per GPU
- GPU utilization: >90%
- Training throughput: ~800-1000 tokens/sec

## Troubleshooting

### Common Issues

1. Out of Memory (OOM):
```bash
# Solutions:
- Reduce batch size
- Increase gradient accumulation steps
- Enable gradient checkpointing
- Reduce sequence length
```

2. GPU Synchronization:
```bash
# Check NCCL environment
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Monitor communication
nvidia-smi nvlink -g 0
```

3. Performance Optimization:
```bash
# Profile training
nsys profile -o profile_output torchrun ...

# Analyze bottlenecks
nsight-systems profile_output.qdrep
```

## References

1. NVIDIA A100 Documentation
   - [A100 Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
   - [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

2. PyTorch Optimization Resources
   - [Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
   - [Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)

3. Related Papers
   - Sheared LLaMA Paper
   - LLM-Pruner Paper

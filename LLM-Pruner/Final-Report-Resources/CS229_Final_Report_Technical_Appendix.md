# Technical Appendix: Structured Pruning of LLaMA 3.1 8B
## CS229 Final Project Technical Details

This document serves as a technical appendix to the main report, providing detailed mathematical derivations, algorithm descriptions, and in-depth analyses.

## Table of Contents
- [C. Mathematical Derivations](#c-mathematical-derivations)
- [D. Detailed GQA Analysis](#d-detailed-gqa-analysis)
- [E. Comparative Analysis](#e-comparative-analysis)
- [F. Detailed Algorithm Descriptions](#f-detailed-algorithm-descriptions)
- [G. Hardware-Specific Optimizations](#g-hardware-specific-optimizations)

## C. Mathematical Derivations

### C.1 Hard Concrete Distribution Gradients

The hard concrete distribution is defined as:

```
s = sigmoid((log α + log u - log(1-u))/β)
z = min(1, max(0, s))
```

The gradient computation follows:

1. Forward Pass:
```
l = log α
u ~ Uniform(0,1)
noise = log u - log(1-u)
s = sigmoid((l + noise)/β)
z = min(1, max(0, s))
```

2. Backward Pass:
```
∂L/∂α = ∂L/∂z * ∂z/∂s * ∂s/∂l * ∂l/∂α
where:
∂z/∂s = 1 if 0 < s < 1, else 0
∂s/∂l = s(1-s)/β
∂l/∂α = 1/α
```

3. Expected Gradient:
```
E[∂L/∂α] = E[∂L/∂z] * (1/β) * P(0 < s < 1)
```

### C.2 GQA Constraint Preservation

For GQA with n query heads and m key/value heads (n > m):

1. Attention Pattern:
```
Q = [Q₁, ..., Qₙ] ∈ R^{b×n×d}
K = [K₁, ..., Kₘ] ∈ R^{b×m×d}
V = [V₁, ..., Vₘ] ∈ R^{b×m×d}

Attention(Qᵢ, K_j, V_j) = softmax(QᵢK_j^T/√d)V_j
where j = ⌊i/k⌋, k = n/m
```

2. Pruning Constraint:
```
For each key/value head j:
z_head[i] = z_head[i'] ∀i,i' where ⌊i/k⌋ = ⌊i'/k⌋ = j
```

3. Proof of Preservation:
```
Let G_j = {i | ⌊i/k⌋ = j} be the group of query heads sharing KV head j
For any i,i' ∈ G_j:
P(z_head[i] = 1) = P(z_head[i'] = 1)
= sigmoid(mean_{k∈G_j}(log α_k)/β)
```

### C.3 Optimization Convergence Analysis

1. Lagrangian Formulation:
```
L(θ,z,λ) = L_LM(θ,z) + Σ_j λ_j(||z_j||₀ - k_j)
```

2. KKT Conditions:
```
Stationarity: ∇_θL = 0, ∇_zL = 0
Complementary Slackness: λ_j(||z_j||₀ - k_j) = 0
Feasibility: ||z_j||₀ ≤ k_j
```

3. Convergence Rate:
```
For gradient descent with step size η:
||θₜ - θ*||² ≤ (1 - ηL)ᵗ||θ₀ - θ*||²
where L is the Lipschitz constant
```

## D. Detailed GQA Analysis

### D.1 Mathematical Formulation of Grouped Attention

1. Standard Attention:
```
Attention(Q,K,V) = softmax(QK^T/√d)V
Q,K,V ∈ R^{b×h×s×d}
```

2. Grouped Query Attention:
```
Q ∈ R^{b×h×n×d}  # n query heads
K,V ∈ R^{b×h×m×d}  # m KV heads, m < n

For i-th query head:
j = ⌊i/k⌋  # corresponding KV head
out_i = softmax(Q_iK_j^T/√d)V_j
```

3. Memory Efficiency:
```
Standard: O(bhsd + bh(s²))
GQA: O(bhsd + bh(sm))
where s is sequence length
```

### D.2 Impact on Pruning Decisions

1. Group-wise Importance Scoring:
```
I_group(j) = mean_{i∈G_j} I(Q_i) + I(K_j) + I(V_j)
where:
G_j = {i | ⌊i/k⌋ = j}
I(·) is importance score
```

2. Pruning Decision Rule:
```
For each group j:
if I_group(j) < threshold:
    prune all Q_i where i ∈ G_j
    prune K_j, V_j
```

3. Constraint Satisfaction:
```
Σ_j z_group(j) = m'  # target KV heads
Σ_i z_query(i) = n'  # target query heads
where n'/m' = n/m  # maintain ratio
```

### D.3 Group Structure Preservation

1. Mask Generation:
```python
def generate_group_mask(self, temperature=1.0):
    # Generate base mask for KV heads
    kv_mask = self.sample_concrete(self.log_alpha_kv, temperature)
    
    # Expand to query heads
    query_mask = kv_mask.repeat_interleave(self.heads_per_kv, dim=1)
    
    return query_mask
```

2. Loss Adjustment:
```python
def compute_group_loss(self, loss, masks):
    group_penalty = 0
    for j in range(self.num_kv_heads):
        group_j = masks['head'][:, j*self.heads_per_kv:(j+1)*self.heads_per_kv]
        group_penalty += torch.var(group_j, dim=1).mean()
    return loss + self.group_lambda * group_penalty
```

## E. Comparative Analysis

### E.1 Comparison with Other Pruning Methods

1. Unstructured Pruning:
```
Method: Individual weight pruning
Advantages: 
- Fine-grained control
- Higher theoretical compression
Disadvantages:
- Limited hardware acceleration
- Irregular sparsity patterns
```

2. Magnitude-based Pruning:
```
Method: Remove weights based on |w_ij|
Advantages:
- Simple implementation
- No training required
Disadvantages:
- Ignores weight interactions
- Sub-optimal for structured pruning
```

3. Sheared LLaMA (Ours):
```
Method: Structured pruning with GQA awareness
Advantages:
- Hardware-friendly sparsity
- Maintains architectural constraints
- Optimized for modern architectures
Disadvantages:
- Coarser granularity
- Requires careful constraint balancing
```

### E.2 Theoretical Analysis

1. Computational Complexity:
```
Unstructured: O(nd) operations, O(nd) memory
Magnitude: O(nd log(nd)) operations, O(nd) memory
Ours: O(nd + km) operations, O(nd) memory
where:
n = number of neurons
d = dimension
k = number of groups
m = group size
```

2. Optimization Landscape:
```
Loss surface smoothness:
- Unstructured: High variability
- Magnitude: Medium variability
- Ours: Lower variability due to group constraints
```

3. Convergence Guarantees:
```
Under mild conditions (L-smoothness, μ-strong convexity):
Rate = O(log(1/ε))
where ε is the target accuracy
```

## F. Detailed Algorithm Descriptions

### F.1 Step-by-step Pruning Process

```python
Algorithm 2: GQA-aware Structured Pruning

Input: 
- Model θ with n query heads, m KV heads
- Target sparsities k_layer, k_head, k_hidden, k_int
- Training data D
- Learning rate η

Initialize:
- Mask parameters α for each component
- Lagrange multipliers λ
- Temperature schedule τ(t)

for epoch = 1 to N do:
    # Temperature annealing
    τ = τ(epoch)
    
    for batch in D do:
        # 1. Sample masks
        z_layer = sample_concrete(α_layer, τ)
        z_head = sample_group_mask(α_head, τ)
        z_hidden = sample_concrete(α_hidden, τ)
        z_int = sample_concrete(α_int, τ)
        
        # 2. Forward pass with masks
        loss = compute_masked_loss(batch, θ, z)
        
        # 3. Add constraint penalties
        for j in {layer, head, hidden, int}:
            loss += λ_j * (||z_j||₀ - k_j)
        
        # 4. Backward pass
        grads = compute_gradients(loss)
        
        # 5. Update parameters
        θ = θ - η * grads_θ
        α = α - η * grads_α
        
        # 6. Update multipliers
        for j in {layer, head, hidden, int}:
            λ_j += η * (||z_j||₀ - k_j)
        
        # 7. Project to feasible set
        project_constraints(z)

Output: Pruned model θ', binary masks z
```

### F.2 Memory Optimization Techniques

1. Gradient Checkpointing:
```python
def forward_with_checkpointing(self, x):
    def custom_forward(*inputs):
        return self.block(inputs[0])
    
    return checkpoint.checkpoint(
        custom_forward,
        x,
        preserve_rng_state=True
    )
```

2. Mixed Precision Training:
```python
scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. Memory Manager:
```python
class MemoryManager:
    def __init__(self):
        self.handles = []
    
    def pin_memory(self, tensor):
        handle = tensor.pin_memory()
        self.handles.append(handle)
        return handle
    
    def clear(self):
        for handle in self.handles:
            handle.free()
        self.handles.clear()
```

### F.3 Training Stability Considerations

1. Loss Scaling:
```python
def compute_scaled_loss(self, base_loss, masks):
    # Scale factors for different components
    scale_factors = {
        'layer': self.num_layers,
        'head': self.num_heads,
        'hidden': self.hidden_dim,
        'intermediate': self.intermediate_dim
    }
    
    # Normalize penalties
    normalized_loss = base_loss
    for component, factor in scale_factors.items():
        mask = masks[component]
        penalty = self.compute_penalty(mask)
        normalized_loss += penalty / factor
    
    return normalized_loss
```

2. Gradient Clipping:
```python
def clip_gradients(self, model, max_norm):
    # Compute total norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Apply clipping
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

3. Mask Smoothing:
```python
def smooth_masks(self, masks, beta=0.9):
    """Apply exponential moving average to masks"""
    with torch.no_grad():
        for key in masks:
            if not hasattr(self, f'smooth_{key}'):
                self.register_buffer(
                    f'smooth_{key}',
                    masks[key].clone()
                )
            smooth = getattr(self, f'smooth_{key}')
            smooth.mul_(beta).add_(masks[key], alpha=1-beta)
    return {k: getattr(self, f'smooth_{k}') for k in masks}
```

## G. Hardware-Specific Optimizations

### G.1 A100 Architecture Utilization

1. Tensor Core Operations:
```python
# Enable high precision matrix multiplications
torch.set_float32_matmul_precision('high')

# Mixed precision training
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(input_ids, attention_mask, labels)
```

2. Memory Management:
```python
# Memory allocator configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Efficient data transfer
batch = {k: v.to(f"cuda:{local_rank}", non_blocking=True) for k, v in batch.items()}
```

3. Distributed Training:
```python
# DDP with A100 optimizations
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # Performance optimization
    static_graph=True  # Enable static graph optimization
)
```

### G.2 Memory Hierarchy

1. GPU Memory Layout:
```
HBM2e (80GB per GPU):
├── Model Parameters (~16GB)
├── Optimizer States (~4GB)
├── Gradients (~20GB)
├── Activations (~30GB)
└── Cache/Workspace (~10GB)
```

2. Memory Access Patterns:
```python
class MemoryEfficientAttention:
    def forward(self, q, k, v):
        # Chunk-based attention computation
        attention_chunks = []
        chunk_size = self.chunk_size
        
        for i in range(0, q.size(1), chunk_size):
            q_chunk = q[:, i:i+chunk_size]
            chunk_attn = self.compute_attention(q_chunk, k, v)
            attention_chunks.append(chunk_attn)
        
        return torch.cat(attention_chunks, dim=1)
```

### G.3 Training Configuration Analysis

1. Batch Size Optimization:
```python
Effective_Batch_Size = Per_GPU_Batch * Num_GPUs * Gradient_Accumulation
64 = 4 * 2 * 8

Memory_Per_Sample = Model_Size + Activation_Size + Gradient_Size
~40GB = 16GB + 15GB + 9GB
```

2. Sequence Length Impact:
```python
Memory_Complexity = O(sequence_length^2)
Optimal_Length = 128  # Balances context window and memory usage
```

3. Optimizer Settings:
```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1e-4,
    eps=1e-8,
    fused=True,    # Use fused implementation
    betas=(0.9, 0.95)  # Tuned for larger batches
)
```

### G.4 Performance Analysis

1. Throughput Metrics:
```
Training Speed:
- Tokens per second: ~800-1000
- Samples per second: ~6-8
- Time per epoch: ~2-3 hours
```

2. Memory Utilization:
```
GPU Memory Usage:
- Peak: 75GB/80GB (93.75%)
- Steady State: 65GB/80GB (81.25%)
- Gradient Storage: ~20GB
- Activation Cache: ~30GB
```

3. Communication Overhead:
```
Inter-GPU Communication:
- Gradient Synchronization: ~2GB/step
- Parameter Updates: ~1GB/step
- NVLink Bandwidth: ~300GB/s
```

### G.5 Optimization Decisions

1. Choice of Mixed Precision:
```python
# BFloat16 vs Float16 Analysis
BFloat16:
- Range: [-3.39e+38, +3.39e+38]
- Precision: ~3 decimal digits
- Better numerical stability
- Native A100 support

Float16:
- Range: [-65504, +65504]
- Precision: ~3.3 decimal digits
- More memory efficient
- Risk of overflow
```

2. Gradient Accumulation:
```python
# Memory vs. Performance Tradeoff
steps = 8  # Chosen based on:
- Memory constraints
- Effective batch size target
- Update frequency requirements
```

3. Attention Implementation:
```python
class OptimizedAttention(nn.Module):
    def forward(self, q, k, v):
        # Memory-efficient attention with A100 optimizations
        with torch.cuda.amp.autocast():
            # Use flash attention when available
            if FLASH_AVAILABLE:
                return flash_attention(q, k, v)
            # Fallback to chunked attention
            return chunked_attention(q, k, v)
```

This appendix provides detailed technical insights into the mathematical foundations, algorithmic implementations, and hardware-specific optimizations used in our approach. The configurations and optimizations were carefully chosen to maximize the utilization of A100 GPUs while maintaining training stability and efficiency.

"""Sheared LLaMA pruning implementation optimized for LLaMA 3.1 8B"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
import gc
import torch.distributed as dist

class ShearedLLaMAPruner:
    def __init__(
        self,
        model: nn.Module,
        target_layer_sparsity: float = 0.2,
        target_head_sparsity: float = 0.3,
        target_hidden_sparsity: float = 0.1,
        target_intermediate_sparsity: float = 0.4,
        temperature: float = 1.0,
        lambda_init: float = 0.1,
        chunk_size: int = 8  # Reduced chunk size for better memory management
    ):
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
        
        # Initialize masks
        self.masks = {}
        self._initialize_masks()
        
        # Initialize optimizer
        self.optimizer = self._initialize_optimizer(
            target_layer_sparsity,
            target_head_sparsity,
            target_hidden_sparsity,
            target_intermediate_sparsity,
            lambda_init
        )
        
        # Get available devices
        self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        self.current_device_idx = 0
        
        # Set memory allocator settings
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory for cuda operations
    
    def _clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection"""
        for device in self.devices:
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    def _get_next_device(self):
        """Get the next available GPU device in round-robin fashion"""
        device = self.devices[self.current_device_idx]
        self.current_device_idx = (self.current_device_idx + 1) % len(self.devices)
        return device
    
    def _initialize_masks(self):
        """Initialize pruning masks for LLaMA 3.1 8B architecture"""
        # Get model configuration
        num_layers = len(self.model.model.layers)
        hidden_dim = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        intermediate_dim = self.model.config.intermediate_size  # 14336 for LLaMA 3.1
        
        # Initialize masks with appropriate sizes
        self.masks['layer'] = LayerMask(num_layers)
        self.masks['head'] = HeadMask(num_heads, num_layers, num_kv_heads)
        self.masks['hidden'] = HiddenMask(hidden_dim)
        self.masks['intermediate'] = IntermediateMask(intermediate_dim, num_layers)
    
    def _initialize_optimizer(
        self,
        target_layer_sparsity: float,
        target_head_sparsity: float,
        target_hidden_sparsity: float,
        target_intermediate_sparsity: float,
        lambda_init: float
    ):
        """Initialize Lagrangian optimizer"""
        return LagrangianOptimizer(
            target_layer_sparsity,
            target_head_sparsity,
            target_hidden_sparsity,
            target_intermediate_sparsity,
            lambda_init
        )
    
    def compute_loss(self, base_loss: torch.Tensor) -> torch.Tensor:
        """Compute pruning-aware loss"""
        return self.optimizer.compute_total_loss(base_loss, self.masks)
    
    def update_lagrange_multipliers(self):
        """Update Lagrange multipliers"""
        self.optimizer.update_multipliers(self.get_pruning_mask())
    
    def get_pruning_mask(self) -> Dict[str, torch.Tensor]:
        """Get binary pruning masks"""
        return {
            name: mask.get_mask(self.temperature)
            for name, mask in self.masks.items()
        }
    
    def _safe_to_device(self, tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Safely move a tensor to a device in smaller chunks with memory management"""
        if tensor.device == target_device:
            return tensor
        
        try:
            # Try direct transfer first
            return tensor.to(target_device)
        except RuntimeError:  # Out of memory
            self._clear_gpu_memory()
            
            # Process in smaller chunks
            chunks = []
            total_size = tensor.size(0)
            
            for i in range(0, total_size, self.chunk_size):
                # Clear memory before each chunk
                self._clear_gpu_memory()
                
                end_idx = min(i + self.chunk_size, total_size)
                chunk = tensor[i:end_idx].to(target_device)
                chunks.append(chunk)
            
            # Concatenate chunks with memory management
            try:
                result = torch.cat(chunks, dim=0)
            except RuntimeError:
                # If concatenation fails, try with smaller chunks
                self._clear_gpu_memory()
                smaller_chunks = []
                for chunk in chunks:
                    smaller_chunks.extend(torch.chunk(chunk, 2))
                result = torch.cat(smaller_chunks, dim=0)
            
            # Clean up
            del chunks
            self._clear_gpu_memory()
            
            return result
    
    def _safe_index_select(self, tensor: torch.Tensor, dim: int, indices: torch.Tensor) -> torch.Tensor:
        """Safely perform index_select operation with memory management"""
        target_device = self._get_next_device()
        
        try:
            # Try direct operation first
            return tensor.index_select(dim, indices.to(tensor.device))
        except RuntimeError:  # Out of memory
            self._clear_gpu_memory()
            
            # Move to CPU for processing
            cpu_tensor = tensor.cpu()
            cpu_indices = indices.cpu()
            
            # Free GPU memory
            del tensor
            self._clear_gpu_memory()
            
            # Process in chunks
            chunks = []
            if dim == -1 or dim == cpu_tensor.dim() - 1:
                # Last dimension: split first dimension
                total_size = cpu_tensor.size(0)
                for start_idx in range(0, total_size, self.chunk_size):
                    self._clear_gpu_memory()
                    
                    end_idx = min(start_idx + self.chunk_size, total_size)
                    chunk = cpu_tensor[start_idx:end_idx].index_select(dim, cpu_indices)
                    chunks.append(chunk.cpu())  # Keep on CPU temporarily
            else:
                # First dimension: split indices
                total_indices = cpu_indices.size(0)
                for start_idx in range(0, total_indices, self.chunk_size):
                    self._clear_gpu_memory()
                    
                    end_idx = min(start_idx + self.chunk_size, total_indices)
                    chunk_indices = cpu_indices[start_idx:end_idx]
                    chunk = cpu_tensor.index_select(dim, chunk_indices)
                    chunks.append(chunk.cpu())  # Keep on CPU temporarily
            
            # Concatenate on CPU
            try:
                result = torch.cat(chunks, dim=0)
            except RuntimeError:
                # If concatenation fails, try with smaller chunks
                self._clear_gpu_memory()
                smaller_chunks = []
                for chunk in chunks:
                    smaller_chunks.extend(torch.chunk(chunk, 2))
                result = torch.cat(smaller_chunks, dim=0)
            
            # Clean up
            del cpu_tensor, cpu_indices, chunks
            self._clear_gpu_memory()
            
            # Move back to target device safely
            return self._safe_to_device(result, target_device)
    
    def _synchronize_pruning(self):
        """Synchronize pruning across GPUs"""
        if dist.is_initialized():
            dist.barrier()
            self._clear_gpu_memory()
    
    def apply_pruning(self):
        """Apply final pruning to model with improved memory management"""
        binary_masks = self.get_pruning_mask()
        
        # Move masks to CPU initially
        binary_masks = {k: v.cpu() for k, v in binary_masks.items()}
        self._clear_gpu_memory()
        
        try:
            # Apply pruning in stages with synchronization
            self._apply_layer_pruning(binary_masks)
            self._synchronize_pruning()
            
            self._apply_head_pruning(binary_masks)
            self._synchronize_pruning()
            
            self._apply_hidden_pruning(binary_masks)
            self._synchronize_pruning()
            
            self._apply_intermediate_pruning(binary_masks)
            self._synchronize_pruning()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM occurs, try with smaller chunk size
                original_chunk_size = self.chunk_size
                self.chunk_size = max(1, self.chunk_size // 2)
                print(f"OOM occurred. Retrying with chunk_size={self.chunk_size}")
                
                # Clear memory and retry
                self._clear_gpu_memory()
                self.apply_pruning()
                
                # Restore original chunk size
                self.chunk_size = original_chunk_size
            else:
                raise e
    
    def _apply_layer_pruning(self, binary_masks: Dict[str, torch.Tensor]):
        """Apply layer-level pruning"""
        layer_mask = binary_masks['layer']
        pruned_layers = []
        
        for layer_idx, keep_layer in enumerate(layer_mask):
            if not keep_layer:
                pruned_layers.append(layer_idx)
        
        # Remove pruned layers
        layers = list(self.model.model.layers)
        for idx in reversed(pruned_layers):
            del layers[idx]
        
        self.model.model.layers = nn.ModuleList(layers)
        self._clear_gpu_memory()
    
    def _apply_head_pruning(self, binary_masks: Dict[str, torch.Tensor]):
        """Apply attention head pruning with GQA support"""
        head_mask = binary_masks['head']
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', self.model.config.num_attention_heads)
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            self._clear_gpu_memory()
            
            # Get masks for this layer
            target_device = self._get_next_device()
            layer_head_mask = head_mask[layer_idx].to(target_device)
            
            # Handle GQA structure
            heads_per_kv = self.model.config.num_attention_heads // num_kv_heads
            kv_head_mask = layer_head_mask.view(-1, heads_per_kv).any(dim=1)
            
            # Prune query heads
            q_weight = layer.self_attn.q_proj.weight.to(target_device)
            q_bias = layer.self_attn.q_proj.bias.to(target_device) if layer.self_attn.q_proj.bias is not None else None
            
            # Reshape for grouped heads
            head_dim = q_weight.size(0) // self.model.config.num_attention_heads
            q_weight = q_weight.view(self.model.config.num_attention_heads, head_dim, -1)
            
            # Apply mask
            q_weight = q_weight[layer_head_mask.bool()]
            if q_bias is not None:
                q_bias = q_bias.view(self.model.config.num_attention_heads, head_dim)
                q_bias = q_bias[layer_head_mask.bool()].flatten()
            
            # Update weights
            layer.self_attn.q_proj.weight = nn.Parameter(q_weight.reshape(-1, q_weight.size(-1)))
            if q_bias is not None:
                layer.self_attn.q_proj.bias = nn.Parameter(q_bias)
            
            # Prune key and value heads according to GQA structure
            for proj in ['k_proj', 'v_proj']:
                weight = getattr(layer.self_attn, proj).weight.to(target_device)
                bias = getattr(layer.self_attn, proj).bias.to(target_device) if getattr(layer.self_attn, proj).bias is not None else None
                
                head_dim = weight.size(0) // num_kv_heads
                weight = weight.view(num_kv_heads, head_dim, -1)
                
                weight = weight[kv_head_mask.bool()]
                if bias is not None:
                    bias = bias.view(num_kv_heads, head_dim)
                    bias = bias[kv_head_mask.bool()].flatten()
                
                setattr(layer.self_attn, proj, nn.Linear(
                    weight.size(-1),
                    weight.size(0) * weight.size(1),
                    bias=bias is not None
                ))
                getattr(layer.self_attn, proj).weight = nn.Parameter(weight.reshape(-1, weight.size(-1)))
                if bias is not None:
                    getattr(layer.self_attn, proj).bias = nn.Parameter(bias)
            
            self._clear_gpu_memory()
    
    def _apply_hidden_pruning(self, binary_masks: Dict[str, torch.Tensor]):
        """Apply hidden dimension pruning with improved memory management"""
        hidden_mask = binary_masks['hidden']
        keep_indices = hidden_mask.bool().nonzero().squeeze(1)
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            self._clear_gpu_memory()
            
            # Get the target device for this layer
            target_device = self._get_next_device()
            layer_keep_indices = keep_indices.to(target_device)
            
            # Process each projection with memory cleanup
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                self._clear_gpu_memory()
                
                proj = getattr(layer.self_attn, name)
                if name == 'o_proj':
                    new_weight = self._safe_index_select(proj.weight, 1, layer_keep_indices)
                else:
                    new_weight = self._safe_index_select(proj.weight, -1, layer_keep_indices)
                
                proj.weight = nn.Parameter(new_weight)
                del new_weight
                self._clear_gpu_memory()
            
            # Process MLP layers with memory cleanup
            for proj_name in ['gate_proj', 'up_proj']:
                self._clear_gpu_memory()
                
                proj = getattr(layer.mlp, proj_name)
                new_weight = self._safe_index_select(proj.weight, -1, layer_keep_indices)
                proj.weight = nn.Parameter(new_weight)
                del new_weight
            
            # Down projection
            self._clear_gpu_memory()
            new_down_weight = self._safe_index_select(layer.mlp.down_proj.weight, 1, layer_keep_indices)
            layer.mlp.down_proj.weight = nn.Parameter(new_down_weight)
            del new_down_weight
            
            self._clear_gpu_memory()
            
            # Synchronize after each layer
            self._synchronize_pruning()
    
    def _apply_intermediate_pruning(self, binary_masks: Dict[str, torch.Tensor]):
        """Apply intermediate dimension pruning with improved memory management"""
        int_mask = binary_masks['intermediate']
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            self._clear_gpu_memory()
            
            layer_int_mask = int_mask[layer_idx]
            keep_indices = layer_int_mask.bool().nonzero().squeeze(1)
            
            # Get the target device for this layer
            target_device = self._get_next_device()
            layer_keep_indices = keep_indices.to(target_device)
            
            # Process gate and up projections with memory cleanup
            for proj_name in ['gate_proj', 'up_proj']:
                self._clear_gpu_memory()
                
                proj = getattr(layer.mlp, proj_name)
                new_weight = self._safe_index_select(proj.weight, 0, layer_keep_indices)
                proj.weight = nn.Parameter(new_weight)
                
                if proj.bias is not None:
                    new_bias = proj.bias.index_select(0, layer_keep_indices)
                    proj.bias = nn.Parameter(new_bias)
                    del new_bias
                
                del new_weight
            
            # Process down projection
            self._clear_gpu_memory()
            new_down_weight = self._safe_index_select(layer.mlp.down_proj.weight, 1, layer_keep_indices)
            layer.mlp.down_proj.weight = nn.Parameter(new_down_weight)
            del new_down_weight
            
            self._clear_gpu_memory()
            
            # Synchronize after each layer
            self._synchronize_pruning()

class StructuredPruningMask(nn.Module):
    """Base class for all structured pruning masks"""
    def __init__(self, size, init_value=0.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(size, init_value))
        
    def forward(self, temperature=1.0):
        return self.get_mask(temperature)
    
    def get_mask(self, temperature=1.0):
        u = torch.rand_like(self.log_alpha)
        noise = torch.log(u) - torch.log(1 - u)
        s = torch.sigmoid((self.log_alpha + noise) / temperature)
        return torch.clamp(s, 0, 1)

class LayerMask(StructuredPruningMask):
    """Layer-level pruning mask"""
    def __init__(self, num_layers, init_value=0.0):
        super().__init__((num_layers,), init_value)

class HeadMask(StructuredPruningMask):
    """Head-level pruning mask with GQA support"""
    def __init__(self, num_heads, num_layers, num_kv_heads=None):
        super().__init__((num_layers, num_heads))
        self.num_kv_heads = num_kv_heads
    
    def forward(self, temperature=1.0):
        mask = super().forward(temperature)
        if self.num_kv_heads is not None:
            # Handle GQA structure
            heads_per_kv = mask.size(1) // self.num_kv_heads
            mask = mask.view(mask.size(0), self.num_kv_heads, heads_per_kv)
            mask = mask.mean(dim=-1, keepdim=True).expand(-1, -1, heads_per_kv)
            mask = mask.reshape(mask.size(0), -1)
        return mask

class HiddenMask(StructuredPruningMask):
    """Hidden dimension pruning mask"""
    def __init__(self, hidden_dim, init_value=0.0):
        super().__init__((hidden_dim,), init_value)

class IntermediateMask(StructuredPruningMask):
    """Intermediate dimension pruning mask"""
    def __init__(self, intermediate_dim, num_layers):
        super().__init__((num_layers, intermediate_dim))

class LagrangianOptimizer:
    """Lagrangian optimization for constrained pruning"""
    def __init__(
        self,
        target_layer_sparsity: float,
        target_head_sparsity: float,
        target_hidden_sparsity: float,
        target_intermediate_sparsity: float,
        lambda_init: float = 0.1
    ):
        self.lambda_layer = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_head = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_hidden = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_int = nn.Parameter(torch.tensor(lambda_init))
        
        self.target_sparsities = {
            'layer': target_layer_sparsity,
            'head': target_head_sparsity,
            'hidden': target_hidden_sparsity,
            'intermediate': target_intermediate_sparsity
        }
    
    def compute_total_loss(self, base_loss: torch.Tensor, masks: Dict[str, StructuredPruningMask]) -> torch.Tensor:
        """Compute total loss with Lagrangian penalties"""
        total_loss = base_loss
        
        # Add constraint penalties
        total_loss += self.compute_penalty(
            masks['layer'].get_mask(),
            self.lambda_layer,
            self.target_sparsities['layer']
        )
        total_loss += self.compute_penalty(
            masks['head'].get_mask(),
            self.lambda_head,
            self.target_sparsities['head']
        )
        total_loss += self.compute_penalty(
            masks['hidden'].get_mask(),
            self.lambda_hidden,
            self.target_sparsities['hidden']
        )
        total_loss += self.compute_penalty(
            masks['intermediate'].get_mask(),
            self.lambda_int,
            self.target_sparsities['intermediate']
        )
        
        return total_loss
    
    def compute_penalty(self, mask: torch.Tensor, lambda_param: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """Compute Lagrangian penalty term"""
        current_sparsity = 1 - mask.float().mean()
        penalty = lambda_param * (current_sparsity - target_sparsity)
        return penalty + 0.5 * (current_sparsity - target_sparsity)**2
    
    def update_multipliers(self, binary_masks: Dict[str, torch.Tensor]):
        """Update Lagrange multipliers"""
        with torch.no_grad():
            # Update each multiplier based on constraint violation
            for name, mask in binary_masks.items():
                current_sparsity = 1 - mask.float().mean()
                target_sparsity = self.target_sparsities[name]
                violation = current_sparsity - target_sparsity
                
                if name == 'layer':
                    self.lambda_layer.add_(violation)
                elif name == 'head':
                    self.lambda_head.add_(violation)
                elif name == 'hidden':
                    self.lambda_hidden.add_(violation)
                elif name == 'intermediate':
                    self.lambda_int.add_(violation)

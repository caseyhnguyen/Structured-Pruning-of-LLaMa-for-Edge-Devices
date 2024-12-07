import torch
import torch.nn as nn
import torch.nn.functional as F

class HardConcreteDistribution:
    """Implementation of Hard Concrete Distribution for binary gates"""
    def __init__(self, beta=1/3):
        self.beta = beta
        
    def sample(self, log_alpha, temperature=1.0):
        # Sample from Hard Concrete distribution
        u = torch.rand_like(log_alpha)
        noise = torch.log(u) - torch.log(1 - u)
        s = torch.sigmoid((log_alpha + noise) / temperature)
        s_bar = s * (1 - 0) + 0  # stretching
        mask = torch.clamp(s_bar, 0, 1)
        return mask

class StructuredPruningMask(nn.Module):
    """Base class for all structured pruning masks"""
    def __init__(self, size, init_value=0.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(size, init_value))
        self.concrete = HardConcreteDistribution()
        
    def forward(self, temperature=1.0):
        return self.concrete.sample(self.log_alpha, temperature)

class LayerMask(StructuredPruningMask):
    """Mask for layer-level pruning"""
    def __init__(self, num_layers, init_value=0.0):
        super().__init__((num_layers,), init_value)
        
    def get_expected_sparsity(self):
        """Get expected number of pruned layers"""
        probs = torch.sigmoid(self.log_alpha)
        return (1 - probs).mean()

class HeadMask(StructuredPruningMask):
    """Mask for attention head pruning with GQA support"""
    def __init__(self, num_heads, num_layers, num_kv_heads=None, init_value=0.0):
        super().__init__((num_layers, num_heads), init_value)
        self.num_kv_heads = num_kv_heads  # For GQA support
        
    def forward(self, temperature=1.0):
        mask = super().forward(temperature)
        if self.num_kv_heads is not None:
            # Ensure GQA structure is preserved
            # Each group of query heads shares the same key/value head
            heads_per_kv = mask.size(1) // self.num_kv_heads
            mask = mask.view(mask.size(0), self.num_kv_heads, heads_per_kv)
            # Use same mask for all query heads in a group
            mask = mask.mean(dim=-1, keepdim=True).expand(-1, -1, heads_per_kv)
            mask = mask.reshape(mask.size(0), -1)
        return mask
    
    def get_expected_sparsity(self):
        """Get expected number of pruned heads"""
        probs = torch.sigmoid(self.log_alpha)
        return (1 - probs).mean()

class HiddenMask(StructuredPruningMask):
    """Mask for hidden dimension pruning"""
    def __init__(self, hidden_dim, init_value=0.0):
        super().__init__((hidden_dim,), init_value)
        
    def get_expected_sparsity(self):
        """Get expected number of pruned hidden dimensions"""
        probs = torch.sigmoid(self.log_alpha)
        return (1 - probs).mean()

class IntermediateMask(StructuredPruningMask):
    """Mask for intermediate (FFN) dimension pruning"""
    def __init__(self, intermediate_dim, num_layers, init_value=0.0):
        super().__init__((num_layers, intermediate_dim), init_value)
        
    def get_expected_sparsity(self):
        """Get expected number of pruned intermediate dimensions"""
        probs = torch.sigmoid(self.log_alpha)
        return (1 - probs).mean()


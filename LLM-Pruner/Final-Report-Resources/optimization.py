import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedOptimization:
    """Handles constrained optimization for structured pruning"""
    def __init__(self, 
                 target_layer_sparsity=0.0,
                 target_head_sparsity=0.0, 
                 target_hidden_sparsity=0.0,
                 target_intermediate_sparsity=0.0,
                 lambda_init=1.0):
        # Initialize Lagrange multipliers
        self.lambda_layer = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_head = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_hidden = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_int = nn.Parameter(torch.tensor(lambda_init))
        
        # Target sparsity ratios
        self.target_layer = target_layer_sparsity
        self.target_head = target_head_sparsity
        self.target_hidden = target_hidden_sparsity
        self.target_int = target_intermediate_sparsity
        
    def compute_layer_penalty(self, layer_mask):
        """Compute penalty for layer pruning constraint"""
        expected_sparsity = layer_mask.get_expected_sparsity()
        penalty = self.lambda_layer * (expected_sparsity - self.target_layer)
        return F.relu(penalty)  # Only penalize when constraint is violated
        
    def compute_head_penalty(self, head_mask):
        """Compute penalty for attention head pruning constraint"""
        expected_sparsity = head_mask.get_expected_sparsity()
        penalty = self.lambda_head * (expected_sparsity - self.target_head)
        return F.relu(penalty)
        
    def compute_hidden_penalty(self, hidden_mask):
        """Compute penalty for hidden dimension pruning constraint"""
        expected_sparsity = hidden_mask.get_expected_sparsity()
        penalty = self.lambda_hidden * (expected_sparsity - self.target_hidden)
        return F.relu(penalty)
        
    def compute_intermediate_penalty(self, intermediate_mask):
        """Compute penalty for intermediate dimension pruning constraint"""
        expected_sparsity = intermediate_mask.get_expected_sparsity()
        penalty = self.lambda_int * (expected_sparsity - self.target_int)
        return F.relu(penalty)
    
    def compute_total_loss(self, lm_loss, masks, temperature=1.0):
        """Compute total loss with all constraints
        
        Args:
            lm_loss: Base language modeling loss
            masks: Dictionary containing all mask objects
            temperature: Temperature for concrete distribution sampling
        """
        total_loss = lm_loss
        
        # Add penalties for each constraint
        if masks.get('layer') is not None:
            total_loss = total_loss + self.compute_layer_penalty(masks['layer'])
            
        if masks.get('head') is not None:
            total_loss = total_loss + self.compute_head_penalty(masks['head'])
            
        if masks.get('hidden') is not None:
            total_loss = total_loss + self.compute_hidden_penalty(masks['hidden'])
            
        if masks.get('intermediate') is not None:
            total_loss = total_loss + self.compute_intermediate_penalty(masks['intermediate'])
            
        return total_loss
    
    def update_lagrange_multipliers(self, masks, lr=0.01):
        """Update Lagrange multipliers using gradient descent
        
        Args:
            masks: Dictionary containing all mask objects
            lr: Learning rate for multiplier updates
        """
        with torch.no_grad():
            # Update each multiplier based on constraint violation
            if masks.get('layer') is not None:
                violation = masks['layer'].get_expected_sparsity() - self.target_layer
                self.lambda_layer.add_(lr * violation)
                
            if masks.get('head') is not None:
                violation = masks['head'].get_expected_sparsity() - self.target_head
                self.lambda_head.add_(lr * violation)
                
            if masks.get('hidden') is not None:
                violation = masks['hidden'].get_expected_sparsity() - self.target_hidden
                self.lambda_hidden.add_(lr * violation)
                
            if masks.get('intermediate') is not None:
                violation = masks['intermediate'].get_expected_sparsity() - self.target_int
                self.lambda_int.add_(lr * violation)

import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np

import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from ptflops import get_model_complexity_info
from ptflops.pytorch_ops import bn_flops_counter_hook, pool_flops_counter_hook

from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.peft import PeftModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def LlamaAttention_counter_hook(module, input, output):
    flops = 0
    q_len = output[0].shape[1]
    linear_dim = output[0].shape[-1]
    num_heads = module.num_heads
    head_dim = module.head_dim

    rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
    attention_flops = num_heads * (q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim)
    linear_flops = 4 * (q_len * linear_dim * num_heads * head_dim)
    flops += rotary_flops + attention_flops + linear_flops
    module.__flops__ += int(flops)

def rmsnorm_flops_counter_hook(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    batch_flops *= 2
    module.__flops__ += int(batch_flops)

def silu_flops_counter_hook(module, input, output):
    input = input[0]
    flops = np.prod(input.shape) * 2
    module.__flops__ += int(flops)

class SiLU(torch.nn.Module):
    def forward(self, x):
        return F.silu(x)

def get_module_metrics(model):
    """Extract detailed metrics for each module"""
    metrics = {}
    
    def process_module(module, prefix=""):
        if hasattr(module, "weight"):
            params = module.weight.numel()
            if hasattr(module, "bias") and module.bias is not None:
                params += module.bias.numel()
            metrics[prefix] = {
                "parameters": params,
                "type": module.__class__.__name__
            }
            if hasattr(module, "__flops__"):
                metrics[prefix]["flops"] = module.__flops__
        
        for name, child in module.named_children():
            child_prefix = f"{prefix}/{name}" if prefix else name
            process_module(child, child_prefix)
    
    process_module(model)
    return metrics

def main(args):
    if args.model_type == 'pretrain':
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    else:
        raise NotImplementedError

    def input_constructor(x):
        return {'input_ids': torch.ones(x).long().to(device)}

    if device == "cuda":
        model.half()
        model = model.cuda()
    
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
                                                    input_constructor=input_constructor,
                                                    print_per_layer_stat=True, verbose=True,
                                                    custom_modules_hooks={
                                                        LlamaAttention: LlamaAttention_counter_hook,
                                                        LlamaRMSNorm: rmsnorm_flops_counter_hook,
                                                        SiLU: silu_flops_counter_hook,
                                                    },)
    else:
        model.float()
        macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
                                                    input_constructor=input_constructor,
                                                    print_per_layer_stat=True, verbose=True,
                                                    custom_modules_hooks={
                                                        LlamaAttention: LlamaAttention_counter_hook,
                                                        LlamaRMSNorm: rmsnorm_flops_counter_hook,
                                                        SiLU: silu_flops_counter_hook,
                                                    },)

    # Get detailed metrics
    detailed_metrics = get_module_metrics(model)
    
    # Prepare metrics dictionary
    metrics = {
        "model_type": args.model_type,
        "computational_complexity": macs,
        "parameters": params,
        "gpu_memory_mib": torch.cuda.memory_allocated()/1024/1024 if device == "cuda" else 0,
        "model_architecture": {
            "num_layers": len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        },
        "detailed_metrics": detailed_metrics
    }
    
    # Print summary
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("GPU Memory Requirement: {} MiB\n".format(metrics["gpu_memory_mib"]))
    
    # Save metrics if path provided
    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
        with open(args.save_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.save_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--model_type', type=str, required=True, help = 'choose from [pretrain, pruneLLM]')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--save_metrics', type=str, default=None, help='path to save metrics json file')
    
    args = parser.parse_args()
    main(args)

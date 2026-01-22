"""
Export TinyPFN weights to C header for ESP32 deployment.
This creates a header file with all model weights as C arrays.

Usage:
    python export_weights_to_c.py --checkpoint tinypfn_prior_trained.pt
"""

import argparse
import torch
import numpy as np


def export_weights(checkpoint_path: str, output_path: str):
    """Export PyTorch weights to C header file."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint['model_kwargs']
    
    print(f"Model config: {config}")
    print(f"Total parameters: {sum(p.numel() for p in state_dict.values())}")
    
    # Start building the header file
    lines = []
    lines.append("// TinyPFN Weights for ESP32")
    lines.append("// Auto-generated - do not edit")
    lines.append(f"// Config: embedding={config['embedding_size']}, heads={config['num_attention_heads']}, "
                 f"mlp={config['mlp_hidden_size']}, layers={config['num_layers']}, outputs={config['num_outputs']}")
    lines.append("")
    lines.append("#ifndef TINYPFN_WEIGHTS_H")
    lines.append("#define TINYPFN_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    
    # Model configuration
    lines.append("// Model configuration")
    lines.append(f"#define EMBEDDING_SIZE {config['embedding_size']}")
    lines.append(f"#define NUM_HEADS {config['num_attention_heads']}")
    lines.append(f"#define HEAD_DIM (EMBEDDING_SIZE / NUM_HEADS)")
    lines.append(f"#define MLP_HIDDEN {config['mlp_hidden_size']}")
    lines.append(f"#define NUM_LAYERS {config['num_layers']}")
    lines.append(f"#define NUM_OUTPUTS {config['num_outputs']}")
    lines.append("")
    
    def format_array(name: str, tensor: torch.Tensor, comment: str = ""):
        """Format a tensor as a C array."""
        arr = tensor.detach().numpy().flatten()
        shape_str = "x".join(map(str, tensor.shape))
        
        result = []
        if comment:
            result.append(f"// {comment}")
        result.append(f"// Shape: [{shape_str}], Total: {arr.size}")
        result.append(f"const float {name}[{arr.size}] = {{")
        
        # Format values in rows of 8
        for i in range(0, len(arr), 8):
            chunk = arr[i:i+8]
            values = ", ".join(f"{v:.8f}f" for v in chunk)
            if i + 8 < len(arr):
                values += ","
            result.append(f"    {values}")
        
        result.append("};")
        result.append("")
        return "\n".join(result)
    
    # Export all weights
    lines.append("// ============================================")
    lines.append("// Feature Encoder")
    lines.append("// ============================================")
    lines.append(format_array("feature_enc_weight", 
                              state_dict['feature_encoder.linear_layer.weight'],
                              "Linear(1, embedding_size)"))
    lines.append(format_array("feature_enc_bias",
                              state_dict['feature_encoder.linear_layer.bias']))
    
    lines.append("// ============================================")
    lines.append("// Target Encoder")
    lines.append("// ============================================")
    lines.append(format_array("target_enc_weight",
                              state_dict['target_encoder.linear_layer.weight'],
                              "Linear(1, embedding_size)"))
    lines.append(format_array("target_enc_bias",
                              state_dict['target_encoder.linear_layer.bias']))
    
    # Transformer blocks
    for layer in range(config['num_layers']):
        lines.append(f"// ============================================")
        lines.append(f"// Transformer Block {layer}")
        lines.append(f"// ============================================")
        
        prefix = f"transformer_blocks.{layer}"
        
        # Feature attention
        lines.append(f"// Feature Attention (between features)")
        lines.append(format_array(f"layer{layer}_feat_q_weight",
                                  state_dict[f'{prefix}.self_attention_between_features.q_proj.weight']))
        lines.append(format_array(f"layer{layer}_feat_q_bias",
                                  state_dict[f'{prefix}.self_attention_between_features.q_proj.bias']))
        lines.append(format_array(f"layer{layer}_feat_k_weight",
                                  state_dict[f'{prefix}.self_attention_between_features.k_proj.weight']))
        lines.append(format_array(f"layer{layer}_feat_k_bias",
                                  state_dict[f'{prefix}.self_attention_between_features.k_proj.bias']))
        lines.append(format_array(f"layer{layer}_feat_v_weight",
                                  state_dict[f'{prefix}.self_attention_between_features.v_proj.weight']))
        lines.append(format_array(f"layer{layer}_feat_v_bias",
                                  state_dict[f'{prefix}.self_attention_between_features.v_proj.bias']))
        lines.append(format_array(f"layer{layer}_feat_out_weight",
                                  state_dict[f'{prefix}.self_attention_between_features.out_proj.weight']))
        lines.append(format_array(f"layer{layer}_feat_out_bias",
                                  state_dict[f'{prefix}.self_attention_between_features.out_proj.bias']))
        
        # Datapoint attention
        lines.append(f"// Datapoint Attention (between datapoints)")
        lines.append(format_array(f"layer{layer}_dp_q_weight",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.q_proj.weight']))
        lines.append(format_array(f"layer{layer}_dp_q_bias",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.q_proj.bias']))
        lines.append(format_array(f"layer{layer}_dp_k_weight",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.k_proj.weight']))
        lines.append(format_array(f"layer{layer}_dp_k_bias",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.k_proj.bias']))
        lines.append(format_array(f"layer{layer}_dp_v_weight",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.v_proj.weight']))
        lines.append(format_array(f"layer{layer}_dp_v_bias",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.v_proj.bias']))
        lines.append(format_array(f"layer{layer}_dp_out_weight",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.out_proj.weight']))
        lines.append(format_array(f"layer{layer}_dp_out_bias",
                                  state_dict[f'{prefix}.self_attention_between_datapoints.out_proj.bias']))
        
        # Layer norms
        lines.append(f"// Layer Norms")
        lines.append(format_array(f"layer{layer}_norm1_weight",
                                  state_dict[f'{prefix}.norm1.weight']))
        lines.append(format_array(f"layer{layer}_norm1_bias",
                                  state_dict[f'{prefix}.norm1.bias']))
        lines.append(format_array(f"layer{layer}_norm2_weight",
                                  state_dict[f'{prefix}.norm2.weight']))
        lines.append(format_array(f"layer{layer}_norm2_bias",
                                  state_dict[f'{prefix}.norm2.bias']))
        lines.append(format_array(f"layer{layer}_norm3_weight",
                                  state_dict[f'{prefix}.norm3.weight']))
        lines.append(format_array(f"layer{layer}_norm3_bias",
                                  state_dict[f'{prefix}.norm3.bias']))
        
        # MLP
        lines.append(f"// MLP")
        lines.append(format_array(f"layer{layer}_mlp1_weight",
                                  state_dict[f'{prefix}.linear1.weight']))
        lines.append(format_array(f"layer{layer}_mlp1_bias",
                                  state_dict[f'{prefix}.linear1.bias']))
        lines.append(format_array(f"layer{layer}_mlp2_weight",
                                  state_dict[f'{prefix}.linear2.weight']))
        lines.append(format_array(f"layer{layer}_mlp2_bias",
                                  state_dict[f'{prefix}.linear2.bias']))
    
    # Decoder
    lines.append("// ============================================")
    lines.append("// Decoder")
    lines.append("// ============================================")
    lines.append(format_array("decoder_linear1_weight",
                              state_dict['decoder.linear1.weight']))
    lines.append(format_array("decoder_linear1_bias",
                              state_dict['decoder.linear1.bias']))
    lines.append(format_array("decoder_linear2_weight",
                              state_dict['decoder.linear2.weight']))
    lines.append(format_array("decoder_linear2_bias",
                              state_dict['decoder.linear2.bias']))
    
    lines.append("#endif // TINYPFN_WEIGHTS_H")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\nWeights exported to: {output_path}")
    
    # Print summary
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params}")
    print(f"Memory for weights: {total_params * 4} bytes ({total_params * 4 / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Export TinyPFN weights to C header')
    parser.add_argument('--checkpoint', type=str, default='tinypfn_prior_trained.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='tinypfn_weights.h',
                        help='Output header file path')
    args = parser.parse_args()
    
    export_weights(args.checkpoint, args.output)


if __name__ == '__main__':
    main()

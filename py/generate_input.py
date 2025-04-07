#!/usr/bin/env python3
import json
import numpy as np
import os
import argparse

def generate_random_input(config_path, output_path):
    """Generate random input features based on model configuration."""
    
    # Read the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get input dimension from config (3 * n_mfcc for base + first and second derivatives)
    n_mfcc = config.get('n_mfcc', 23)  # Default to 23 if not specified
    input_dim = 3 * n_mfcc  # Base MFCC + first derivative + second derivative
    
    # Generate random features (using normal distribution for MFCC-like features)
    # Shape the features to show the structure: (3, n_mfcc) for visualization
    features = np.random.normal(0, 1, size=(3, n_mfcc)).astype(np.float32)
    
    # Normalize features to reasonable range (-1 to 1)
    # This matches typical MFCC value ranges
    features = np.clip(features, -1, 1)
    
    # Flatten the features for saving (C code expects a flat array)
    features_flat = features.reshape(-1)
    
    # Save as raw binary file
    features_flat.tofile(output_path)
    
    print(f"Generated random input with {input_dim} features ({n_mfcc} MFCC coefficients x 3)")
    print(f"Feature structure:")
    print(f"  Base MFCC features:      {features[0,:5]} ...")
    print(f"  First derivatives:       {features[1,:5]} ...")
    print(f"  Second derivatives:      {features[2,:5]} ...")
    print(f"Saved to: {output_path}")
    print(f"Available classes: {', '.join(config['label_names'])}")

def main():
    parser = argparse.ArgumentParser(description='Generate random input for KWS Mamba model')
    parser.add_argument('--config', type=str, default='../my_kws_model/config.json',
                      help='Path to config.json file')
    parser.add_argument('--output', type=str, default='input.raw',
                      help='Path to output raw file')
    
    args = parser.parse_args()
    
    # Make sure config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    
    generate_random_input(args.config, args.output)

if __name__ == '__main__':
    main()
# Mamba Project

## Overview
This project implements an optimized version of the Mamba model in C, supporting both language modeling and audio keyword spotting (KWS) tasks.

## Features
- Language Model Support: Text generation with configurable parameters
- Keyword Spotting Support: Audio classification using MFCC features
- INT8/BFLOAT8 Quantization Support
- Optimized C Implementation

## Directory Structure
```
Mamba_Pro/
│── debug_weights.log  # Debugging logs
│── LICENSE            # License file
│── Makefile          # Makefile for building the project
│── mamba             # Executable binary
│── README.md         # Documentation
│── tokenizer.bin     # Tokenizer binary for language models
│
├── include/          # Header files
│   ├── config.h      # Model configuration and data structures
│   ├── mamba.h       # Core Mamba model implementation
│   ├── math_ops.h    # Mathematical operations
│   └── ...
│
├── models/           # Model files
│   ├── model.bin     # Language model weights
│   └── model.bin.scales
│
├── my_kws_model/     # Keyword spotting model
│   ├── config.json   # KWS model configuration
│   └── ...
│
├── py/               # Python scripts
│   ├── export.py     # Model export utilities
│   ├── generate_input.py  # KWS input generation
│   └── tokenizer.py
│
└── src/             # Source files
    ├── main.c       # Main program
    ├── mamba.c      # Model implementation
    └── ...
```

## Build and Run

### For Language Models
```sh
# Generate tokenizer (for language models only)
python3 py/tokenizer.py

# Export the model to INT8 format
python3 py/export.py state-spaces/mamba-130m models/model.bin --int8

# Build the project
make

# Run text generation
./mamba models/model.bin -n 20 -i "Customer Support should" -t 0.0
```

### For Keyword Spotting
```sh
# Export your KWS model
python3 py/export.py my_kws_model/best_model.pth output.bin --int8

# Generate test input (69 MFCC features = 23 coefficients × 3 for derivatives)
python3 py/generate_input.py --config my_kws_model/config.json --output input.raw

# Run inference
./mamba output.bin -i input.raw
```

## KWS Model Details
- Input: 69-dimensional MFCC features (23 coefficients × 3)
  - Base MFCC coefficients
  - First derivatives (delta)
  - Second derivatives (delta-delta)
- Output: 10 classes for keyword recognition
  - Available classes: yes, no, up, down, left, right, on, off, stop, go
- Model Architecture:
  - 2 Mamba layers
  - Input dimension: 69
  - Hidden dimension: 136
  - Inner dimension: 272
  - State dimension: 51
  - Convolution kernel size: 10

## Recent Updates
1. Added support for Keyword Spotting models
2. Implemented proper handling of MFCC features with derivatives
3. Updated model configuration to support classification tasks
4. Added input generation utility for KWS testing
5. Enhanced model export to handle classification heads

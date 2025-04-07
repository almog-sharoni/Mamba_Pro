# Mamba Project

## Overview
This project implements an optimized version of the Mamba model in C.

## Directory Structure
```
Mamba_Pro/
│── debug_weights.log  # Debugging logs
│── LICENSE            # License file
│── Makefile           # Makefile for building the project
│── mamba              # Executable binary
│── model_fp16.bin     # FP16 model file
│── README.md          # Documentation
│── tokenizer.bin      # Tokenizer binary
│
├── build/             # Compiled output
│   └── libmamba.a     # Static library
│
├── include/           # Header files
│   ├── config.h
│   ├── fp16.h
│   ├── half.hpp
│   ├── mamba.h
│   ├── math_ops.h
│   ├── memory_alloc.h
│   ├── sampler.h
│   ├── tokenizer.h
│   └── utils.h
│
├── models/            # Model files
│   ├── model.bin
│   └── model.bin.scales
│
├── py/                # Python scripts
│   ├── export.py
│   └── tokenizer.py
│
└── src/               # Source files
    ├── main.c
    ├── main.o
    ├── mamba.c
    ├── mamba.o
    ├── math_ops.c
    ├── math_ops.o
    ├── memory_alloc.c
    ├── memory_alloc.o
    ├── sampler.c
    ├── sampler.o
    ├── tokenizer.c
    ├── tokenizer.o
    ├── utils.c
    └── utils.o
```

## Build and Run
```sh
# Step 1: Generate tokenizer
python3 py/tokenizer.py

# Step 2: Export the model to INT8 format
python3 py/export.py state-spaces/mamba-130m models/model.bin --int8

# Step 3: Build the project using Makefile
make

# Step 4: Run the Mamba model with a sample input
./mamba models/model.bin -n 20 -i "Customer Support should" -t 0.0
```

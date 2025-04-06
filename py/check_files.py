#!/usr/bin/env python3
import os
import sys

def check_file(path):
    """Check if a file exists and print its status."""
    exists = os.path.exists(path)
    print(f"{'✓' if exists else '✗'} {path} {'exists' if exists else 'does not exist'}")
    return exists

def main():
    base_dir = "/workspace/libs/Mamba_Pro"
    lp5x_dir = "/workspace/libs/LP5X-PIMSimulator-release"
    
    print("\nChecking Mamba_Pro files:")
    mamba_files = [
        "include/config.h",
        "include/math_ops.h",
        "include/utils.h",
        "include/mamba.h",
        "src/math_ops.c",
        "src/mamba.c",
        "src/memory_alloc.c",
        "src/utils.c"
    ]
    
    for file in mamba_files:
        check_file(os.path.join(base_dir, file))
    
    print("\nChecking LP5X-PIMSimulator files:")
    lp5x_files = [
        "include/fp16.h",
        "include/half.hpp"
    ]
    
    for file in lp5x_files:
        check_file(os.path.join(lp5x_dir, file))

if __name__ == "__main__":
    main() 
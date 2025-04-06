#!/usr/bin/env python3
import os
import sys

# Define the base directory for the Mamba_Pro project
BASE_DIR = "/workspace/libs/Mamba_Pro"

# Define the files to check
FILES_TO_CHECK = [
    "include/config.h",
    "include/math_ops.h",
    "include/utils.h",
    "include/mamba.h",
    "include/fp16.h",
    "src/math_ops.c",
    "src/mamba.c",
    "src/memory_alloc.c",
    "src/utils.c"
]

def check_file(file_path):
    """Check if a file exists and print its first few lines."""
    try:
        if os.path.exists(file_path):
            print(f"\nFile: {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"File size: {len(content)} bytes")
                print("First 20 lines:")
                for i, line in enumerate(content.split('\n')[:20]):
                    print(f"{i+1}: {line}")
            return True
        else:
            print(f"\nFile not found: {file_path}")
            return False
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False

def main():
    print("Checking files after FP16 conversion:")
    
    for file_path in FILES_TO_CHECK:
        full_path = os.path.join(BASE_DIR, file_path)
        check_file(full_path)
    
    print("\nFP16 conversion check completed.")

if __name__ == "__main__":
    main() 
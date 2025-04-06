#!/usr/bin/env python3
import os
import re
import sys
import shutil

# Define the base directory for the Mamba_Pro project
BASE_DIR = "/workspace/libs/Mamba_Pro"
LP5X_DIR = "/workspace/libs/LP5X-PIMSimulator-release"

# Define the files to modify
HEADER_FILES = [
    "include/config.h",
    "include/math_ops.h",
    "include/utils.h",
    "include/mamba.h"
]

SOURCE_FILES = [
    "src/math_ops.c",
    "src/mamba.c",
    "src/memory_alloc.c",
    "src/utils.c"
]

def copy_fp16_header():
    """Copy the fp16.h file from LP5X-PIMSimulator-release to Mamba_Pro include directory."""
    source_file = os.path.join(LP5X_DIR, "include/fp16.h")
    dest_file = os.path.join(BASE_DIR, "include/fp16.h")
    
    # Also copy the half.hpp file which is included by fp16.h
    half_source = os.path.join(LP5X_DIR, "include/half.hpp")
    half_dest = os.path.join(BASE_DIR, "include/half.hpp")
    
    try:
        shutil.copy2(source_file, dest_file)
        print(f"Copied {source_file} to {dest_file}")
        
        if os.path.exists(half_source):
            shutil.copy2(half_source, half_dest)
            print(f"Copied {half_source} to {half_dest}")
        else:
            print(f"Warning: {half_source} not found")
        
        return True
    except Exception as e:
        print(f"Error copying fp16.h: {e}")
        return False

def replace_in_file(file_path, replacements):
    """Replace content in a file based on the provided replacements."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply all replacements
        for old, new in replacements:
            content = content.replace(old, new)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # First, copy the fp16.h file
    if not copy_fp16_header():
        print("Failed to copy fp16.h file. Exiting.")
        return
    
    # Define replacements for header files
    header_replacements = [
        ("int8_t*", "PimSim::float16*"),
        ("int8_t", "PimSim::float16"),
        ("#include <stdint.h>", "#include <stdint.h>\n#include <fp16.h>"),
    ]
    
    # Define replacements for source files
    source_replacements = [
        ("(int8_t *)", "(PimSim::float16 *)"),
        ("(int8_t*)", "(PimSim::float16*)"),
        ("int8_t", "PimSim::float16"),
        ("dequantize(int8_t q, float scale)", "dequantize(PimSim::float16 q, float scale)"),
        ("return q * scale;", "return PimSim::convertH2F(q) * scale;"),
    ]
    
    # Process header files
    for header_file in HEADER_FILES:
        file_path = os.path.join(BASE_DIR, header_file)
        if os.path.exists(file_path):
            replace_in_file(file_path, header_replacements)
        else:
            print(f"Warning: Header file {file_path} not found")
    
    # Process source files
    for source_file in SOURCE_FILES:
        file_path = os.path.join(BASE_DIR, source_file)
        if os.path.exists(file_path):
            replace_in_file(file_path, source_replacements)
        else:
            print(f"Warning: Source file {file_path} not found")
    
    # Special handling for math_ops.c to update the dequantize function
    math_ops_path = os.path.join(BASE_DIR, "src/math_ops.c")
    if os.path.exists(math_ops_path):
        with open(math_ops_path, 'r') as f:
            content = f.read()
        
        # Find the dequantize function and replace it
        dequantize_pattern = r'inline float dequantize\(PimSim::float16 q, float scale\) \{([^}]*)\}'
        dequantize_replacement = r'inline float dequantize(PimSim::float16 q, float scale) {\n    return PimSim::convertH2F(q) * scale;\n}'
        
        content = re.sub(dequantize_pattern, dequantize_replacement, content)
        
        with open(math_ops_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated dequantize function in {math_ops_path}")
    
    print("Conversion from int8_t to PimSim::float16 completed successfully!")

if __name__ == "__main__":
    main() 
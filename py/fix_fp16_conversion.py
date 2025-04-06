#!/usr/bin/env python3
import os
import re
import sys

# Define the base directory for the Mamba_Pro project
BASE_DIR = "/workspace/libs/Mamba_Pro"

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
    # Define replacements for header files
    header_replacements = [
        ("fp16*", "PimSim::float16*"),
        ("fp16", "PimSim::float16"),
        ("#include <fp16.h>\n#include <fp16.h>", "#include <fp16.h>"),
    ]
    
    # Define replacements for source files
    source_replacements = [
        ("(fp16 *)", "(PimSim::float16 *)"),
        ("(fp16*)", "(PimSim::float16*)"),
        ("fp16", "PimSim::float16"),
        ("dequantize(fp16 q, float scale)", "dequantize(PimSim::float16 q, float scale)"),
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
    
    print("Fixed FP16 conversion issues successfully!")

if __name__ == "__main__":
    main() 
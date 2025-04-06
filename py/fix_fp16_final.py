#!/usr/bin/env python3
import os
import re
import sys
import shutil

def fix_file(file_path, replacements, file_type=""):
    """Fix a file by applying the given replacements."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply all replacements
        for old, new in replacements:
            content = content.replace(old, new)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {file_type} file: {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    base_dir = "/workspace/libs/Mamba_Pro"
    lp5x_dir = "/workspace/libs/LP5X-PIMSimulator-release"
    
    # First, copy the required header files
    print("\nCopying header files...")
    shutil.copy2(os.path.join(lp5x_dir, "include/fp16.h"), os.path.join(base_dir, "include/fp16.h"))
    shutil.copy2(os.path.join(lp5x_dir, "include/half.hpp"), os.path.join(base_dir, "include/half.hpp"))
    
    # Fix config.h
    config_replacements = [
        ("int8_t*", "PimSim::float16*"),
        ("int8_t *", "PimSim::float16 *"),
        ("#include <stdlib.h>", "#include <stdlib.h>\n#include <fp16.h>")
    ]
    fix_file(os.path.join(base_dir, "include/config.h"), config_replacements, "config.h")
    
    # Fix math_ops.h
    math_ops_h_replacements = [
        ("int8_t", "PimSim::float16"),
        ("#include <stdint.h>", "#include <stdint.h>\n#include <fp16.h>"),
        ("inline float dequantize(PimSim::float16 q, float scale);", 
         "inline float dequantize(PimSim::float16 q, float scale);")
    ]
    fix_file(os.path.join(base_dir, "include/math_ops.h"), math_ops_h_replacements, "math_ops.h")
    
    # Fix math_ops.c
    math_ops_c_replacements = [
        ("int8_t", "PimSim::float16"),
        ("return q * scale;", "return PimSim::convertH2F(q) * scale;"),
        ("inline float dequantize(PimSim::float16 q, float scale) {",
         "inline float dequantize(PimSim::float16 q, float scale) {")
    ]
    fix_file(os.path.join(base_dir, "src/math_ops.c"), math_ops_c_replacements, "math_ops.c")
    
    # Fix mamba.c
    mamba_c_replacements = [
        ("(int8_t *)", "(PimSim::float16 *)"),
        ("(int8_t*)", "(PimSim::float16*)"),
        ("int8_t", "PimSim::float16")
    ]
    fix_file(os.path.join(base_dir, "src/mamba.c"), mamba_c_replacements, "mamba.c")
    
    # Fix memory_alloc.c
    memory_alloc_replacements = [
        ("int8_t*", "PimSim::float16*"),
        ("int8_t *", "PimSim::float16 *")
    ]
    fix_file(os.path.join(base_dir, "src/memory_alloc.c"), memory_alloc_replacements, "memory_alloc.c")
    
    # Fix utils.c
    utils_replacements = [
        ("int8_t*", "PimSim::float16*"),
        ("int8_t *", "PimSim::float16 *"),
        ("int8_t", "PimSim::float16")
    ]
    fix_file(os.path.join(base_dir, "src/utils.c"), utils_replacements, "utils.c")
    
    print("\nFP16 conversion completed successfully!")

if __name__ == "__main__":
    main() 
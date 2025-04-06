#!/usr/bin/env python3
import os
import sys

# Define the base directory for the Mamba_Pro project
BASE_DIR = "/workspace/libs/Mamba_Pro"

def fix_math_ops_h():
    """Fix the math_ops.h file."""
    file_path = os.path.join(BASE_DIR, "include/math_ops.h")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove duplicate include
        content = content.replace("#include <fp16.h>\n#include <fp16.h>", "#include <fp16.h>")
        
        # Update dequantize function signature
        content = content.replace("inline float dequantize(fp16 q, float scale);", 
                                 "inline float dequantize(PimSim::float16 q, float scale);")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_math_ops_c():
    """Fix the math_ops.c file."""
    file_path = os.path.join(BASE_DIR, "src/math_ops.c")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Update dequantize function implementation
        content = content.replace("inline float dequantize(fp16 q, float scale) {", 
                                 "inline float dequantize(PimSim::float16 q, float scale) {")
        content = content.replace("return q * scale;", 
                                 "return PimSim::convertH2F(q) * scale;")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("Fixing remaining issues with FP16 conversion...")
    
    fix_math_ops_h()
    fix_math_ops_c()
    
    print("Fixed remaining issues successfully!")

if __name__ == "__main__":
    main() 
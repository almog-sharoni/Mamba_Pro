#include "fp16.h"

// IEEE-754 half-precision format:
// 1 bit sign, 5 bits exponent, 10 bits mantissa
// Exponent bias is 15

fp16_t float_to_fp16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 0) {
        // Zero or denormal
        return (sign << 15);
    } else if (exp == 0xFF) {
        // Infinity or NaN
        return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
    }
    
    // Adjust exponent bias from float (127) to fp16 (15)
    int16_t fp16_exp = exp - 127 + 15;
    
    // Check for overflow/underflow
    if (fp16_exp > 31) {
        // Overflow to infinity
        return (sign << 15) | 0x7C00;
    } else if (fp16_exp < 0) {
        // Underflow to zero
        return (sign << 15);
    }
    
    // Construct fp16
    return (sign << 15) | (fp16_exp << 10) | (mantissa >> 13);
}

float fp16_to_float(fp16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    // Handle special cases
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            float f = sign ? -0.0f : 0.0f;
            return f;
        } else {
            // Denormal
            float f = (float)mantissa / 1024.0f;
            return sign ? -f : f;
        }
    } else if (exp == 0x1F) {
        if (mantissa == 0) {
            // Infinity
            return sign ? -INFINITY : INFINITY;
        } else {
            // NaN
            return NAN;
        }
    }
    
    // Adjust exponent bias from fp16 (15) to float (127)
    uint32_t float_exp = exp - 15 + 127;
    
    // Construct float
    uint32_t float_bits = (sign << 31) | (float_exp << 23) | (mantissa << 13);
    return *(float*)&float_bits;
} 
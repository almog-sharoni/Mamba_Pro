#ifndef FP16_H
#define FP16_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Define fp16 as a 16-bit type
typedef uint16_t fp16_t;

// Function declarations
float fp16_to_float(fp16_t h);
fp16_t float_to_fp16(float f);

#ifdef __cplusplus
}
#endif

#endif // FP16_H

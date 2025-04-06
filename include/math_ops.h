#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <stdint.h>
#include "fp16.h"

void rmsnorm(float* o, float* x, float* weight, int size);
void matmul(float* xout, float* x, void* w, int d, int n, float scale);
void linear(float* xout, float* x, void* w, void* b, int d, int n, float scale_w, float scale_b);
void softmax(float* x, int size);
float softplus(float x);
float sigmoid(float x);
float silu(float x);
void rowwise_dot_product(float* out, float* matrix, float* weights, int rows, int cols);
void shift_matrix_left(float* matrix, int rows, int cols);
void update_last_column(float* matrix, float* x, int rows, int cols);
void broadcast_multiply(float* out, float* x, float* y, int d, int n);
void elementwise_multiply(float* result, float* matrix1, void* matrix2, int total_elements,float scale);
void elementwise_add(float* result, float* matrix1, void* matrix2, int total_elements,float scale);
void elementwise_multiply_and_add(float* result, float* matrix1, float* matrix2, float* matrix3, int total_elements);
void outer_product(float* out, float* x, float* y, int d, int n);
void sum_along_last_dim(float* result, float* matrix, int rows, int cols);
inline float dequantize(fp16_t q, float scale);

#endif // MATH_OPS_H

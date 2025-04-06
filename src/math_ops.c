#include "math_ops.h"
#include <math.h>
#include <stdio.h>

// Optimized rmsnorm with SIMD reduction
void rmsnorm(float *o, float *x, float *weight, int size)
{
    float ss = 0.0f;
#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss = sqrtf(ss / size + 1e-5f);
#pragma omp simd
    for (int j = 0; j < size; j++)
    {
        o[j] = x[j] * weight[j] / ss;
    }
}

// Optimized matmul with SIMD inner loop
void matmul(float *xout, float *x, void *w, int d, int n, float scale)
{
#pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        float sum = 0.0f;
        int offset = i * n;
        if (scale == 0.0f)
        {
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < n; j++)
            {
                sum += ((float *)w)[offset + j] * x[j];
            }
        }
        else
        {
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < n; j++)
            {
                sum += dequantize(((fp16_t *)w)[offset + j], scale) * x[j];
            }
        }
        xout[i] = sum;
    }
}

// Optimized linear: similar to matmul but adds bias after inner loop
void linear(float *xout, float *x, void *w, void *b, int d, int n, float scale_w, float scale_b)
{
#pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        float sum = 0.0f;
        int offset = i * n;
#pragma omp simd reduction(+ : sum)
        for (int j = 0; j < n; j++)
        {
            sum += dequantize(((fp16_t *)w)[offset + j], scale_w) * x[j];
        }
        xout[i] = sum + dequantize(((fp16_t *)b)[i], scale_b);
    }
}

void softmax(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

float softplus(float x)
{
    return logf(1.0f + expf(x));
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float silu(float x)
{
    return x * sigmoid(x);
}
// Optimized rowwise_dot_product using SIMD reduction
void rowwise_dot_product(float *out, float *matrix, float *weights, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        int offset = i * cols;
#pragma omp simd reduction(+ : sum)
        for (int j = 0; j < cols; j++)
        {
            sum += matrix[offset + j] * weights[j];
        }
        out[i] = sum;
    }
}
void shift_matrix_left(float *matrix, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            matrix[i * cols + j] = matrix[i * cols + j + 1];
        }
    }
}
void update_last_column(float *matrix, float *x, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        matrix[i * cols + cols - 1] = x[i];
    }
}
void broadcast_multiply(float *out, float *x, float *y, int d, int n)
{
// x[d], y[d,n] -> out[d,n]
#pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int index = i * n + j;
            out[index] = x[i] * y[index];
            // out[i * n + j] = x[i] * y[i * n + j];
        }
    }
}
void elementwise_multiply(float *result, float *matrix1, void *matrix2, int total_elements, float scale)
{
#pragma omp parallel for
    for (int i = 0; i < total_elements; i++)
    {
        // result[i] = matrix1[i] * ((fp16_t*)matrix2)[i];
        result[i] = matrix1[i] * dequantize(((fp16_t *)matrix2)[i], scale);
    }
}
void elementwise_add(float *result, float *matrix1, void *matrix2, int total_elements, float scale)
{
#pragma omp parallel for
    for (int i = 0; i < total_elements; i++)
    {
        result[i] = matrix1[i] + dequantize(((fp16_t *)matrix2)[i], scale);
    }
}
void elementwise_multiply_and_add(float *result, float *matrix1, float *matrix2, float *matrix3, int total_elements)
{
#pragma omp parallel for
    for (int i = 0; i < total_elements; i++)
    {
        result[i] = matrix1[i] * matrix2[i] + matrix3[i];
    }
}

void outer_product(float *out, float *x, float *y, int d, int n)
{
// x[d], y[n] -> out[d,n]
#pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j] = x[i] * y[j];
        }
    }
}
void sum_along_last_dim(float *result, float *matrix, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            val += matrix[i * cols + j];
        }
        result[i] = val;
    }
}

inline float dequantize(fp16_t q, float scale) {
    return fp16_to_float(q) * 1.0f;
}

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "config.h"

// Function declarations
void print_int8_weights_sample(const fp16_t* weights, int size, const char* layer_name);
void print_float_weights_sample(const float* weights, int size, const char* layer_name);
void print_matrix(const float* matrix, int rows, int cols, const char* name);
void print_vector(const float* vector, int size, const char* name);
void print_int_vector(const int* vector, int size, const char* name);
void print_token(const char* token, int id);
void print_sampler_state(const Sampler* sampler);
void print_tokenizer_state(const Tokenizer* tokenizer);
void print_mamba_state(const Mamba* mamba);
void print_config(const Config* config);
void print_weights(const MambaWeights* weights);
void print_run_state(const RunState* state);
void print_token_index(const TokenIndex* token_index);
void print_prob_index(const ProbIndex* prob_index);
void print_tokenizer(const Tokenizer* tokenizer);
void print_sampler(const Sampler* sampler);
void print_mamba(const Mamba* mamba);

void safe_printf(char *piece);
long time_in_ms();
void read_stdin(const char* guide, char* buffer, size_t bufsize);
void error_usage();

#endif // UTILS_H

#ifndef SAMPLER_H
#define SAMPLER_H
#include <unistd.h>
#include <sys/mman.h>
#include "config.h"
#include "math_ops.h"

int sample_argmax(float* probabilities, int n);
int sample_mult(float* probabilities, int n, float coin);
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
float random_f32(unsigned long long *state);
int sample(Sampler* sampler, float* logits);
void free_sampler(Sampler* sampler);
unsigned int random_u32(unsigned long long *state);
int compare(const void* a, const void* b);
#endif // SAMPLER_H

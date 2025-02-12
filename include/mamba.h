#ifndef MAMBA_H
#define MAMBA_H
#include <unistd.h>
#include <sys/mman.h>
#include "config.h"
#include "math_ops.h"

void load_scales_file(const char* scales_path, MambaWeights *w, Config* p);
void load_model_file(char* model_path, Config* config, MambaWeights* weights,int* fd, float** data, ssize_t* file_size);
void load_model(Mamba* m, char* model_path);
void free_model(Mamba* m);
void forward_layer(Mamba* mamba, unsigned long long l, float* hidden_state);
float* forward(Mamba* mamba, int token);

#endif // MAMBA_H

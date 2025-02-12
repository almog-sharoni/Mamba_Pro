#ifndef MEMORY_ALLOC_H
#define MEMORY_ALLOC_H

#include "config.h"

void malloc_run_state(RunState* s, Config* p);
void reset_internal_state(Mamba* mamba);
char* get_internal_state(Mamba* mamba, int* state_size);
void set_internal_state(Mamba* mamba, char* state, int state_size);
void free_run_state(RunState* s);
void memory_map_weights(MambaWeights *w, Config* p, float* ptr);

#endif // MEMORY_ALLOC_H

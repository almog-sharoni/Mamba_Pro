#include "memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "config.h"
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

void malloc_run_state(RunState* s, Config* p) {
    // memory reused by all layers
    s->cls_token = malloc(p->dim * sizeof(float));
    s->input = malloc(p->dim * sizeof(float));
    s->hidden_state = malloc(p->dim * sizeof(float));
    s->xz = malloc(2 * p->d_inner * sizeof(float));
    s->x_db = malloc((p->dt_rank + 2 * p->d_state) * sizeof(float));
    s->dt = malloc(p->d_inner * sizeof(float));
    s->dA = malloc(p->d_inner * p->d_state * sizeof(float));
    s->dB = malloc(p->d_inner * p->d_state * sizeof(float));
    s->temp = malloc(p->d_inner * p->d_state * sizeof(float));
    s->y = malloc(p->d_inner * sizeof(float));
    s->logits = malloc(p->n_classes * sizeof(float));
    // internal state, separate memory for each layer
    s->conv_state = calloc(p->n_layers * p->d_inner * p->d_conv, sizeof(float));
    s->ssm_state = calloc(p->n_layers * p->d_inner * p->d_state, sizeof(float));
    s->layer_norms = malloc(p->n_layers * p->dim * sizeof(float));
    // ensure all mallocs went fine
    if (!s->cls_token || !s->input || !s->hidden_state || !s->xz || !s->x_db || !s->dt || 
        !s->dA || !s->dB || !s->temp || !s->y || !s->logits || !s->conv_state || 
        !s->ssm_state || !s->layer_norms) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void reset_internal_state(Mamba* mamba) {
    // reset the internal state of the model
    RunState* s = &mamba->state;
    Config* p = &mamba->config;
    memset(s->conv_state, 0, p->n_layers * p->d_inner * p->d_conv * sizeof(float));
    memset(s->ssm_state, 0, p->n_layers * p->d_inner * p->d_state * sizeof(float));
}

char* get_internal_state(Mamba* mamba, int* state_size) {
    // get the internal state of the model
    Config* p = &mamba->config;
    RunState* s = &mamba->state;
    unsigned int conv_state_size = p->n_layers * p->d_inner * p->d_conv * sizeof(float);
    unsigned int ssm_state_size = p->n_layers * p->d_inner * p->d_state * sizeof(float);
    unsigned int total_size = conv_state_size + ssm_state_size;
    char* state = malloc(total_size);
    if (state) {
        memcpy(state, s->conv_state, conv_state_size);
        memcpy(state + conv_state_size, s->ssm_state, ssm_state_size);
        *state_size = total_size;
    }
    return state;
}

void set_internal_state(Mamba* mamba, char* state, int state_size) {
    // set the internal state of the model
    Config* p = &mamba->config;
    RunState* s = &mamba->state;
    unsigned int conv_state_size = p->n_layers * p->d_inner * p->d_conv * sizeof(float);
    unsigned int ssm_state_size = p->n_layers * p->d_inner * p->d_state * sizeof(float);
    if (state_size == conv_state_size + ssm_state_size) {
        memcpy(s->conv_state, state, conv_state_size);
        memcpy(s->ssm_state, state + conv_state_size, ssm_state_size);
    }
}

void free_run_state(RunState* s) {
    free(s->cls_token);
    free(s->input);
    free(s->hidden_state);
    free(s->xz);
    free(s->x_db);
    free(s->dt);
    free(s->dA);
    free(s->dB);
    free(s->temp);
    free(s->y);
    free(s->logits);
    free(s->conv_state);
    free(s->ssm_state);
    free(s->layer_norms);
}

void memory_map_weights(MambaWeights *w, Config* p, float* ptr) {
    unsigned long long n_layers = p->n_layers;

    // Map cls_token and projection
    w->cls_token = ptr;
    ptr += p->dim;
    w->proj_weight = ptr;
    ptr += p->dim * p->input_dim;
    w->proj_bias = ptr;
    ptr += p->dim;

    // Map layer weights
    w->in_proj = (int8_t*)ptr;
    ptr += n_layers * (2 * p->d_inner * p->dim) / 4;
    w->conv1d_weight = (int8_t*)ptr;
    ptr += n_layers * (p->d_inner * p->d_conv) / 4;
    w->conv1d_bias = (int8_t*)ptr;
    ptr += n_layers * p->d_inner / 4;
    w->x_proj = (int8_t*)ptr;
    ptr += n_layers * ((p->dt_rank + 2 * p->d_state) * p->d_inner) / 4;
    w->dt_proj_weight = (int8_t*)ptr;
    ptr += n_layers * (p->d_inner * p->dt_rank) / 4;
    w->dt_proj_bias = (int8_t*)ptr;
    ptr += n_layers * p->d_inner / 4;

    // Map A and D (float)
    w->A = (float*)ptr;
    ptr += n_layers * p->d_inner * p->d_state;
    w->D = (float*)ptr;
    ptr += n_layers * p->d_inner;

    // Map out_proj and layer norms
    w->out_proj = (int8_t*)ptr;
    ptr += n_layers * p->dim * p->d_inner / 4;
    w->layer_norms = (float*)ptr;
    ptr += n_layers * p->dim;

    // Map classification head
    w->fc_weight = (int8_t*)ptr;
    ptr += p->n_classes * p->dim / 4;
    w->fc_bias = (float*)ptr;
    ptr += p->n_classes;
}




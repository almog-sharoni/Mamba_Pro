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
    s->input = malloc(p->dim * sizeof(float));
    s->hidden_state = malloc(p->dim * sizeof(float));
    s->xz = malloc(2 * p->d_inner * sizeof(float));
    s->x_db = malloc((p->dt_rank + 2 * p->d_state) * sizeof(float));
    s->dt = malloc(p->d_inner * sizeof(float));
    s->dA = malloc(p->d_inner * p->d_state * sizeof(float));
    s->dB = malloc(p->d_inner * p->d_state * sizeof(float));
    s->temp = malloc(p->d_inner * p->d_state * sizeof(float));
    s->y = malloc(p->d_inner * sizeof(float));
    s->logits = malloc(p->rounded_vocab_size * sizeof(float));
    // internal state, separate memory for each layer
    s->conv_state = calloc(p->n_layers * p->d_inner * p->d_conv, sizeof(float));
    s->ssm_state = calloc(p->n_layers * p->d_inner * p->d_state, sizeof(float));
    // ensure all mallocs went fine
    if (!s->xz || !s->x_db || !s->dt || !s->dA || !s->dB || !s->temp || !s->y
     || !s->logits || !s->conv_state || !s->ssm_state) {
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
}





void memory_map_weights(MambaWeights *w, Config* p, float* ptr) {
    unsigned long long n_layers = p->n_layers;


    // Token embedding table remains as float
    w->token_embedding_table = ptr;
    ptr += p->rounded_vocab_size * p->dim;

    // Weights for layers

    // INT8 weights with proper casting
    w->in_proj = ptr;  
    // print_int8_weights_sample(w->in_proj, n_layers * (2 * p->d_inner * p->dim) / 4, "in_proj");
    ptr += n_layers * (2 * p->d_inner * p->dim) / 4; // Adjust pointer for int8
    w->conv1d_weight = ptr; 
    // print_int8_weights_sample(w->conv1d_weight, n_layers * (p->d_inner * p->d_conv) / 4, "conv1d_weight");
    ptr += n_layers * (p->d_inner * p->d_conv) / 4; 
    w->conv1d_bias = ptr; 
    // print_int8_weights_sample(w->conv1d_bias, n_layers * (p->d_inner) / 4, "conv1d_bias");
    ptr += n_layers * (p->d_inner) / 4; 
    w->x_proj = ptr; 
    // print_int8_weights_sample(w->x_proj, n_layers * ((p->dt_rank + 2 * p->d_state) * p->d_inner) / 4, "x_proj");
    ptr += n_layers * ((p->dt_rank + 2 * p->d_state) * p->d_inner) / 4;
    w->dt_proj_weight = ptr; 
    // print_int8_weights_sample(w->dt_proj_weight, n_layers * (p->d_inner * p->dt_rank) / 4, "dt_proj_weight");
    ptr += n_layers * (p->d_inner * p->dt_rank) / 4;
    w->dt_proj_bias = ptr; 
    // print_int8_weights_sample(w->dt_proj_bias, n_layers * (p->d_inner) / 4, "dt_proj_bias");
    ptr += n_layers * (p->d_inner) / 4;

    // Shared layers remain as float
    w->A = (float*)ptr; 
    ptr += n_layers * p->d_inner * p->d_state;
    w->D = (float*)ptr; 
    ptr += n_layers * p->d_inner;

    // INT8 out_proj
    w->out_proj = (fp16_t*)ptr; 
    // print_int8_weights_sample(w->out_proj, n_layers * p->dim * p->d_inner / 4, "out_proj");
    ptr += n_layers * p->dim * p->d_inner / 4;


    // Norm layers and final norm
    w->norm = (float*)ptr; 
    ptr += n_layers * p->dim;
    w->final_norm = (float*)ptr; 
    ptr += p->dim;

    // Classifier weights
    w->lm_head = p->shared_classifier ? w->token_embedding_table : (float*)ptr;

}




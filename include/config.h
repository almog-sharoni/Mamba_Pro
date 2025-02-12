#ifndef CONFIG_H
#define CONFIG_H

#include <stdlib.h>



#define BOS 0
#define EOS 0


// Mamba model

typedef struct {
    int n_layers;   // number of layers
    int vocab_size; // vocabulary size
    int dim;        // embedding dimension
    int d_inner;
    int dt_rank;
    int d_state;
    int d_conv;
    int shared_classifier;
    int rounded_vocab_size; // vocab_size rounded up to the nearest multiple of 8
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table; // (rounded_vocab_size, dim)
    // weights for layers
    int8_t* in_proj;        // (layer, 2*d_inner, dim)
    int8_t* conv1d_weight;  // (layer, d_inner, 1, d_conv)
    int8_t* conv1d_bias;    // (layer, d_inner)
    int8_t* x_proj;         // (layer, dt_rank+2*d_state, d_inner)
    int8_t* dt_proj_weight; // (layer, d_inner, dt_rank)
    int8_t* dt_proj_bias;   // (layer, d_inner)
    float* A;              // (layer, d_inner, d_state)
    float* D;              // (layer, d_inner)
    int8_t* out_proj;       // (layer, dim, d_inner)
    float* norm;           // (layer, dim)
    // final rmsnorm
    float* final_norm;     // (dim)
    // (optional) classifier weights for the logits, on the last layer
    float* lm_head;        // (rounded_vocab_size, dim)


    float* in_proj_scale;
    float* conv1d_weight_scale;
    float* conv1d_bias_scale;
    float* x_proj_scale;
    float* dt_proj_weight_scale;
    float* dt_proj_bias_scale;
    float* out_proj_scale;
} MambaWeights;



typedef struct {
    // memory reused by all layers
    float* input;        // (dim)
    float* hidden_state; // (dim)
    float *xz;     // (2*d_inner)          x and z are pointers into this buffer
    float *x_db;   // (dt_rank+2*d_state)  dt, B, C are pointers into this buffer
    float *dt;     // (d_inner)            later, dt is a pointer to this buffer
    float *dA;     // (d_inner, d_state)
    float *dB;     // (d_inner, d_state)
    float *temp;   // (d_inner, d_state)
    float *y;      // (d_inner)
    float *logits; // (rounded_vocab_size)
    // internal state, separate memory for each layer
    float* conv_state; // (n_layers, d_inner, d_conv)
    float* ssm_state;  // (n_layers, d_inner, d_state)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    MambaWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
} Mamba;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

#endif

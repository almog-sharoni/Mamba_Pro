#ifndef CONFIG_H
#define CONFIG_H

#include <stdlib.h>



#define BOS 0
#define EOS 0


// Mamba model

typedef struct {
    int n_layers;      // number of layers
    int n_classes;     // number of output classes (10)
    int dim;          // embedding dimension (136)
    int input_dim;    // input dimension (3*23=69 for MFCC + delta + delta-delta)
    int d_inner;      // inner dimension (272)
    int dt_rank;      // delta rank (9)
    int d_state;      // state dimension (51)
    int d_conv;       // conv dimension (10)
} Config;

typedef struct {
    // model core components
    float* cls_token;              // (1, 1, dim)
    float* proj_weight;            // (dim, input_dim)
    float* proj_bias;              // (dim)
    // layer weights
    int8_t* in_proj;              // (layer, 2*d_inner, dim)
    int8_t* conv1d_weight;        // (layer, d_inner, d_conv)
    int8_t* conv1d_bias;          // (layer, d_inner)
    int8_t* x_proj;               // (layer, dt_rank+2*d_state, d_inner)
    int8_t* dt_proj_weight;       // (layer, d_inner, dt_rank)
    int8_t* dt_proj_bias;         // (layer, d_inner)
    float* A;                     // (layer, d_inner, d_state)
    float* D;                     // (layer, d_inner)
    int8_t* out_proj;            // (layer, dim, d_inner)
    float* layer_norms;          // (layer, dim)
    // classification head
    int8_t* fc_weight;           // (n_classes, dim)
    float* fc_bias;              // (n_classes)
    // quantization scales
    float* in_proj_scale;
    float* conv1d_weight_scale;
    float* conv1d_bias_scale;
    float* x_proj_scale;
    float* dt_proj_weight_scale;
    float* dt_proj_bias_scale;
    float* out_proj_scale;
    float* fc_weight_scale;
} MambaWeights;

typedef struct {
    // memory reused by all layers
    float* cls_token;     // (dim)
    float* input;         // (dim)
    float* hidden_state;  // (dim)
    float* xz;           // (2*d_inner)
    float* x_db;         // (dt_rank+2*d_state)
    float* dt;           // (d_inner)
    float* dA;           // (d_inner, d_state)
    float* dB;           // (d_inner, d_state)
    float* temp;         // (d_inner, d_state)
    float* y;            // (d_inner)
    float* logits;       // (n_classes)
    // internal state
    float* conv_state;   // (n_layers, d_inner, d_conv)
    float* ssm_state;    // (n_layers, d_inner, d_state)
    float* layer_norms;  // (n_layers, dim)
} RunState;

typedef struct {
    Config config;      // model configuration
    MambaWeights weights; // model weights
    RunState state;     // runtime buffers
    int fd;            // file descriptor for memory mapping
    float* data;       // memory mapped data pointer
    size_t file_size;  // size of checkpoint file in bytes
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

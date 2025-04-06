#include "mamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>
#include "memory_alloc.h"


// ----------------------------------------------------------------------------
void load_scales_file(const char* scales_path, MambaWeights *w, Config* p) {
    FILE *file = fopen(scales_path, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open scales file %s\n", scales_path);
        exit(EXIT_FAILURE);
    }
    
    // Read scale factors for each quantized weight tensor
    w->in_proj_scale = malloc(p->n_layers * sizeof(float));
    w->conv1d_weight_scale = malloc(p->n_layers * sizeof(float));
    w->conv1d_bias_scale = malloc(p->n_layers * sizeof(float));
    w->x_proj_scale = malloc(p->n_layers * sizeof(float));
    w->dt_proj_weight_scale = malloc(p->n_layers * sizeof(float));
    w->dt_proj_bias_scale = malloc(p->n_layers * sizeof(float));
    w->out_proj_scale = malloc(p->n_layers * sizeof(float));
    
    if (!w->in_proj_scale || !w->conv1d_weight_scale || !w->conv1d_bias_scale ||
        !w->x_proj_scale || !w->dt_proj_weight_scale || !w->dt_proj_bias_scale ||
        !w->out_proj_scale) {
        fprintf(stderr, "Failed to allocate memory for scale factors.\n");
        exit(EXIT_FAILURE);
    }
    
    fread(w->in_proj_scale, sizeof(float), p->n_layers, file);
    fread(w->conv1d_weight_scale, sizeof(float), p->n_layers, file);
    fread(w->conv1d_bias_scale, sizeof(float), p->n_layers, file);
    fread(w->x_proj_scale, sizeof(float), p->n_layers, file);
    fread(w->dt_proj_weight_scale, sizeof(float), p->n_layers, file);
    fread(w->dt_proj_bias_scale, sizeof(float), p->n_layers, file);
    fread(w->out_proj_scale, sizeof(float), p->n_layers, file);
    
    fclose(file);
}

void load_model_file(char* model_path, Config* config, MambaWeights* weights,
    int* fd, float** data, ssize_t* file_size) {
FILE *file = fopen(model_path, "rb");
if (!file) { fprintf(stderr, "Couldn't open file %s\n", model_path); exit(EXIT_FAILURE); }
// read the magic number
unsigned int magic;
if (fread(&magic, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
if (magic != 0x4d616d62) { fprintf(stderr, "Invalid magic number: %x\n", magic); exit(EXIT_FAILURE); }
// read the version
int version;
if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
if (version != 1) { fprintf(stderr, "Invalid version: %d\n", version); exit(EXIT_FAILURE); }
// read the config
if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
if (config->vocab_size % 8 != 0) {
config->rounded_vocab_size = config->vocab_size + (8 - (config->vocab_size % 8));
} else {
config->rounded_vocab_size = config->vocab_size;
}
// figure out the file size
ssize_t header_bytes = 4 + 4 + sizeof(Config);
fseek(file, 0, SEEK_END); // move file pointer to end of file
*file_size = ftell(file); // get the file size, in bytes
fclose(file);
// memory map the model weights into the data pointer
*fd = open(model_path, O_RDONLY); // open in read only mode
if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
float* weights_ptr = *data + (256 / 4);
// float* weights_ptr = (float*)(((char*)*data) + header_bytes);

memory_map_weights(weights, config, weights_ptr);

}



void load_model(Mamba* m, char* model_path) {
    // read the Config and the Weights from the model file
    load_model_file(model_path, &m->config, &m->weights, &m->fd, &m->data, &m->file_size);
    



    // allocate the RunState buffers
    malloc_run_state(&m->state, &m->config);

    // Load scale factors from separate scales file
    char scales_path[512];
    snprintf(scales_path, sizeof(scales_path), "%s.scales", model_path);
    load_scales_file(scales_path, &m->weights, &m->config);

}

void free_model(Mamba* m) {
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

void forward_layer(Mamba* mamba, unsigned long long l, float* hidden_state) {
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    int dim = p->dim, d_inner = p->d_inner, d_conv = p->d_conv, d_state = p->d_state, dt_rank = p->dt_rank;
    float* dA = s->dA;  // (d_inner, d_state)
    float* dB = s->dB;  // (d_inner, d_state)
    float* y = s->y;    // (d_inner)

    // conv_state, ssm_state = self._get_states_from_cache(inference_params)
    float* conv_state = s->conv_state + l * d_inner * d_conv;
    float* ssm_state = s->ssm_state + l * d_inner * d_state;

    // xz = self.in_proj(hidden_states)  # hidden_states: (dim), in_proj (2*d_inner, dim), xz (2*d_inner)
    float scale_in_proj = w->in_proj_scale[l];
    matmul(s->xz, hidden_state, (fp16_t*)w->in_proj + l * 2*d_inner*dim, 2*d_inner, dim, scale_in_proj);
    

    // x, z = xz.chunk(2, dim=-1)
    float* x = s->xz;            // x (d_inner)
    float* z = s->xz + d_inner;  // z (d_inner)

    // Conv step
    shift_matrix_left(conv_state, d_inner, d_conv);        // Roll conv_state
    update_last_column(conv_state, x, d_inner, d_conv);

    // Dequantize conv1d_weight and conv1d_bias
    float scale = w->conv1d_weight_scale[l];
    elementwise_multiply(s->temp, conv_state, (fp16_t*)w->conv1d_weight + l * d_inner * d_conv, d_inner * d_conv, scale);
    sum_along_last_dim(x, s->temp, d_inner, d_conv);
    float scale_bias = w->conv1d_bias_scale[l];
    elementwise_add(x, x, (fp16_t*)w->conv1d_bias + l*d_inner, d_inner, scale_bias);



    for (int i = 0; i < d_inner; i++) {
        x[i] = silu(x[i]);
    }

    // SSM step

    // Dequantize x_proj and dt_proj_weight
    float scale_x_proj = w->x_proj_scale[l];
    matmul(s->x_db, x, (fp16_t*)w->x_proj + l * (dt_rank + 2 * d_state) * d_inner, dt_rank + 2 * d_state, d_inner, scale_x_proj);
    // Apply dequantization if needed
    // dequantize_float(s->x_db, scale_x_proj); 


    float *dt = s->x_db;                   // dt (dt_rank)
    float *B = s->x_db + dt_rank;          // B  (d_state)
    float *C = s->x_db + dt_rank + d_state;  // C  (d_state)

    // Dequantize dt_proj_weight and dt_proj_bias
    float scale_dt_proj = w->dt_proj_weight_scale[l];
    float scale_dt_proj_bias = w->dt_proj_bias_scale[l];
    linear(s->dt, dt, (fp16_t*)w->dt_proj_weight + l * d_inner * dt_rank, (fp16_t*)w->dt_proj_bias + l * d_inner, d_inner, dt_rank,scale_dt_proj,scale_dt_proj_bias);
    // Apply dequantization if needed
    // dequantize_float(s->dt, scale_dt_proj);

    dt = s->dt;  // NOTE: dt is now bigger: (d_inner) instead of (dt_rank)


    for (int i = 0; i < d_inner; i++) {
        dt[i] = softplus(dt[i]);
    }

    // Discretize A and B
    broadcast_multiply(dA, dt, w->A + l * d_inner * d_state, d_inner, d_state);
    for (int i = 0; i < d_inner * d_state; i++) {
        dA[i] = expf(dA[i]);
    }
    outer_product(dB, dt, B, d_inner, d_state);

    // Update ssm_state
    broadcast_multiply(s->temp, x, dB, d_inner, d_state);
    elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp, d_inner * d_state);

    // Compute y
    rowwise_dot_product(y, ssm_state, C, d_inner, d_state);
    elementwise_multiply_and_add(y, w->D + l * d_inner, x, y, d_inner);

    for (int i = 0; i < d_inner; i++) {
        y[i] = y[i] * silu(z[i]);
    }


    // Dequantize out_proj
    float scale_out_proj = w->out_proj_scale[l];
    matmul(hidden_state, y, (fp16_t*)w->out_proj + l * dim * d_inner, dim, d_inner, scale_out_proj);
    // Apply dequantization if needed
    // dequantize_float(hidden_state, scale_out_proj);   

}


float* forward(Mamba* mamba, int token) {
    // a few convenience variables
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    int dim = p->dim;
    float *input = s->input;
    float *hidden_state = s->hidden_state;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(input, content_row, dim * sizeof(float));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // normalize the input
        rmsnorm(hidden_state, input, w->norm + l * dim, dim);
        // forward this layer
        forward_layer(mamba, l, hidden_state);
        // residual connection back into hidden_state
        for (int i = 0; i < dim; i++) {
            hidden_state[i] += input[i];
            // copy hidden_state back into input for the next layer
            input[i] = hidden_state[i];
        }
    }

    // final rmsnorm
    rmsnorm(hidden_state, hidden_state, w->final_norm, dim);

    // classifier into logits
    matmul(s->logits, hidden_state, w->lm_head, p->rounded_vocab_size, p->dim, 0.0f);
    return s->logits;
}


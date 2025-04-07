#include <stdio.h>
#include "mamba.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Classification inference function
void classify(Mamba *mamba, float* input) {
    // Forward pass through the model
    float* logits = forward(mamba, input);
    
    // Find the highest scoring class
    int best_class = 0;
    float best_score = logits[0];
    for (int i = 1; i < mamba->config.n_classes; i++) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_class = i;
        }
    }
    
    // Print the prediction
    printf("Predicted class: %d (confidence: %.2f%%)\n", best_class, 100.0f * best_score);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *model_path = NULL;    // e.g. model.bin
    char *input_path = NULL;    // path to input features
    float *input_features = NULL;

    // parse command line arguments
    if (argc >= 2) { model_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'i') { input_path = argv[i + 1]; }
        else { error_usage(); }
    }

    if (!input_path) {
        fprintf(stderr, "Error: input file path (-i) is required\n");
        error_usage();
    }

    // load the model
    Mamba mamba;
    load_model(&mamba, model_path);

    // print the config
    fprintf(stderr, "Model config: n_layers=%d, dim=%d, d_inner=%d, dt_rank=%d, d_state=%d, d_conv=%d, n_classes=%d\n",
            mamba.config.n_layers, mamba.config.dim, mamba.config.d_inner,
            mamba.config.dt_rank, mamba.config.d_state, mamba.config.d_conv,
            mamba.config.n_classes);

    // load input features
    FILE *f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: couldn't open input file %s\n", input_path);
        exit(EXIT_FAILURE);
    }
    input_features = malloc(mamba.config.input_dim * sizeof(float));
    size_t read_count = fread(input_features, sizeof(float), (size_t)mamba.config.input_dim, f);
    if (read_count != (size_t)mamba.config.input_dim) {
        fprintf(stderr, "Error: couldn't read input features (read %zu/%d)\n", read_count, mamba.config.input_dim);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    // run classification
    classify(&mamba, input_features);

    // cleanup
    free(input_features);
    free_model(&mamba);
    return 0;
}

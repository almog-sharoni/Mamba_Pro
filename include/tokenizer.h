#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "config.h"
#include <stdlib.h>


void build_tokenizer(Tokenizer* t, char* tokenizer_path, int model_vocab_size);
void free_tokenizer(Tokenizer* t);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer* t, char *text, int8_t add_bos, int8_t add_eos, int *tokens, int *n_tokens);
int compare_tokens(const void *a, const void *b);
char* decode(Tokenizer* t, int prev_token, int token);
#endif // TOKENIZER_H

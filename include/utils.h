#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h> 
#include "config.h"


void safe_printf(char *piece);
long time_in_ms();
void read_stdin(const char* guide, char* buffer, size_t bufsize);
void error_usage();
void print_int8_weights_sample(const int8_t* weights, int size, const char* layer_name);
#endif // UTILS_H

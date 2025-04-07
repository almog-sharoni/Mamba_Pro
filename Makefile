CC = gcc
CFLAGS = -Wall -Wextra -O2 -g
LDFLAGS = -lm

SRC = src/main.c src/mamba.c src/math_ops.c src/tokenizer.c src/sampler.c src/utils.c src/memory_alloc.c
OBJ = $(SRC:.c=.o)

INCLUDE = -Iinclude

all: mamba

mamba: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJ) -o mamba

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f $(OBJ) mamba

.PHONY: all clean

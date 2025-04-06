CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -Iinclude
LDFLAGS = -lm

SRCS = src/main.c src/mamba.c src/memory_alloc.c src/utils.c src/math_ops.c src/sampler.c src/tokenizer.c src/fp16.c
OBJS = $(SRCS:.c=.o)
TARGET = mamba

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean

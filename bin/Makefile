CC = gcc
CFLAGS = -O3 -march=native -g -Wall -Wextra
LDLIBS = -lOpenCL -lm

BIN = main

SRCS_HOST = $(wildcard bin_host/*.c)
SRCS_DEVICE = $(wildcard bin_device/*.c)
SRCS_MAIN = main.c

HDRS_HOST = $(wildcard bin_host/*.h)
HDRS_DEVICE = $(wildcard bin_device/*.h)

# Object files
OBJS_HOST = $(SRCS_HOST:.c=.o)
OBJS_DEVICE = $(SRCS_DEVICE:.c=.o)
OBJS_MAIN = $(SRCS_MAIN:.c=.o)

all: $(BIN)

$(BIN): $(OBJS_HOST) $(OBJS_DEVICE) $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

bin_host/%.o: bin_host/%.c $(HDRS_HOST)
	$(CC) $(CFLAGS) -c -o $@ $<

bin_device/%.o: bin_device/%.c $(HDRS_DEVICE)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS_HOST) $(OBJS_DEVICE) $(OBJS_MAIN) $(BIN)

.PHONY: all clean

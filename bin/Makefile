CC = gcc
CFLAGS = -O3 -march=native -g -Wall -Wextra
LDLIBS = -lOpenCL

# Ottieni la lista di tutti i file sorgente .c nella directory corrente
SRCS = $(wildcard *.c)

# Sostituisci l'estensione .c con .o per ottenere la lista dei file oggetto
OBJS = $(SRCS:.c=.o)

# Nome dell'eseguibile da creare
EXEC = main

# Regola predefinita per creare l'eseguibile
all: $(EXEC)

# Regola per creare l'eseguibile utilizzando i file oggetto
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Regola generica per compilare i file sorgente in file oggetto
%.o: %.c
	$(CC) $(CFLAGS) -c $<

# Pulizia dei file oggetto e dell'eseguibile
clean:
	rm -f $(OBJS) $(EXEC)

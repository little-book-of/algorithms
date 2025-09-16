CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2
LDFLAGS :=

OBJS_MAIN := main.o dec_to_bin.o bin_to_dec.o
OBJS_EXER := exercises_starter.o dec_to_bin.o bin_to_dec.o

.PHONY: all run test clean

all: main exercises

main: $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

exercises: $(OBJS_EXER)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

main.o: main.c dec_to_bin.h bin_to_dec.h
exercises_starter.o: exercises_starter.c dec_to_bin.h bin_to_dec.h

run: main
	./main

test: all
	./tests.sh

clean:
	rm -f main exercises *.o

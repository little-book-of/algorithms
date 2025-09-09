CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2
LDFLAGS :=

OBJS_MAIN := demo_bases.o twos_complement.o
OBJS_EXER := exercises_starter.o twos_complement.o

.PHONY: all run test clean

all: demo exercises

demo: $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

exercises: $(OBJS_EXER)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

demo_bases.o: demo_bases.c
twos_complement.o: twos_complement.c twos_complement.h
exercises_starter.o: exercises_starter.c twos_complement.h

run: demo
	./demo

test: all
	./tests.sh

clean:
	rm -f demo exercises *.o

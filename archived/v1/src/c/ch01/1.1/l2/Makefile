CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2
LDFLAGS := -lm

OBJS_DEMO := demo_floats.o ieee754_utils.o
OBJS_EXER := exercises_starter.o ieee754_utils.o

.PHONY: all run test clean

all: demo exercises

demo: $(OBJS_DEMO)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

exercises: $(OBJS_EXER)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

demo_floats.o: demo_floats.c ieee754_utils.h
exercises_starter.o: exercises_starter.c ieee754_utils.h
ieee754_utils.o: ieee754_utils.c ieee754_utils.h

run: demo
	./demo

test: all
	./tests.sh

clean:
	rm -f demo exercises *.o

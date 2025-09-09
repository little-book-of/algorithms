CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2

.PHONY: all run test clean

all: demo_basic_ops binary_ops exercises

demo_basic_ops: demo_basic_ops.o
	$(CC) $(CFLAGS) -o $@ $^

binary_ops: binary_ops.o
	$(CC) $(CFLAGS) -o $@ $^

exercises: exercises_starter.o
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./demo_basic_ops
	./binary_ops
	./exercises

test: all
	./tests.sh

clean:
	rm -f demo_basic_ops binary_ops exercises *.o

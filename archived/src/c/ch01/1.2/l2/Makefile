CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2

.PHONY: all run test clean

all: demo_multiplication karatsuba_demo modexp_demo exercises

demo_multiplication: demo_multiplication.o
	$(CC) $(CFLAGS) -o $@ $^

karatsuba_demo: karatsuba.o karatsuba_demo.o
	$(CC) $(CFLAGS) -o $@ $^

modexp_demo: modexp.o modexp_demo.o
	$(CC) $(CFLAGS) -o $@ $^

exercises: karatsuba.o modexp.o exercises_starter.o
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./demo_multiplication
	./karatsuba_demo
	./modexp_demo
	./exercises

test: all
	./tests.sh

clean:
	rm -f *.o demo_multiplication karatsuba_demo modexp_demo exercises

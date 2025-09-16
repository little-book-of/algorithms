CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2

.PHONY: all run test clean

all: demo_number_theory exercises

demo_number_theory: demo_number_theory.o gcd_lcm.o modular_identities.o fractions_utils.o
	$(CC) $(CFLAGS) -o $@ $^

exercises: exercises_starter.o gcd_lcm.o modular_identities.o fractions_utils.o
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./demo_number_theory
	./exercises

test: all
	./tests.sh

clean:
	rm -f demo_number_theory exercises *.o
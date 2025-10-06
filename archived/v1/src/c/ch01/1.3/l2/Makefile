CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2

.PHONY: all run test clean

all: demo_number_theory_advanced exercises

demo_number_theory_advanced: demo_number_theory_advanced.o extgcd_inv.o totient.o miller_rabin.o crt.o
	$(CC) $(CFLAGS) -o $@ $^

exercises: exercises_starter.o extgcd_inv.o totient.o miller_rabin.o crt.o
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./demo_number_theory_advanced
	./exercises

test: all
	./tests.sh

clean:
	rm -f demo_number_theory_advanced exercises *.o
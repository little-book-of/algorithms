CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2 $(SAN)
LDFLAGS := $(SAN)

.PHONY: all clean test run

all: demo_overflow_precision_practical exercises

demo_overflow_precision_practical: demo_overflow_precision_practical.o int_arith.o float_num.o fixed_point.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

exercises: exercises_starter.o int_arith.o float_num.o fixed_point.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: all
	./demo_overflow_precision_practical
	./exercises

test: all
	./tests.sh

clean:
	rm -f demo_overflow_precision_practical exercises *.o

CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2

.PHONY: all run test clean

all: demo_properties exercises

demo_properties: demo_properties.o parity.o divisibility.o remainder.o
	$(CC) $(CFLAGS) -o $@ $^

exercises: exercises_starter.o parity.o divisibility.o remainder.o
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./demo_properties
	./exercises

test: all
	./tests.sh

clean:
	rm -f demo_properties exercises *.o

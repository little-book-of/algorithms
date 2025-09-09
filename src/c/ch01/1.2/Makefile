CC      := cc
CFLAGS  := -std=c23 -Wall -Wextra -O2

.PHONY: all run test clean

all: demo_divmod modulo_examples efficiency_demo exercises

demo_divmod: demo_divmod.o
	$(CC) $(CFLAGS) -o $@ $^

modulo_examples: modulo_examples.o
	$(CC) $(CFLAGS) -o $@ $^

efficiency_demo: efficiency_demo.o
	$(CC) $(CFLAGS) -o $@ $^

exercises: exercises_starter.o
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./demo_divmod
	./modulo_examples
	./efficiency_demo
	./exercises

test: all
	./tests.sh

clean:
	rm -f demo_divmod modulo_examples efficiency_demo exercises *.o

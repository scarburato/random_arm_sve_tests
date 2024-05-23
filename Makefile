CFLAGS = -march=armv8-a+sve -g -std=c++20 -fno-exceptions -O3 -Ofast

.SUFFIXES:
.PHONY: all

all: test_exp test_log test_tanh
	echo "done"

%.o: %.cpp
	g++ $(CFLAGS) -c $^ -o $@

test_%: test_%.o Benchmark.o
	g++ $(CLFLAGS) $^ -o $@

clean:
	rm *.o test_exp test_log test_tanh
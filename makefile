# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h
ifeq ($(DEBUG),1)
CFLAGS = -fPIC  -std=c99 -ggdb -mavx2 -march=native -Wall -Wextra -Wshadow -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address
else
CFLAGS = -fPIC -std=c99 -O3 -mavx2  -march=native -Wall -Wextra -Wshadow
endif # debug
all:  basic_benchmark

HEADERS=./include/avx_hamming_weight.h ./include/hamming_weight.h ./include/popcnt_hamming_weight.h ./include/scalar_hamming_weight.h ./include/tabulated_hamming_weight.h

OBJECTS= avx_hamming_weight.o popcnt_hamming_weight.o scalar_hamming_weight.o \
		 tabulated_hamming_weight.o

%.o: ./src/%.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -Iinclude

basic_benchmark: ./benchmarks/basic_benchmark.c  ./benchmarks/benchmark.h  $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o basic_benchmark ./benchmarks/basic_benchmark.c -Iinclude  $(OBJECTS)

clean:
	rm -f basic_benchmark *.o

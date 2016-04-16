# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h
CFLAGS= -fPIC -std=c99 -Wall -Wextra -Wshadow
ifeq ($(DEBUG),1)
CFLAGS += -ggdb -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address
else
CFLAGS += -O3
endif # debug

ifeq ($(SSE),1)
CLFAGS += -msse -march=native
else
CLFAGS += -mavx2 -march=native -DHAVE_AVX2_INSTRUCTIONS
endif # sse

ifneq ($(NOPOPCNT),1)
CFLAGS += -DHAVE_POPCNT_INSTRUCTION
endif


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

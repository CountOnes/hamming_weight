# minimalist makefile
.SUFFIXES:
.PHONY: all clean
#
.SUFFIXES: .cpp .o .c .h
CFLAGS= -fPIC -std=c99 -Wall -Wextra -Wshadow
ifeq ($(DEBUG),1)
CFLAGS += -ggdb -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address
else
CFLAGS += -O3
endif # debug

ifeq ($(SSE),1)
CFLAGS += -msse -march=native
else
ifeq ($(AVX512),1)
CFLAGS += -mavx512vbmi -march=native -DHAVE_AVX2_INSTRUCTIONS -DHAVE_AVX512_INSTRUCTIONS
else
CFLAGS += -mavx2 -march=native -DHAVE_AVX2_INSTRUCTIONS
endif # avx512
endif # sse

ifneq ($(NOPOPCNT),1)
CFLAGS += -DHAVE_POPCNT_INSTRUCTION
endif


all: unit basic_benchmark jaccard_benchmark

HEADERS=./include/avx_hamming_weight.h ./include/hamming_weight.h ./include/popcnt_hamming_weight.h ./include/scalar_hamming_weight.h ./include/tabulated_hamming_weight.h ./include/avx_harley_seal_hamming_weight.h ./include/config.h ./include/avx512_hamming_weight.h ./include/sse_hamming_weight.h

OBJECTS= avx_hamming_weight.o popcnt_hamming_weight.o scalar_hamming_weight.o \
		 tabulated_hamming_weight.o avx_harley_seal_hamming_weight.o \
         avx512_hamming_weight.o \
         sse_hamming_weight.o

JACCARD_OBJ=sse_jaccard_index.o \
            popcnt_jaccard_index.o

%.o: ./src/%.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -Iinclude

basic_benchmark: ./benchmarks/basic_benchmark.c  ./benchmarks/benchmark.h  $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ ./benchmarks/basic_benchmark.c -Iinclude  $(OBJECTS)

jaccard_benchmark: ./benchmarks/jaccard_benchmark.c  ./benchmarks/benchmark.h  $(HEADERS) $(JACCARD_OBJ)
	$(CC) $(CFLAGS) -o $@ ./benchmarks/jaccard_benchmark.c -Iinclude  $(JACCARD_OBJ)

unit: ./tests/unit.c  $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o unit ./tests/unit.c -Iinclude  $(OBJECTS)


avx512: basic_benchmark
	sde -cnl -- ./basic_benchmark

clean:
	rm -f unit basic_benchmark *.o

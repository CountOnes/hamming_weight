#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>


#include "benchmark.h"
#include "hamming_weight.h"
void *aligned_malloc(size_t alignment, size_t size) {
    void *mem;
    if (posix_memalign(&mem, alignment, size)) exit(1);
    return mem;
}

void demo(int size) {
    printf("size = %d words or %lu bytes \n",size,  size*sizeof(uint64_t));
    int repeat = 500;
    uint64_t * prec = aligned_malloc(32,size * sizeof(uint64_t));
    for(int k = 0; k < size; ++k) prec[k] = -k;
    int expected = scalar_bitset64_weight(prec,size);

    BEST_TIME(lauradoux_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(scalar_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(table_bitset8_weight((uint8_t*)prec,size*8),expected,, repeat, size);
    BEST_TIME(table_bitset16_weight((uint16_t*)prec,size*4),expected,, repeat, size);
#if defined(HAVE_POPCNT_INSTRUCTION)
    BEST_TIME(popcnt_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(unrolled_popcnt_bitset64_weight(prec,size),expected,, repeat, size);
#else
    printf("no popcnt instruction\n");
#endif
#if defined(HAVE_AVX2_INSTRUCTIONS)
    BEST_TIME(avx2_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx_lauradoux_bitset64_weight(prec,size),expected,, repeat, size);
#else
    printf("no AVX2 instructions\n");
#endif

    free(prec);
    printf("\n");
}

int main() {
    for(int w = 8; w <= 1024; w *= 2) {
      demo(w);
    }
    return 0;
}
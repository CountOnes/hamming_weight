#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>


#include "benchmark.h"
#include "hamming_weight.h"

void demo(int size) {
    printf("size = %d words or %lu bytes \n",size,  size*sizeof(uint64_t));
    int repeat = 500;
    uint64_t * prec = malloc(size * sizeof(uint64_t));
    for(int k = 0; k < size; ++k) prec[k] = -k;
    int expected = scalar_bitset64_weight(prec,size);

    BEST_TIME(lauradoux_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(scalar_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(popcnt_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(table_bitset8_weight((uint8_t*)prec,size*8),expected,, repeat, size);
    BEST_TIME(table_bitset16_weight((uint16_t*)prec,size*4),expected,, repeat, size);
    BEST_TIME(unrolled_popcnt_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx2_bitset64_weight(prec,size),expected,, repeat, size);

    free(prec);
    printf("\n");
}

int main() {
    demo(16);
    demo(32);
    demo(64);
    demo(128);
    demo(1024);
    return 0;
}

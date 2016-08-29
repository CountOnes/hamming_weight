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
    for(int k = 0; k < size; ++k) prec[k]  = -k;

    int expected = scalar_bitset64_weight(prec,size);

    BEST_TIME(lauradoux_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(scalar_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(scalar_harley_seal8_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(scalar_harley_seal_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(table_bitset8_weight((uint8_t*)prec,size*8),expected,, repeat, size);
    BEST_TIME(table_bitset16_weight((uint16_t*)prec,size*4),expected,, repeat, size);
#if defined(HAVE_POPCNT_INSTRUCTION)
    BEST_TIME(popcnt_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(unrolled_popcnt_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(yee_popcnt_bitset64_weight(prec,size),expected,, repeat, size);
#else
    printf("no popcnt instruction\n");
#endif

    BEST_TIME(sse_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(sse_harley_seal_bitset64_weight(prec,size),expected,, repeat, size);
#if defined(HAVE_AVX2_INSTRUCTIONS)
    BEST_TIME(avx2_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx2_lookup_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx2_lauradoux_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx2_harley_seal_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx2_harley_seal_nate_bitset64_weight(prec,size),expected,, repeat, size);
    BEST_TIME(avx2_harley_seal_bitset64_weight_unrolled_twice(prec,size),expected,, repeat, size);
#else
    printf("no AVX2 instructions\n");
#endif
#if defined(HAVE_AVX512_INSTRUCTIONS)
    BEST_TIME(avx512_harley_seal(prec,size),    expected,, repeat, size);
    BEST_TIME(avx512_vpermb(prec,size),         expected,, repeat, size);
    BEST_TIME(avx512_vperm2b(prec,size),        expected,, repeat, size);
#else
#   if defined(HAVE_AVX512F_INSTRUCTIONS)
        BEST_TIME(avx512f_harley_seal(prec,size),   expected,, repeat, size);
        BEST_TIME(avx512f_gather(prec,size),        expected,, repeat, size);
#   endif
#   if defined(HAVE_AVX512CD_INSTRUCTIONS)
        BEST_TIME(avx512cd_naive(prec,size),   expected,, repeat, size);
#   endif
#endif

#if !defined(HAVE_AVX512_INSTRUCTIONS) && !defined(HAVE_AVX512F_INSTRUCTIONS) && !defined(HAVE_AVX512CD_INSTRUCTIONS)
    printf("no AVX512 instructions\n");
#endif

    free(prec);
    printf("\n");
}

int main() {

#if defined(HAVE_AVX512F_INSTRUCTIONS)
    avx512f_gather_init();
#endif

    for(int w = 8; w <= 8192; w *= 2) {
        demo(w);
        demo(w*3/2);
    }
    return 0;
}

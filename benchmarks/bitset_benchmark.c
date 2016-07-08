#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>


#include "benchmark.h"
#include "bitset.h"

void *aligned_malloc(size_t alignment, size_t size) {
    void *mem;
    if (posix_memalign(&mem, alignment, size)) exit(1);
    return mem;
}

#if defined(HAVE_AVX2_INSTRUCTIONS)

#include <x86intrin.h>

#define BITSET_CONTAINER_FN(opname, opsymbol)                \
int scalar_nocard_##opname(const uint64_t * restrict array_1,          \
        const uint64_t * restrict array_2,                             \
		  size_t length, uint64_t * restrict out) {                    \
  for (size_t i =  0; i < length; i ++) {                  \
   const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]); \
   out[i] = word_1;                                          \
  }                                                          \
  return 0;                                                   \
}
BITSET_CONTAINER_FN(and, &)

#undef BITSET_CONTAINER_FN

#define BITSET_CONTAINER_FN(opname, opsymbol, avx_intrinsic)             \
int avx_nocard_##opname(const uint64_t * restrict array_1,                        \
                              const uint64_t * restrict array_2,                  \
							  size_t length, uint64_t * restrict out) {           \
    const size_t m256length = length / 4;                               \
    for (size_t idx = 0; idx + 3 < m256length; idx += 4) {              \
        __m256i A1, A2, ymm1;                                     \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 0);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 0);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 0, ymm1);            \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 1);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 1);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 1, ymm1);            \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 2);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 2);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 2, ymm1);            \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 3);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 3);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 3, ymm1);            \
    }                                                                   \
    for (size_t i =  length - length % 16; i < length; i ++) {           \
            const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]);   \
            out[i] = word_1;                                            \
    }                                                                   \
    return 0;                                                 \
}
BITSET_CONTAINER_FN(and, &, _mm256_and_si256)

#undef BITSET_CONTAINER_FN

#endif

void demo(int size) {
    printf("size = %d words or %lu bytes \n",size,  size*sizeof(uint64_t));
    int repeat = 500;
    uint64_t * dataA = aligned_malloc(32,size * sizeof(uint64_t));
    uint64_t * dataB = aligned_malloc(32,size * sizeof(uint64_t));
    uint64_t * out = aligned_malloc(32,size * sizeof(uint64_t));

    for(int k = 0; k < size; ++k) {
    	dataA[k]  = -k;
    	dataB[k] = k;
    }
    int expected =  scalar_and(dataA, dataB, size, out);
    BEST_TIME_CHECK(memcpy(out, dataA, size * sizeof(uint64_t)), !memcmp (out, dataA, size * sizeof(uint64_t)),, repeat, size);
    BEST_TIME(scalar_and(dataA, dataB, size, out),expected,, repeat, size);
    BEST_TIME(scalar_nocard_and(dataA, dataB, size, out),0,, repeat, size);

#if defined(HAVE_POPCNT_INSTRUCTION)
    BEST_TIME(popcnt_and(dataA, dataB, size, out),expected,, repeat, size);
#else
    printf("no popcnt instruction\n");
#endif

#if defined(HAVE_AVX2_INSTRUCTIONS)
    BEST_TIME(avx_nocard_and(dataA, dataB, size, out),0,, repeat, size);
    BEST_TIME(avx_lookup_and(dataA, dataB, size, out),expected,, repeat, size);
    BEST_TIME(avx_harley_seal_and(dataA, dataB, size, out),expected,, repeat, size);
#else
    printf("no AVX2 instructions\n");
#endif

    free(dataA);
    free(dataB);
    free(out);
    printf("\n");
}

int main() {
    for(int w = 8; w <= 8192; w *= 2) {
      demo(w);
      demo(w*3/2);
    }
    return 0;
}




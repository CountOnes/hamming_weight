#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include "hamming_weight.h"

#define CHECK_VALUE(test, expected)                                 \
  do {                                                              \
    if((int)test != expected) {                                     \
      printf("%-40s\t: ", #test);                                   \
      printf (" line %d of file \"%s\" (function <%s>) ",          \
                      __LINE__, __FILE__, __func__);                 \
      printf("not expected  (%d , %d )\n",(int)test,expected);      \
      return false;                                                 \
    }                                                               \
 } while (0)



bool check_continuous(int size, int runstart, int runend) {
    uint64_t * prec = malloc(size * sizeof(uint64_t));
    memset(prec,0,size * sizeof(uint64_t));
    for(int i = runstart; i < runend; ++i) {
        prec[i/64] |= (UINT64_C(1)<<(i%64));
    }
    int expected = scalar_bitset64_weight(prec,size);
    CHECK_VALUE(lauradoux_bitset64_weight(prec,size),expected);
    CHECK_VALUE(scalar_bitset64_weight(prec,size),expected);
    CHECK_VALUE(scalar_harley_seal_bitset64_weight(prec,size),expected);
    CHECK_VALUE(table_bitset8_weight((uint8_t*)prec,size*8),expected);
    CHECK_VALUE(table_bitset16_weight((uint16_t*)prec,size*4),expected);
#if defined(HAVE_POPCNT_INSTRUCTION)
    CHECK_VALUE(popcnt_bitset64_weight(prec,size),expected);
    CHECK_VALUE(unrolled_popcnt_bitset64_weight(prec,size),expected);
    CHECK_VALUE(yee_popcnt_bitset64_weight(prec,size),expected);
#endif
#if defined(HAVE_AVX2_INSTRUCTIONS)
    CHECK_VALUE(avx2_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_lookup_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_lauradoux_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_harley_seal_bitset64_weight(prec,size),expected);
#endif
#if defined(HAVE_AVX512_INSTRUCTIONS)
    CHECK_VALUE(avx512_harley_seal(prec,size),    expected);
    CHECK_VALUE(avx512_vpermb(prec,size),         expected);
    CHECK_VALUE(avx512_vperm2b(prec,size),        expected);
#endif
    free(prec);
    return true;
}

bool check_constant(int size, uint8_t w) {
    uint64_t * prec = malloc(size * sizeof(uint64_t));
    memset(prec,w,size * sizeof(uint64_t));
    int expected = scalar_bitset64_weight(prec,size);
    CHECK_VALUE(lauradoux_bitset64_weight(prec,size),expected);
    CHECK_VALUE(scalar_bitset64_weight(prec,size),expected);
    CHECK_VALUE(scalar_harley_seal_bitset64_weight(prec,size),expected);
    CHECK_VALUE(table_bitset8_weight((uint8_t*)prec,size*8),expected);
    CHECK_VALUE(table_bitset16_weight((uint16_t*)prec,size*4),expected);
#if defined(HAVE_POPCNT_INSTRUCTION)
    CHECK_VALUE(popcnt_bitset64_weight(prec,size),expected);
    CHECK_VALUE(unrolled_popcnt_bitset64_weight(prec,size),expected);
    CHECK_VALUE(yee_popcnt_bitset64_weight(prec,size),expected);
#endif
#if defined(HAVE_AVX2_INSTRUCTIONS)
    CHECK_VALUE(avx2_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_lookup_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_lauradoux_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_harley_seal_bitset64_weight(prec,size),expected);
#endif
#if defined(HAVE_AVX512_INSTRUCTIONS)
    CHECK_VALUE(avx512_harley_seal(prec,size),    expected);
    CHECK_VALUE(avx512_vpermb(prec,size),         expected);
    CHECK_VALUE(avx512_vperm2b(prec,size),        expected);
#endif
    free(prec);
    return true;
}


int main() {
    for(int w = 8; w <= 8192; w = 2 * w - 1) {
        printf(".");
        fflush(stdout);
        for(uint64_t value = 0; value < 256; value += 1) {
            if(!check_constant(w,value)) return -1;
        }
    }
    printf("\n");
    for(int w = 8; w <= 8192; w = 2 * w - 1) {
        printf(".");
        fflush(stdout);
        for(int runstart = 0; runstart < w * (int) sizeof(uint64_t); runstart+= w / 33 + 1) {
            for(int endstart = runstart; endstart < w * (int) sizeof(uint64_t); endstart += w / 11 + 1) {
                if(!check_continuous(w,runstart,endstart)) return -1;
            }
        }
    }
    printf("\n");
    printf("Code looks ok.\n");
    return 0;
}

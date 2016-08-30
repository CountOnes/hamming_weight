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
      printf (" line %d of file \"%s\" (function <%s>) ",           \
                      __LINE__, __FILE__, __func__);                \
      printf("result: %d, expected: %d\n",(int)test,expected);      \
      return false;                                                 \
    }                                                               \
 } while (0)

static bool check(uint64_t * prec, int size) {
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
    CHECK_VALUE(sse_bitset64_weight(prec,size),expected);
    CHECK_VALUE(sse_harley_seal_bitset64_weight(prec,size),expected);
#if defined(HAVE_AVX2_INSTRUCTIONS)
    CHECK_VALUE(avx2_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_lookup_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_lauradoux_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_harley_seal_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_harley_seal_nate_bitset64_weight(prec,size),expected);
    CHECK_VALUE(avx2_harley_seal_bitset64_weight_unrolled_twice(prec,size),expected);
#endif
#if defined(HAVE_AVX512_INSTRUCTIONS)
    CHECK_VALUE(avx512_harley_seal(prec,size),    expected);
    CHECK_VALUE(avx512_vpermb(prec,size),         expected);
    CHECK_VALUE(avx512_vperm2b(prec,size),        expected);
#endif
#if defined(HAVE_AVX512F_INSTRUCTIONS)
    CHECK_VALUE(avx512f_harley_seal(prec,size),                    expected);
    CHECK_VALUE(avx512f_harley_seal__hardware_popcnt(prec,size),   expected);
    CHECK_VALUE(avx512f_harley_seal__hardware_popcnt_2(prec,size), expected);
    CHECK_VALUE(avx512f_gather(prec,size),                         expected);
#endif
#if defined(HAVE_AVX512CD_INSTRUCTIONS)
    CHECK_VALUE(avx512cd_naive(prec,size),   expected);
#endif
    return true;
}

void *aligned_malloc(size_t alignment, size_t size) {
    void *mem;
    if (posix_memalign(&mem, alignment, size)) exit(1);
    return mem;
}

bool check_continuous(int size, int runstart, int runend) {
    uint64_t * prec = aligned_malloc(64, size * sizeof(uint64_t));
    memset(prec,0,size * sizeof(uint64_t));
    for(int i = runstart; i < runend; ++i) {
        prec[i/64] |= (UINT64_C(1)<<(i%64));
    }
    bool answer = check(prec,size);
    free(prec);
    return answer;
}

bool check_constant(int size, uint8_t w) {
    uint64_t * prec = aligned_malloc(64, size * sizeof(uint64_t));
    memset(prec,w,size * sizeof(uint64_t));
    bool answer = check(prec,size);
    free(prec);
    return answer;
}

bool check_step(int size, int step) {
    uint64_t * prec = aligned_malloc(64, size * sizeof(uint64_t));
    memset(prec,0,size * sizeof(uint64_t));
    for(int i = 0; i < size * (int) sizeof(uint64_t); i+= step) {
        prec[i/64] |= (UINT64_C(1)<<(i%64));
    }
    bool answer = check(prec,size);
    free(prec);
    return answer;
}

bool check_exponential_step(int size, int start) {
    uint64_t * prec = aligned_malloc(64, size * sizeof(uint64_t));
    memset(prec,0,size * sizeof(uint64_t));
    for(int i = start + 1; i < size * (int) sizeof(uint64_t); i+= i) {
        prec[i/64] |= (UINT64_C(1)<<(i%64));
    }
    bool answer = check(prec,size);
    free(prec);
    return answer;
}

int main() {

#if defined(HAVE_AVX512F_INSTRUCTIONS)
    avx512f_gather_init();
#endif

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
	for (int w = 8; w <= 8192; w = 2 * w - 1) {
		printf(".");
		fflush(stdout);
		for (int step = 1; step < w * (int) sizeof(uint64_t);
				step += w / 33 + 1) {
			if (!check_step(w, step))
				return -1;

		}
	}
	printf("\n");
	for (int w = 8; w <= 8192; w = 2 * w - 1) {
		printf(".");
		fflush(stdout);
		for (int start = 0; start < w * (int) sizeof(uint64_t);
				start += w / 11 + 1) {
			if (!check_exponential_step(w, start))
				return -1;

		}
	}
	printf("\n");
	printf("Code looks ok.\n");
    return 0;
}

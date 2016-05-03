#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

#ifdef HAVE_AVX2_INSTRUCTIONS


#include "avx_jaccard_index.h"


void avx2_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* sum, uint64_t* inters) {
    size_t i = 0;
    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc_sum = _mm256_setzero_si256();
    __m256i acc_int = _mm256_setzero_si256();

#define ITER { \
        const __m256i vecA = _mm256_loadu_si256((const __m256i*)(dataA + i)); \
        const __m256i vecB = _mm256_loadu_si256((const __m256i*)(dataB + i)); \
        const __m256i vec_sum = _mm256_or_si256(vecA, vecB); \
        const __m256i vec_int = _mm256_and_si256(vecA, vecB); \
        const __m256i lo_sum  = _mm256_and_si256(vec_sum, low_mask); \
        const __m256i lo_int  = _mm256_and_si256(vec_int, low_mask); \
        const __m256i hi_sum  = _mm256_and_si256(_mm256_srli_epi16(vec_sum, 4), low_mask); \
        const __m256i hi_int  = _mm256_and_si256(_mm256_srli_epi16(vec_int, 4), low_mask); \
        const __m256i popcnt1_sum = _mm256_shuffle_epi8(lookup, lo_sum); \
        const __m256i popcnt1_int = _mm256_shuffle_epi8(lookup, lo_int); \
        const __m256i popcnt2_sum = _mm256_shuffle_epi8(lookup, hi_sum); \
        const __m256i popcnt2_int = _mm256_shuffle_epi8(lookup, hi_int); \
        local_sum = _mm256_add_epi8(local_sum, popcnt1_sum); \
        local_sum = _mm256_add_epi8(local_sum, popcnt2_sum); \
        local_int = _mm256_add_epi8(local_int, popcnt1_int); \
        local_int = _mm256_add_epi8(local_int, popcnt2_int); \
        i += 4; \
    }

    __m256i local_sum;
    __m256i local_int;

    while (i + 8*4 <= n) {
        local_sum = _mm256_setzero_si256();
        local_int = _mm256_setzero_si256();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc_sum = _mm256_add_epi64(acc_sum, _mm256_sad_epu8(local_sum, _mm256_setzero_si256()));
        acc_int = _mm256_add_epi64(acc_int, _mm256_sad_epu8(local_int, _mm256_setzero_si256()));
    }

    local_sum = _mm256_setzero_si256();
    local_int = _mm256_setzero_si256();

    while (i + 4 <= n) {
        ITER;
    }

    acc_sum = _mm256_add_epi64(acc_sum, _mm256_sad_epu8(local_sum, _mm256_setzero_si256()));
    acc_int = _mm256_add_epi64(acc_int, _mm256_sad_epu8(local_int, _mm256_setzero_si256()));

#undef ITER

    *sum = 0;
    *inters = 0;

    *sum += (uint64_t)(_mm256_extract_epi64(acc_sum, 0));
    *sum += (uint64_t)(_mm256_extract_epi64(acc_sum, 1));
    *sum += (uint64_t)(_mm256_extract_epi64(acc_sum, 2));
    *sum += (uint64_t)(_mm256_extract_epi64(acc_sum, 3));

    *inters += (uint64_t)(_mm256_extract_epi64(acc_int, 0));
    *inters += (uint64_t)(_mm256_extract_epi64(acc_int, 1));
    *inters += (uint64_t)(_mm256_extract_epi64(acc_int, 2));
    *inters += (uint64_t)(_mm256_extract_epi64(acc_int, 3));

    for (/**/; i < n; i++) {
        *sum    += _mm_popcnt_u64(dataA[i] | dataB[i]);
        *inters += _mm_popcnt_u64(dataA[i] & dataB[i]);
    }
}

#endif // HAVE_AVX2_INSTRUCTIONS

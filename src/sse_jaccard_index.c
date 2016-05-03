#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#include "sse_jaccard_index.h"
#include "popcnt_hamming_weight.h"


/*
    Procedure calculates at the same time popcount(A & B) and popcount(A | B)
*/
void sse_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* sum, uint64_t* inters) {
    size_t i = 0;
    const __m128i lookup = _mm_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m128i low_mask = _mm_set1_epi8(0x0f);

    __m128i acc_sum = _mm_setzero_si128();
    __m128i acc_int = _mm_setzero_si128();

#define ITER { \
        const __m128i vecA = _mm_loadu_si128((const __m128i*)(dataA + i)); \
        const __m128i vecB = _mm_loadu_si128((const __m128i*)(dataB + i)); \
        const __m128i vec_sum = _mm_or_si128(vecA, vecB); \
        const __m128i vec_int = _mm_and_si128(vecA, vecB); \
        const __m128i lo_sum  = _mm_and_si128(vec_sum, low_mask); \
        const __m128i lo_int  = _mm_and_si128(vec_int, low_mask); \
        const __m128i hi_sum  = _mm_and_si128(_mm_srli_epi16(vec_sum, 4), low_mask); \
        const __m128i hi_int  = _mm_and_si128(_mm_srli_epi16(vec_int, 4), low_mask); \
        const __m128i popcnt1_sum = _mm_shuffle_epi8(lookup, lo_sum); \
        const __m128i popcnt1_int = _mm_shuffle_epi8(lookup, lo_int); \
        const __m128i popcnt2_sum = _mm_shuffle_epi8(lookup, hi_sum); \
        const __m128i popcnt2_int = _mm_shuffle_epi8(lookup, hi_int); \
        local_sum = _mm_add_epi8(local_sum, popcnt1_sum); \
        local_sum = _mm_add_epi8(local_sum, popcnt2_sum); \
        local_int = _mm_add_epi8(local_int, popcnt1_int); \
        local_int = _mm_add_epi8(local_int, popcnt2_int); \
        i += 2; \
    }

    __m128i local_sum;
    __m128i local_int;

    while (i + 8*2 <= n) {
        local_sum = _mm_setzero_si128();
        local_int = _mm_setzero_si128();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc_sum = _mm_add_epi64(acc_sum, _mm_sad_epu8(local_sum, _mm_setzero_si128()));
        acc_int = _mm_add_epi64(acc_int, _mm_sad_epu8(local_int, _mm_setzero_si128()));
    }

    local_int = _mm_setzero_si128();
    local_sum = _mm_setzero_si128();

    while (i + 2 <= n) {
        ITER;
    }

    acc_sum = _mm_add_epi64(acc_sum, _mm_sad_epu8(local_sum, _mm_setzero_si128()));
    acc_int = _mm_add_epi64(acc_int, _mm_sad_epu8(local_int, _mm_setzero_si128()));

#undef ITER

    *sum     = (uint64_t)(_mm_extract_epi64(acc_sum, 0));
    *sum    += (uint64_t)(_mm_extract_epi64(acc_sum, 1));

    *inters  = (uint64_t)(_mm_extract_epi64(acc_int, 0));
    *inters += (uint64_t)(_mm_extract_epi64(acc_int, 1));

    for (/**/; i < n; i++) {
        const uint64_t a = dataA[i];
        const uint64_t b = dataB[i];

        *sum    += _mm_popcnt_u64(a | b);
        *inters += _mm_popcnt_u64(a & b);
    }
}

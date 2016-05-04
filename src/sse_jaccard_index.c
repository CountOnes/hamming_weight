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
void sse_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* j_union, uint64_t* j_inter) {
    size_t i = 0;
    const __m128i lookup = _mm_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m128i low_mask = _mm_set1_epi8(0x0f);

    __m128i acc_union = _mm_setzero_si128();
    __m128i acc_inter = _mm_setzero_si128();

#define ITER { \
        const __m128i vecA = _mm_loadu_si128((const __m128i*)(dataA + i)); \
        const __m128i vecB = _mm_loadu_si128((const __m128i*)(dataB + i)); \
        const __m128i vec_union = _mm_or_si128(vecA, vecB); \
        const __m128i vec_inter = _mm_and_si128(vecA, vecB); \
        const __m128i lo_union  = _mm_and_si128(vec_union, low_mask); \
        const __m128i lo_inter  = _mm_and_si128(vec_inter, low_mask); \
        const __m128i hi_union  = _mm_and_si128(_mm_srli_epi16(vec_union, 4), low_mask); \
        const __m128i hi_inter  = _mm_and_si128(_mm_srli_epi16(vec_inter, 4), low_mask); \
        const __m128i popcnt1_union = _mm_shuffle_epi8(lookup, lo_union); \
        const __m128i popcnt1_inter = _mm_shuffle_epi8(lookup, lo_inter); \
        const __m128i popcnt2_union = _mm_shuffle_epi8(lookup, hi_union); \
        const __m128i popcnt2_inter = _mm_shuffle_epi8(lookup, hi_inter); \
        local_union = _mm_add_epi8(local_union, popcnt1_union); \
        local_union = _mm_add_epi8(local_union, popcnt2_union); \
        local_inter = _mm_add_epi8(local_inter, popcnt1_inter); \
        local_inter = _mm_add_epi8(local_inter, popcnt2_inter); \
        i += 2; \
    }

    __m128i local_union;
    __m128i local_inter;

    while (i + 8*2 <= n) {
        local_union = _mm_setzero_si128();
        local_inter = _mm_setzero_si128();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc_union = _mm_add_epi64(acc_union, _mm_sad_epu8(local_union, _mm_setzero_si128()));
        acc_inter = _mm_add_epi64(acc_inter, _mm_sad_epu8(local_inter, _mm_setzero_si128()));
    }

    local_inter = _mm_setzero_si128();
    local_union = _mm_setzero_si128();

    while (i + 2 <= n) {
        ITER;
    }

    acc_union = _mm_add_epi64(acc_union, _mm_sad_epu8(local_union, _mm_setzero_si128()));
    acc_inter = _mm_add_epi64(acc_inter, _mm_sad_epu8(local_inter, _mm_setzero_si128()));

#undef ITER

    *j_union  = (uint64_t)(_mm_extract_epi64(acc_union, 0));
    *j_union += (uint64_t)(_mm_extract_epi64(acc_union, 1));

    *j_inter  = (uint64_t)(_mm_extract_epi64(acc_inter, 0));
    *j_inter += (uint64_t)(_mm_extract_epi64(acc_inter, 1));

    for (/**/; i < n; i++) {
        const uint64_t a = dataA[i];
        const uint64_t b = dataB[i];

        *j_union += _mm_popcnt_u64(a | b);
        *j_inter += _mm_popcnt_u64(a & b);
    }
}

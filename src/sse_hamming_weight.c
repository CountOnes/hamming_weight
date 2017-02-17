#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#include "sse_hamming_weight.h"
#include "popcnt_hamming_weight.h"


int sse_bitset64_weight(const uint64_t* data, size_t n) {
    size_t i = 0;
    const __m128i lookup = _mm_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m128i low_mask = _mm_set1_epi8(0x0f);

    __m128i acc = _mm_setzero_si128();

#define ITER { \
        const __m128i vec = _mm_loadu_si128((const __m128i*)(data + i)); \
        const __m128i lo  = _mm_and_si128(vec, low_mask); \
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(vec, 4), low_mask); \
        const __m128i popcnt1 = _mm_shuffle_epi8(lookup, lo); \
        const __m128i popcnt2 = _mm_shuffle_epi8(lookup, hi); \
        local = _mm_add_epi8(local, popcnt1); \
        local = _mm_add_epi8(local, popcnt2); \
        i += 2; \
    }

    while (i + 8*2 <= n) {
        __m128i local = _mm_setzero_si128();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc = _mm_add_epi64(acc, _mm_sad_epu8(local, _mm_setzero_si128()));
    }

    __m128i local = _mm_setzero_si128();

    while (i + 2 <= n) {
        ITER;
    }

    acc = _mm_add_epi64(acc, _mm_sad_epu8(local, _mm_setzero_si128()));

#undef ITER

    uint64_t result = 0;

    result += (uint64_t)(_mm_extract_epi64(acc, 0));
    result += (uint64_t)(_mm_extract_epi64(acc, 1));

    for (/**/; i < n; i++) {
        result += _mm_popcnt_u64(data[i]);
    }
    return result;
}

// Morancho focuses on 32 bytes, 48 bytes....4 popcnt, 3 sse
int sse_morancho_bitset64_weight(const uint64_t * data, size_t n) {
    size_t i = 0;
    const __m128i lookup = _mm_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m128i low_mask = _mm_set1_epi8(0x0f);

    __m128i acc = _mm_setzero_si128();

#define ITER { \
        const __m128i vec = _mm_loadu_si128((const __m128i*)(data + i)); \
        const __m128i lo  = _mm_and_si128(vec, low_mask); \
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(vec, 4), low_mask); \
        const __m128i popcnt1 = _mm_shuffle_epi8(lookup, lo); \
        const __m128i popcnt2 = _mm_shuffle_epi8(lookup, hi); \
        local = _mm_add_epi8(local, popcnt1); \
        local = _mm_add_epi8(local, popcnt2); \
        i += 2; \
    }
    uint64_t result = 0;

    while (i + 10 <= n) {
        __m128i local = _mm_setzero_si128();
        result += _mm_popcnt_u64(data[i]);
        result += _mm_popcnt_u64(data[i + 1]);
        result += _mm_popcnt_u64(data[i + 2]);
        result += _mm_popcnt_u64(data[i + 3]);
        i += 4;
        ITER ITER ITER
        acc = _mm_add_epi64(acc, _mm_sad_epu8(local, _mm_setzero_si128()));
    }

    __m128i local = _mm_setzero_si128();

    while (i + 2 <= n) {
        ITER;
    }

    acc = _mm_add_epi64(acc, _mm_sad_epu8(local, _mm_setzero_si128()));

#undef ITER


    result += (uint64_t)(_mm_extract_epi64(acc, 0));
    result += (uint64_t)(_mm_extract_epi64(acc, 1));

    for (/**/; i < n; i++) {
        result += _mm_popcnt_u64(data[i]);
    }
    return result;
}

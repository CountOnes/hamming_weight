#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#ifdef HAVE_XOP_INSTRUCTIONS

#include "xop_hamming_weight.h"


int xop_bitset64_weight(const uint64_t* data, size_t n) {
    size_t i = 0;
    const __m128i lookup_lo = _mm_setr_epi8( // popcnt(0 .. 15)
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m128i lookup_hi = _mm_setr_epi8( // popcnt(16 .. 31)
        /* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
        /* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
        /* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
        /* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5
    );

    __m128i acc = _mm_setzero_si128();

#define ITER { \
        /* vec0 = packed_byte(abcd_efgh) -- a .. p are bits */ \
        /* vec1 = packed_byte(ijkl_mnop) */ \
        const __m128i vec0 = _mm_loadu_si128((const __m128i*)(data + i)); \
        const __m128i vec1 = _mm_loadu_si128((const __m128i*)(data + i + 2)); \
        \
        /* t0   = packed_byte(...d_efgh) */ \
        const __m128i t0 = _mm_and_si128(vec0, _mm_set1_epi8(0x1f)); \
        /* t1   = packed_byte(...l_mnop) */ \
        const __m128i t1 = _mm_and_si128(vec1, _mm_set1_epi8(0x1f)); \
        /* t2   = packed_byte(??ij_klmn) */ \
        /*                      ^^ ^     */ \
        const __m128i t2 = _mm_srli_epi16(vec1, 2); \
        /* t3   = packed_byte(...._.abc) */ \
        const __m128i t3 = _mm_and_si128(_mm_srli_epi16(vec0, 5), _mm_set1_epi8(0x07)); \
        \
        /* popcnt1 = popcount of bits defgh */ \
        const __m128i popcnt1 = _mm_perm_epi8(lookup_lo, lookup_hi, t0); \
        /* popcnt2 = popcount of bits lmnop */ \
        const __m128i popcnt2 = _mm_perm_epi8(lookup_lo, lookup_hi, t1); \
        \
        /* t4   = packed_byte(...j_kabc) */ \
        const __m128i t4 = _mm_cmov_si128(t2, t3, _mm_set1_epi8(0x18)); \
        /* popcnt3 = popcount of bits jkabc */ \
        const __m128i popcnt3 = _mm_perm_epi8(lookup_lo, lookup_hi, t4); \
        \
        /* five lower bits of local_a stores popcount of 15 bits */ \
        local_a = _mm_add_epi8(local_a, popcnt1); \
        local_a = _mm_add_epi8(local_a, popcnt2); \
        local_a = _mm_add_epi8(local_a, popcnt3); \
        /* three higher bits of local_b stores popcount of one bit */ \
        local_b = _mm_add_epi8(local_b, _mm_and_si128(t2, _mm_set1_epi8(0x20))); \
        i += 4; \
    }

    while (i + 6*4 <= n) {
        __m128i local_a = _mm_setzero_si128();
        __m128i local_b = _mm_setzero_si128();

        ITER ITER
        ITER ITER
        ITER ITER

        local_b = _mm_srli_epi16(local_b, 5);
        const __m128i local = _mm_add_epi8(local_a, local_b);
        acc = _mm_add_epi64(acc, _mm_sad_epu8(local, _mm_setzero_si128()));
    }

#undef ITER

    uint64_t result = 0;

    result += (uint64_t)(_mm_extract_epi64(acc, 0));
    result += (uint64_t)(_mm_extract_epi64(acc, 1));

    for (/**/; i < n; i++) {
        result += _mm_popcnt_u64(data[i]);
    }
    return result;
}


#endif // HAVE_XOP_INSTRUCTIONS

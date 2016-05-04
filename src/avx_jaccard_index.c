#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

#ifdef HAVE_AVX2_INSTRUCTIONS


#include "avx_jaccard_index.h"


void avx2_jaccard_index_lookup(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* sum, uint64_t* inters) {
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


// code copied from avx_harley_seal_hamming_weight.c

static __m256i popcount(__m256i v) {

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

    const __m256i lo  = _mm256_and_si256(v, low_mask);
    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);

    return _mm256_sad_epu8(_mm256_add_epi8(popcnt1, popcnt2), _mm256_setzero_si256());
}


static inline void CSA(__m256i* h, __m256i* l, __m256i a, __m256i b, __m256i c) {
  const __m256i u = _mm256_xor_si256(a , b);
  *h = _mm256_or_si256(_mm256_and_si256(a , b) , _mm256_and_si256(u , c) );
  *l = _mm256_xor_si256(u , c);
}


// looks complicated, but it's just a code repeated twice
static void jaccard_index(const __m256i* dataA, const __m256i* dataB, const uint64_t size, uint64_t* j_union, uint64_t* j_inter) {
  __m256i total_inter     = _mm256_setzero_si256();
  __m256i ones_inter      = _mm256_setzero_si256();
  __m256i twos_inter      = _mm256_setzero_si256();
  __m256i fours_inter     = _mm256_setzero_si256();
  __m256i eights_inter    = _mm256_setzero_si256();
  __m256i sixteens_inter  = _mm256_setzero_si256();
  __m256i twosA_inter, twosB_inter, foursA_inter, foursB_inter, eightsA_inter, eightsB_inter;

  __m256i total_union     = _mm256_setzero_si256();
  __m256i ones_union      = _mm256_setzero_si256();
  __m256i twos_union      = _mm256_setzero_si256();
  __m256i fours_union     = _mm256_setzero_si256();
  __m256i eights_union    = _mm256_setzero_si256();
  __m256i sixteens_union  = _mm256_setzero_si256();
  __m256i twosA_union, twosB_union, foursA_union, foursB_union, eightsA_union, eightsB_union;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for(; i < limit; i += 16) {
    __m256i a0 = _mm256_lddqu_si256(dataA + i);
    __m256i b0 = _mm256_lddqu_si256(dataB + i);
    __m256i a1 = _mm256_lddqu_si256(dataA + i + 1);
    __m256i b1 = _mm256_lddqu_si256(dataB + i + 1);
    CSA(&twosA_inter, &ones_inter, ones_inter, _mm256_and_si256(a0, b0), _mm256_and_si256(a1, b1));
    CSA(&twosA_union, &ones_union, ones_union, _mm256_or_si256(a0, b0), _mm256_or_si256(a1, b1));

    __m256i a2 = _mm256_lddqu_si256(dataA + i + 2);
    __m256i b2 = _mm256_lddqu_si256(dataB + i + 2);
    __m256i a3 = _mm256_lddqu_si256(dataA + i + 3);
    __m256i b3 = _mm256_lddqu_si256(dataB + i + 3);
    CSA(&twosB_inter, &ones_inter, ones_inter, _mm256_and_si256(a2, b2), _mm256_and_si256(a3, b3));
    CSA(&foursA_inter, &twos_inter, twos_inter, twosA_inter, twosB_inter);
    CSA(&twosB_union, &ones_union, ones_union, _mm256_or_si256(a2, b2), _mm256_or_si256(a3, b3));
    CSA(&foursA_union, &twos_union, twos_union, twosA_union, twosB_union);

    __m256i a4 = _mm256_lddqu_si256(dataA + i + 4);
    __m256i b4 = _mm256_lddqu_si256(dataB + i + 4);
    __m256i a5 = _mm256_lddqu_si256(dataA + i + 5);
    __m256i b5 = _mm256_lddqu_si256(dataB + i + 5);
    CSA(&twosA_inter, &ones_inter, ones_inter, _mm256_and_si256(a4, b4), _mm256_and_si256(a5, b5));
    CSA(&twosA_union, &ones_union, ones_union, _mm256_or_si256(a4, b4), _mm256_or_si256(a5, b5));

    __m256i a6 = _mm256_lddqu_si256(dataA + i + 6);
    __m256i b6 = _mm256_lddqu_si256(dataB + i + 6);
    __m256i a7 = _mm256_lddqu_si256(dataA + i + 7);
    __m256i b7 = _mm256_lddqu_si256(dataB + i + 7);
    CSA(&twosB_inter, &ones_inter, ones_inter, _mm256_and_si256(a6, b6), _mm256_and_si256(a7, b7));
    CSA(&foursB_inter,& twos_inter, twos_inter, twosA_inter, twosB_inter);
    CSA(&eightsA_inter,&fours_inter, fours_inter, foursA_inter, foursB_inter);
    CSA(&twosB_union, &ones_union, ones_union, _mm256_or_si256(a6, b6), _mm256_or_si256(a7, b7));
    CSA(&foursB_union,& twos_union, twos_union, twosA_union, twosB_union);
    CSA(&eightsA_union,&fours_union, fours_union, foursA_union, foursB_union);

    __m256i a8 = _mm256_lddqu_si256(dataA + i + 8);
    __m256i b8 = _mm256_lddqu_si256(dataB + i + 8);
    __m256i a9 = _mm256_lddqu_si256(dataA + i + 9);
    __m256i b9 = _mm256_lddqu_si256(dataB + i + 9);
    CSA(&twosA_inter, &ones_inter, ones_inter, _mm256_and_si256(a8, b8), _mm256_and_si256(a9, b9));
    CSA(&twosA_union, &ones_union, ones_union, _mm256_or_si256(a8, b8), _mm256_or_si256(a9, b9));

    __m256i a10 = _mm256_lddqu_si256(dataA + i + 10);
    __m256i b10 = _mm256_lddqu_si256(dataB + i + 10);
    __m256i a11 = _mm256_lddqu_si256(dataA + i + 11);
    __m256i b11 = _mm256_lddqu_si256(dataB + i + 11);
    CSA(&twosB_inter, &ones_inter, ones_inter, _mm256_and_si256(a10, b10), _mm256_and_si256(a11, b11));
    CSA(&foursA_inter, &twos_inter, twos_inter, twosA_inter, twosB_inter);
    CSA(&twosB_union, &ones_union, ones_union, _mm256_or_si256(a10, b10), _mm256_or_si256(a11, b11));
    CSA(&foursA_union, &twos_union, twos_union, twosA_union, twosB_union);

    __m256i a12 = _mm256_lddqu_si256(dataA + i + 12);
    __m256i b12 = _mm256_lddqu_si256(dataB + i + 12);
    __m256i a13 = _mm256_lddqu_si256(dataA + i + 13);
    __m256i b13 = _mm256_lddqu_si256(dataB + i + 13);
    CSA(&twosA_inter, &ones_inter, ones_inter, _mm256_and_si256(a12, b12), _mm256_and_si256(a13, b13));
    CSA(&twosA_union, &ones_union, ones_union, _mm256_or_si256(a12, b12), _mm256_or_si256(a13, b13));

    __m256i a14 = _mm256_lddqu_si256(dataA + i + 14);
    __m256i b14 = _mm256_lddqu_si256(dataB + i + 14);
    __m256i a15 = _mm256_lddqu_si256(dataA + i + 15);
    __m256i b15 = _mm256_lddqu_si256(dataB + i + 15);
    CSA(&twosB_inter, &ones_inter, ones_inter, _mm256_and_si256(a14, b14), _mm256_and_si256(a15, b15));
    CSA(&foursB_inter, &twos_inter, twos_inter, twosA_inter, twosB_inter);
    CSA(&eightsB_inter, &fours_inter, fours_inter, foursA_inter, foursB_inter);
    CSA(&sixteens_inter, &eights_inter, eights_inter, eightsA_inter, eightsB_inter);
    CSA(&twosB_union, &ones_union, ones_union, _mm256_or_si256(a14, b14), _mm256_or_si256(a15, b15));
    CSA(&foursB_union, &twos_union, twos_union, twosA_union, twosB_union);
    CSA(&eightsB_union, &fours_union, fours_union, foursA_union, foursB_union);
    CSA(&sixteens_union, &eights_union, eights_union, eightsA_union, eightsB_union);

    total_inter = _mm256_add_epi64(total_inter, popcount(sixteens_inter));
    total_union = _mm256_add_epi64(total_union, popcount(sixteens_union));
  }

  total_inter = _mm256_slli_epi64(total_inter, 4);     // * 16
  total_inter = _mm256_add_epi64(total_inter, _mm256_slli_epi64(popcount(eights_inter), 3)); // += 8 * ...
  total_inter = _mm256_add_epi64(total_inter, _mm256_slli_epi64(popcount(fours_inter),  2)); // += 4 * ...
  total_inter = _mm256_add_epi64(total_inter, _mm256_slli_epi64(popcount(twos_inter),   1)); // += 2 * ...
  total_inter = _mm256_add_epi64(total_inter, popcount(ones_inter));

  total_union = _mm256_slli_epi64(total_union, 4);     // * 16
  total_union = _mm256_add_epi64(total_union, _mm256_slli_epi64(popcount(eights_union), 3)); // += 8 * ...
  total_union = _mm256_add_epi64(total_union, _mm256_slli_epi64(popcount(fours_union),  2)); // += 4 * ...
  total_union = _mm256_add_epi64(total_union, _mm256_slli_epi64(popcount(twos_union),   1)); // += 2 * ...
  total_union = _mm256_add_epi64(total_union, popcount(ones_union));
  for(; i < size; i++) {
    __m256i a = _mm256_lddqu_si256(dataA + i);
    __m256i b = _mm256_lddqu_si256(dataB + i);
    total_inter = _mm256_add_epi64(total_inter, popcount(_mm256_and_si256(a, b)));
    total_union = _mm256_add_epi64(total_union, popcount(_mm256_or_si256(a, b)));
  }

  *j_inter = (uint64_t)(_mm256_extract_epi64(total_inter, 0))
           + (uint64_t)(_mm256_extract_epi64(total_inter, 1))
           + (uint64_t)(_mm256_extract_epi64(total_inter, 2))
           + (uint64_t)(_mm256_extract_epi64(total_inter, 3));

  *j_union = (uint64_t)(_mm256_extract_epi64(total_union, 0))
           + (uint64_t)(_mm256_extract_epi64(total_union, 1))
           + (uint64_t)(_mm256_extract_epi64(total_union, 2))
           + (uint64_t)(_mm256_extract_epi64(total_union, 3));
}


void avx2_jaccard_index_harley_seal(const uint64_t* dataA, const uint64_t* dataB, size_t size, uint64_t* sum, uint64_t* inters) {
  const unsigned int wordspervector = sizeof(__m256i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  if(size >= minvit) {
    jaccard_index((const __m256i*) dataB, (const __m256i*) dataA, size / wordspervector, sum, inters);
    for (size_t i = size - size % wordspervector; i < size; i++) {
      *sum += _mm_popcnt_u64(dataA[i] | dataB[i]);
      *inters += _mm_popcnt_u64(dataA[i] & dataB[i]);
    }
    return;
  }
  *sum = 0;
  *inters = 0;
  for (size_t i = size - size % minvit; i < size; i++) {
    *sum += _mm_popcnt_u64(dataA[i] | dataB[i]);
    *inters += _mm_popcnt_u64(dataA[i] & dataB[i]);
  }
}


#endif // HAVE_AVX2_INSTRUCTIONS

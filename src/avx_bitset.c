#include "avx_bitset.h"

#include <stdint.h>
#include <stddef.h>
#include <x86intrin.h>

#ifdef HAVE_AVX2_INSTRUCTIONS

#define BITSET_CONTAINER_FN(opname, opsymbol, avx_intrinsic)             \
int avx_lookup_##opname(const uint64_t * restrict array_1,                        \
                              const uint64_t * restrict array_2,                  \
							  size_t length, uint64_t * restrict out) {           \
    const __m256i shuf =                                                \
       _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, \
                        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);\
    const __m256i  mask = _mm256_set1_epi8(0x0f);                       \
    __m256i total = _mm256_setzero_si256();                             \
    __m256i zero = _mm256_setzero_si256();                              \
    const size_t m256length = length / 4;                               \
    for (size_t idx = 0; idx + 3 < m256length; idx += 4) {              \
        __m256i A1, A2, ymm1, ymm2;                                     \
       __m256i innertotal = _mm256_setzero_si256();                     \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 0);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 0);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 0, ymm1);            \
        ymm2 = _mm256_srli_epi32(ymm1,4);                               \
        ymm1 = _mm256_and_si256(ymm1,mask);                             \
        ymm2 = _mm256_and_si256(ymm2,mask);                             \
        ymm1 = _mm256_shuffle_epi8(shuf,ymm1);                          \
        ymm2 = _mm256_shuffle_epi8(shuf,ymm2);                          \
        innertotal = _mm256_add_epi8(innertotal,ymm1);                  \
        innertotal = _mm256_add_epi8(innertotal,ymm2);                  \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 1);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 1);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 1, ymm1);            \
        ymm2 = _mm256_srli_epi32(ymm1,4);                               \
        ymm1 = _mm256_and_si256(ymm1,mask);                             \
        ymm2 = _mm256_and_si256(ymm2,mask);                             \
        ymm1 = _mm256_shuffle_epi8(shuf,ymm1);                          \
        ymm2 = _mm256_shuffle_epi8(shuf,ymm2);                          \
        innertotal = _mm256_add_epi8(innertotal,ymm1);                  \
        innertotal = _mm256_add_epi8(innertotal,ymm2);                  \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 2);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 2);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 2, ymm1);            \
        ymm2 = _mm256_srli_epi32(ymm1,4);                               \
        ymm1 = _mm256_and_si256(ymm1,mask);                             \
        ymm2 = _mm256_and_si256(ymm2,mask);                             \
        ymm1 = _mm256_shuffle_epi8(shuf,ymm1);                          \
        ymm2 = _mm256_shuffle_epi8(shuf,ymm2);                          \
        innertotal = _mm256_add_epi8(innertotal,ymm1);                  \
        innertotal = _mm256_add_epi8(innertotal,ymm2);                  \
        A1 = _mm256_lddqu_si256((__m256i *)array_1 + idx + 3);          \
        A2 = _mm256_lddqu_si256((__m256i *)array_2 + idx + 3);          \
        ymm1 = avx_intrinsic(A2, A1);                                   \
        _mm256_storeu_si256((__m256i *)out + idx + 3, ymm1);            \
        ymm2 = _mm256_srli_epi32(ymm1,4);                               \
        ymm1 = _mm256_and_si256(ymm1,mask);                             \
        ymm2 = _mm256_and_si256(ymm2,mask);                             \
        ymm1 = _mm256_shuffle_epi8(shuf,ymm1);                          \
        ymm2 = _mm256_shuffle_epi8(shuf,ymm2);                          \
        innertotal = _mm256_add_epi8(innertotal,ymm1);                  \
        innertotal = _mm256_add_epi8(innertotal,ymm2);                  \
        innertotal = _mm256_sad_epu8(zero,innertotal);                  \
        total= _mm256_add_epi64(total,innertotal);                      \
    }                                                                   \
    int cardinality = _mm256_extract_epi64(total,0) +                   \
        _mm256_extract_epi64(total,1) +                                 \
        _mm256_extract_epi64(total,2) +                                 \
        _mm256_extract_epi64(total,3);                                  \
    for (size_t i =  length - length % 16; i < length; i ++) {           \
            const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]);   \
            out[i] = word_1;                                            \
            cardinality += _mm_popcnt_u64(word_1);                      \
    }                                                                   \
    return cardinality;                                                 \
}



BITSET_CONTAINER_FN(and, &, _mm256_and_si256)

#undef BITSET_CONTAINER_FN







static __m256i popcountnate(__m256i v) {

    const __m256i lookuppos = _mm256_setr_epi8(
        /* 0 */ 4 + 0, /* 1 */ 4 + 1, /* 2 */ 4 + 1, /* 3 */ 4 + 2,
        /* 4 */ 4 + 1, /* 5 */ 4 + 2, /* 6 */ 4 + 2, /* 7 */ 4 + 3,
        /* 8 */ 4 + 1, /* 9 */ 4 + 2, /* a */ 4 + 2, /* b */ 4 + 3,
        /* c */ 4 + 2, /* d */ 4 + 3, /* e */ 4 + 3, /* f */ 4 + 4,

        /* 0 */ 4 + 0, /* 1 */ 4 + 1, /* 2 */ 4 + 1, /* 3 */ 4 + 2,
        /* 4 */ 4 + 1, /* 5 */ 4 + 2, /* 6 */ 4 + 2, /* 7 */ 4 + 3,
        /* 8 */ 4 + 1, /* 9 */ 4 + 2, /* a */ 4 + 2, /* b */ 4 + 3,
        /* c */ 4 + 2, /* d */ 4 + 3, /* e */ 4 + 3, /* f */ 4 + 4
    );
    const __m256i lookupneg = _mm256_setr_epi8(
        /* 0 */ 4 - 0, /* 1 */ 4 - 1, /* 2 */ 4 - 1, /* 3 */ 4 - 2,
        /* 4 */ 4 - 1, /* 5 */ 4 - 2, /* 6 */ 4 - 2, /* 7 */ 4 - 3,
        /* 8 */ 4 - 1, /* 9 */ 4 - 2, /* a */ 4 - 2, /* b */ 4 - 3,
        /* c */ 4 - 2, /* d */ 4 - 3, /* e */ 4 - 3, /* f */ 4 - 4,

        /* 0 */ 4 - 0, /* 1 */ 4 - 1, /* 2 */ 4 - 1, /* 3 */ 4 - 2,
        /* 4 */ 4 - 1, /* 5 */ 4 - 2, /* 6 */ 4 - 2, /* 7 */ 4 - 3,
        /* 8 */ 4 - 1, /* 9 */ 4 - 2, /* a */ 4 - 2, /* b */ 4 - 3,
        /* c */ 4 - 2, /* d */ 4 - 3, /* e */ 4 - 3, /* f */ 4 - 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo  = _mm256_and_si256(v, low_mask);
    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookuppos, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookupneg, hi);
    return _mm256_sad_epu8(popcnt1, popcnt2);
}


static inline void CSA(__m256i* h, __m256i* l, __m256i a, __m256i b, __m256i c) {
  const __m256i u = _mm256_xor_si256(a , b);
  *h = _mm256_or_si256(_mm256_and_si256(a , b) , _mm256_and_si256(u , c) );
  *l = _mm256_xor_si256(u , c);
}

static uint64_t popcntnate_and(const __m256i* data1, const __m256i* data2,
		const uint64_t size, __m256i* out) {
  __m256i total     = _mm256_setzero_si256();
  __m256i ones      = _mm256_setzero_si256();
  __m256i twos      = _mm256_setzero_si256();
  __m256i fours     = _mm256_setzero_si256();
  __m256i eights    = _mm256_setzero_si256();
  __m256i sixteens  = _mm256_setzero_si256();
  __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for(; i < limit; i += 16) {
	__m256i dataA, dataB;
	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i), _mm256_lddqu_si256(data2 + i));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 1), _mm256_lddqu_si256(data2 + i + 1));
    _mm256_storeu_si256((__m256i *)out + i + 0, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 1, dataB);

	CSA(&twosA, &ones, ones, dataA, dataB);

	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 2), _mm256_lddqu_si256(data2 + i + 2));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 3), _mm256_lddqu_si256(data2 + i + 3));
    _mm256_storeu_si256((__m256i *)out + i + 2, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 3, dataB);


	CSA(&twosB, &ones, ones, dataA, dataB);
    CSA(&foursA, &twos, twos, twosA, twosB);

	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 4), _mm256_lddqu_si256(data2 + i + 4));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 5), _mm256_lddqu_si256(data2 + i + 5));
    _mm256_storeu_si256((__m256i *)out + i + 4, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 5, dataB);

    CSA(&twosA, &ones, ones, dataA, dataB);

	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 6), _mm256_lddqu_si256(data2 + i + 6));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 7), _mm256_lddqu_si256(data2 + i + 7));
    _mm256_storeu_si256((__m256i *)out + i + 6, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 7, dataB);

    CSA(&twosB, &ones, ones, dataA, dataB);
    CSA(&foursB,& twos, twos, twosA, twosB);
    CSA(&eightsA,&fours, fours, foursA, foursB);

	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 8), _mm256_lddqu_si256(data2 + i + 8));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 9), _mm256_lddqu_si256(data2 + i + 9));
    _mm256_storeu_si256((__m256i *)out + i + 8, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 9, dataB);

    CSA(&twosA, &ones, ones,  dataA, dataB);
	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 10), _mm256_lddqu_si256(data2 + i + 10));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 11), _mm256_lddqu_si256(data2 + i + 11));
    _mm256_storeu_si256((__m256i *)out + i + 10, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 11, dataB);

    CSA(&twosB, &ones, ones, dataA, dataB);
    CSA(&foursA, &twos, twos, twosA, twosB);
	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 12), _mm256_lddqu_si256(data2 + i + 12));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 13), _mm256_lddqu_si256(data2 + i + 13));
    _mm256_storeu_si256((__m256i *)out + i + 12, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 13, dataB);

    CSA(&twosA, &ones, ones, dataA, dataB);
	dataA = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 14), _mm256_lddqu_si256(data2 + i + 14));
	dataB = _mm256_and_si256(_mm256_lddqu_si256(data1 + i + 15), _mm256_lddqu_si256(data2 + i + 15));
    _mm256_storeu_si256((__m256i *)out + i + 14, dataA);
    _mm256_storeu_si256((__m256i *)out + i + 15, dataB);

    CSA(&twosB, &ones, ones, dataA, dataB);
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsB, &fours, fours, foursA, foursB);
    CSA(&sixteens, &eights, eights, eightsA, eightsB);

    total = _mm256_add_epi64(total, popcountnate(sixteens));
  }

  total = _mm256_slli_epi64(total, 4);     // * 16
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcountnate(eights), 3)); // += 8 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcountnate(fours),  2)); // += 4 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcountnate(twos),   1)); // += 2 * ...
  total = _mm256_add_epi64(total, popcountnate(ones));
  for(; i < size; i++) {
	__m256i data = _mm256_and_si256(_mm256_lddqu_si256(data1 + i), _mm256_lddqu_si256(data2 + i));

	total = _mm256_add_epi64(total, popcountnate(data));
  }
  return (uint64_t)(_mm256_extract_epi64(total, 0))
	       + (uint64_t)(_mm256_extract_epi64(total, 1))
	       + (uint64_t)(_mm256_extract_epi64(total, 2))
	       + (uint64_t)(_mm256_extract_epi64(total, 3));
}



int avx_harley_seal_and(const uint64_t*  restrict  dataA, const uint64_t*  restrict  dataB,size_t length,
		uint64_t * restrict  out) {
  const unsigned int wordspervector = sizeof(__m256i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  int total;
  if(length >= minvit) {
    total = popcntnate_and((const __m256i*) dataA,(const __m256i*) dataB, length / wordspervector, (__m256i*) out );
    for (size_t i = length - length % wordspervector; i < length; i++) {
        const uint64_t word_1 = (dataA[i]) & (dataB[i]);
        out[i] = word_1;
        total += _mm_popcnt_u64(word_1);
    }
    return total;
  }
  total = 0;
  for (size_t i = 0; i < length; i++) {
    const uint64_t word_1 = (dataA[i]) & (dataB[i]);
    out[i] = word_1;
    total += _mm_popcnt_u64(word_1);
  }
  return total;
}


#endif //HAVE_AVX2_INSTRUCTIONS


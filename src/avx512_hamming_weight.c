#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>
#include "config.h"
#ifdef HAVE_AVX512_INSTRUCTIONS

#include "avx512_hamming_weight.h"
#include "small_table.c"


static __m512i popcount(const __m512i v)
{
  const __m512i m1 = _mm512_set1_epi8(0x55);
  const __m512i m2 = _mm512_set1_epi8(0x33);
  const __m512i m4 = _mm512_set1_epi8(0x0F);

  const __m512i t1 = _mm512_sub_epi8(v, _mm512_and_si512(_mm512_srli_epi16(v,  1), m1));
  const __m512i t2 = _mm512_add_epi8(_mm512_and_si512(t1, m2), _mm512_and_si512(_mm512_srli_epi16(t1, 2), m2));
  const __m512i t3 = _mm512_and_si512(_mm512_add_epi8(t2, _mm512_srli_epi16(t2, 4)), m4);

  return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

static void CSA(__m512i* h, __m512i* l, __m512i a, __m512i b, __m512i c) {

  *l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
  *h = _mm512_ternarylogic_epi32(c, b, a, 0xe8);
}

static uint64_t avx2_sum_epu64(const __m256i v) {
    
    return _mm256_extract_epi64(v, 0)
         + _mm256_extract_epi64(v, 1)
         + _mm256_extract_epi64(v, 2)
         + _mm256_extract_epi64(v, 3);
}

uint64_t avx512_sum_epu64(const __m512i v) {

    const __m256i lo = _mm512_extracti64x4_epi64(v, 0);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);

    return avx2_sum_epu64(lo) + avx2_sum_epu64(hi);
}


// ------------------------------


static uint64_t popcnt_harley_seal(const __m512i* data, const uint64_t size)
{
  __m512i total     = _mm512_setzero_si512();
  __m512i ones      = _mm512_setzero_si512();
  __m512i twos      = _mm512_setzero_si512();
  __m512i fours     = _mm512_setzero_si512();
  __m512i eights    = _mm512_setzero_si512();
  __m512i sixteens  = _mm512_setzero_si512();
  __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for(; i < limit; i += 16)
  {
    CSA(&twosA, &ones, ones, data[i+0], data[i+1]);
    CSA(&twosB, &ones, ones, data[i+2], data[i+3]);
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, data[i+4], data[i+5]);
    CSA(&twosB, &ones, ones, data[i+6], data[i+7]);
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsA,&fours, fours, foursA, foursB);
    CSA(&twosA, &ones, ones, data[i+8], data[i+9]);
    CSA(&twosB, &ones, ones, data[i+10], data[i+11]);
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, data[i+12], data[i+13]);
    CSA(&twosB, &ones, ones, data[i+14], data[i+15]);
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsB, &fours, fours, foursA, foursB);
    CSA(&sixteens, &eights, eights, eightsA, eightsB);

    total = _mm512_add_epi64(total, popcount(sixteens));
  }

  total = _mm512_slli_epi64(total, 4);     // * 16
  total = _mm512_add_epi64(total, _mm512_slli_epi64(popcount(eights), 3)); // += 8 * ...
  total = _mm512_add_epi64(total, _mm512_slli_epi64(popcount(fours),  2)); // += 4 * ...
  total = _mm512_add_epi64(total, _mm512_slli_epi64(popcount(twos),   1)); // += 2 * ...
  total = _mm512_add_epi64(total, popcount(ones));

  for(; i < size; i++)
    total = _mm512_add_epi64(total, popcount(data[i]));


  return avx512_sum_epu64(total);
}


uint64_t popcnt_vpermb(const __m512i* data, const uint64_t count) {
    
    __m512i result = _mm512_setzero_si512();
    const __m512i lookup = _mm512_loadu_si512(small_table);
    __m512i a, b, c, d;

    const uint64_t tail = count % 3;
    const uint64_t size = count - tail;

    uint64_t i;
    for (i=0; i < size; i += 3) {
        a = _mm512_load_si512(data + i + 0);
        b = _mm512_load_si512(data + i + 1);
        c = _mm512_load_si512(data + i + 2);

        d = _mm512_srli_epi64(_mm512_and_si512(a, _mm512_set1_epi8(0xc0)), 6);
        d = _mm512_or_si512(d, _mm512_srli_epi64(_mm512_and_si512(b, _mm512_set1_epi8(0xc0)), 4));
        d = _mm512_or_si512(d, _mm512_srli_epi64(_mm512_and_si512(c, _mm512_set1_epi8(0xc0)), 2));

        // lookup
        a = _mm512_sad_epu8(_mm512_permutexvar_epi8(a, lookup), _mm512_setzero_si512()); // masking input is not needed
        b = _mm512_sad_epu8(_mm512_permutexvar_epi8(b, lookup), _mm512_setzero_si512());
        c = _mm512_sad_epu8(_mm512_permutexvar_epi8(c, lookup), _mm512_setzero_si512());
        d = _mm512_sad_epu8(_mm512_permutexvar_epi8(d, lookup), _mm512_setzero_si512());

        result = _mm512_add_epi64(result, a); 
        result = _mm512_add_epi64(result, b);
        result = _mm512_add_epi64(result, c);
        result = _mm512_add_epi64(result, d);
    }

    // looks unoptimal, I know
    if (tail) {
        i = size;

        a = _mm512_load_si512(data + i + 0);
        if (i + 1 < count) {
            b = _mm512_load_si512(data + i + 1);
        } else {
            b = _mm512_setzero_si512();
        }

        if (i + 2 < count) {
            c = _mm512_load_si512(data + i + 2);
        } else {
            c = _mm512_setzero_si512();
        }

        d = _mm512_srli_epi64(_mm512_and_si512(a, _mm512_set1_epi8(0xc0)), 6);
        d = _mm512_or_si512(d, _mm512_srli_epi64(_mm512_and_si512(b, _mm512_set1_epi8(0xc0)), 4));
        d = _mm512_or_si512(d, _mm512_srli_epi64(_mm512_and_si512(c, _mm512_set1_epi8(0xc0)), 2));

        a = _mm512_sad_epu8(_mm512_permutexvar_epi8(a, lookup), _mm512_setzero_si512());
        b = _mm512_sad_epu8(_mm512_permutexvar_epi8(b, lookup), _mm512_setzero_si512());
        c = _mm512_sad_epu8(_mm512_permutexvar_epi8(c, lookup), _mm512_setzero_si512());
        d = _mm512_sad_epu8(_mm512_permutexvar_epi8(d, lookup), _mm512_setzero_si512());

        result = _mm512_add_epi64(result, a); 
        result = _mm512_add_epi64(result, b);
        result = _mm512_add_epi64(result, c);
        result = _mm512_add_epi64(result, d);
    }

    return avx512_sum_epu64(result);
}


uint64_t popcnt_vperm2b(const __m512i* data, const uint64_t count) {
    
    __m512i result = _mm512_setzero_si512();
    const __m512i lookup0 = _mm512_loadu_si512(small_table);
    const __m512i lookup1 = _mm512_loadu_si512(small_table + 64);
    __m512i v, t;
    __mmask64 m;

    uint64_t i;
    for (i=0; i < count; i++) {
        v = _mm512_load_si512(data + i);

        // lookup
        t = _mm512_permutex2var_epi8(lookup0, v, lookup1);

        // add 7-th bit
#if 0
        v = _mm512_srli_epi64(v, 7); // (v & 0x80) ? 1 : 0
        v = _mm512_and_si512(v, _mm512_set1_epi8(0x01));
        t = _mm512_add_epi8(t, v);
#else // Nathan's idea
        m = _mm512_movepi8_mask(v);
        v = _mm512_movm_epi8(m);
        t = _mm512_sub_epi8(t, v);
#endif

        result = _mm512_add_epi64(result, _mm512_sad_epu8(t, _mm512_setzero_si512()));
    }

    return avx512_sum_epu64(result);
}


// --- public -------------------------------------------------


uint64_t avx512_harley_seal(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_harley_seal((const __m512i*) data, size / wordspervector);
    i = size - size % wordspervector;
  } else {
    total = 0;
    i = 0;
  }

  for (/**/; i < size; i++) {
    total += _mm_popcnt_u64(data[i]);
  }
  return total;
}


uint64_t avx512_vpermb(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_vpermb((const __m512i*) data, size / wordspervector);
    i = size - size % wordspervector;
  } else {
    total = 0;
    i = 0;
  }

  for (/**/; i < size; i++) {
    total += _mm_popcnt_u64(data[i]);
  }
  return total;
}


uint64_t avx512_vperm2b(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_vperm2b((const __m512i*) data, size / wordspervector);
    i = size - size % wordspervector;
  } else {
    total = 0;
    i = 0;
  }

  for (/**/; i < size; i++) {
    total += _mm_popcnt_u64(data[i]);
  }
  return total;
}


#endif // HAVE_AVX512_INSTRUCTIONS


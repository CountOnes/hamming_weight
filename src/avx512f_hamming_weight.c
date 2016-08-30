#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>
#include "config.h"

#ifdef HAVE_AVX512F_INSTRUCTIONS

#include "avx512f_hamming_weight.h"

static __m256i avx2_popcount(const __m256i vec) {

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

    const __m256i lo  = _mm256_and_si256(vec, low_mask);
    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);

    return _mm256_add_epi8(popcnt1, popcnt2);
}


static __m256i popcount(const __m512i v)
{
    const __m256i lo = _mm512_extracti64x4_epi64(v, 0);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    const __m256i s  = _mm256_add_epi8(avx2_popcount(lo), avx2_popcount(hi));

    return _mm256_sad_epu8(s, _mm256_setzero_si256());
}


static uint64_t avx2_sum_epu64(const __m256i v) {
    
    return _mm256_extract_epi64(v, 0)
         + _mm256_extract_epi64(v, 1)
         + _mm256_extract_epi64(v, 2)
         + _mm256_extract_epi64(v, 3);
}


static void CSA(__m512i* h, __m512i* l, __m512i a, __m512i b, __m512i c) {

  *l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
  *h = _mm512_ternarylogic_epi32(c, b, a, 0xe8);
}


// ------------------------------


static uint64_t popcnt_harley_seal(const __m512i* data, const uint64_t size)
{
  __m256i total     = _mm256_setzero_si256();
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

    total = _mm256_add_epi64(total, popcount(sixteens));
  }

  total = _mm256_slli_epi64(total, 4);     // * 16
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(eights), 3)); // += 8 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(fours),  2)); // += 4 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(twos),   1)); // += 2 * ...
  total = _mm256_add_epi64(total, popcount(ones));

  for(; i < size; i++) {
    total = _mm256_add_epi64(total, popcount(data[i]));
  }


  return avx2_sum_epu64(total);
}


// ---------------


static uint64_t _mm256_popcnt(const __m256i v) {
    return _mm_popcnt_u64(_mm256_extract_epi64(v, 0))
         + _mm_popcnt_u64(_mm256_extract_epi64(v, 1))
         + _mm_popcnt_u64(_mm256_extract_epi64(v, 2))
         + _mm_popcnt_u64(_mm256_extract_epi64(v, 3));
}


static uint64_t _mm512_popcnt(const __m512i v) {
    return _mm256_popcnt(_mm512_extracti64x4_epi64(v, 0))
         + _mm256_popcnt(_mm512_extracti64x4_epi64(v, 1));
}


static uint64_t popcnt_harley_seal__hardware_popcnt(const __m512i* data, const uint64_t size)
{
  uint64_t total = 0;
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

    total += _mm512_popcnt(sixteens);
  }

  total *= 16;
  total += 8 * _mm512_popcnt(eights);
  total += 4 * _mm512_popcnt(fours);
  total += 2 * _mm512_popcnt(twos);
  total += _mm512_popcnt(ones);

  for(; i < size; i++) {
    total += _mm512_popcnt(data[i]);
  }

  return total;
}


// ------------------------------


static uint64_t popcnt_harley_seal__hardware_popcnt_2(const __m512i* data, const uint64_t size)
{
  uint64_t total = 0;
  __m512i ones      = _mm512_setzero_si512();
  __m512i twos      = _mm512_setzero_si512();
  __m512i fours     = _mm512_setzero_si512();
  __m512i eights    = _mm512_setzero_si512();
  __m512i sixteens  = _mm512_setzero_si512();
  __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

  __m256i lo, hi;

#define UPDATE_POPCNT(var, vec) \
        lo = _mm512_extracti64x4_epi64(vec, 0); \
        hi = _mm512_extracti64x4_epi64(vec, 1); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(lo, 0)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(lo, 1)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(lo, 2)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(lo, 3)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(hi, 0)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(hi, 1)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(hi, 2)); \
        var += _mm_popcnt_u64(_mm256_extract_epi64(hi, 3));

#define CSA_BLOCK \
    CSA(&twosA, &ones, ones, data[i+0], data[i+1]); \
    CSA(&twosB, &ones, ones, data[i+2], data[i+3]); \
    CSA(&foursA, &twos, twos, twosA, twosB); \
    CSA(&twosA, &ones, ones, data[i+4], data[i+5]); \
    CSA(&twosB, &ones, ones, data[i+6], data[i+7]); \
    CSA(&foursB, &twos, twos, twosA, twosB); \
    CSA(&eightsA,&fours, fours, foursA, foursB); \
    CSA(&twosA, &ones, ones, data[i+8], data[i+9]); \
    CSA(&twosB, &ones, ones, data[i+10], data[i+11]); \
    CSA(&foursA, &twos, twos, twosA, twosB); \
    CSA(&twosA, &ones, ones, data[i+12], data[i+13]); \
    CSA(&twosB, &ones, ones, data[i+14], data[i+15]); \
    CSA(&foursB, &twos, twos, twosA, twosB); \
    CSA(&eightsB, &fours, fours, foursA, foursB); \
    CSA(&sixteens, &eights, eights, eightsA, eightsB);

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  if (i <= 16) {
    CSA_BLOCK
    i += 16;
  }

  for(; i < limit; i += 16)
  {
    UPDATE_POPCNT(total, sixteens);
    CSA_BLOCK
  }

  total += _mm512_popcnt(sixteens);
  total *= 16;
  total += 8 * _mm512_popcnt(eights);
  total += 4 * _mm512_popcnt(fours);
  total += 2 * _mm512_popcnt(twos);
  total += _mm512_popcnt(ones);

  for(; i < size; i++) {
    total += _mm512_popcnt(data[i]);
  }

#undef CSA_BLOCK
#undef UPDATE_POPCNT

  return total;
}


// ------------------------------


#include "small_table.c"

static uint32_t small_table_32bit[256];

void avx512f_gather_init() {
    for (int i=0; i < 256; i++) {
        small_table_32bit[i] = small_table[i];
    }
}


static uint32_t sse_sum_epu32(const __m128i v) {
    return _mm_extract_epi32(v, 0)
         + _mm_extract_epi32(v, 1)
         + _mm_extract_epi32(v, 2)
         + _mm_extract_epi32(v, 3);
        
}


static uint64_t avx512_sum_epu32(const __m512i v) {
    
    return sse_sum_epu32(_mm512_extracti32x4_epi32(v, 0))
         + sse_sum_epu32(_mm512_extracti32x4_epi32(v, 1))
         + sse_sum_epu32(_mm512_extracti32x4_epi32(v, 2))
         + sse_sum_epu32(_mm512_extracti32x4_epi32(v, 3));
}


static uint64_t popcnt_gather(const __m512i* data, const uint64_t size)
{
    __m512i b0, b1, b2, b3;
    __m512i p0, p1, p2, p3;
    __m512i v;
    __m512i total = _mm512_setzero_si512();


    for(uint64_t i=0; i < size; i++) {

        v = data[i];
        b0 = _mm512_and_epi32(v, _mm512_set1_epi32(0xff));
        b1 = _mm512_and_epi32(_mm512_srli_epi32(v, 1*8), _mm512_set1_epi32(0xff));
        b2 = _mm512_and_epi32(_mm512_srli_epi32(v, 2*8), _mm512_set1_epi32(0xff));
        b3 = _mm512_srli_epi32(v, 3*8);

        p0 = _mm512_i32gather_epi32(b0, (const int*)small_table_32bit, 4);   
        p1 = _mm512_i32gather_epi32(b1, (const int*)small_table_32bit, 4);   
        p2 = _mm512_i32gather_epi32(b2, (const int*)small_table_32bit, 4);   
        p3 = _mm512_i32gather_epi32(b3, (const int*)small_table_32bit, 4);   

        total = _mm512_add_epi32(total, p0);
        total = _mm512_add_epi32(total, p1);
        total = _mm512_add_epi32(total, p2);
        total = _mm512_add_epi32(total, p3);
    }

    return avx512_sum_epu32(total);
}


// --- public -------------------------------------------------


uint64_t avx512f_harley_seal(const uint64_t * data, size_t size) {
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


uint64_t avx512f_harley_seal__hardware_popcnt(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_harley_seal__hardware_popcnt((const __m512i*) data, size / wordspervector);
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


uint64_t avx512f_harley_seal__hardware_popcnt_2(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_harley_seal__hardware_popcnt_2((const __m512i*) data, size / wordspervector);
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


uint64_t avx512f_gather(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_gather((const __m512i*) data, size / wordspervector);
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


#endif // HAVE_AVX512F_INSTRUCTIONS


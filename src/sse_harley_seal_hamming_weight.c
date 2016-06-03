#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#include "sse_harley_seal_hamming_weight.h"

static inline __attribute__((always_inline)) __m128i popcount(__m128i v) {

    const __m128i lookup = _mm_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m128i low_mask = _mm_set1_epi8(0x0f);

    const __m128i lo  = _mm_and_si128(v, low_mask);
    const __m128i hi  = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
    const __m128i popcnt1 = _mm_shuffle_epi8(lookup, lo);
    const __m128i popcnt2 = _mm_shuffle_epi8(lookup, hi);
    return _mm_sad_epu8(_mm_add_epi8(popcnt1, popcnt2), _mm_setzero_si128());
}


static inline __attribute__((always_inline)) void CSA(__m128i *h, __m128i* l, __m128i a, __m128i b, __m128i c)
{
  const __m128i u = _mm_xor_si128(a,b);
  *h = _mm_or_si128(_mm_and_si128(a,b) , _mm_and_si128(u , c));
  *l = _mm_xor_si128(u , c);
}

uint64_t popcnt(const __m128i* data, const uint64_t size)
{
  __m128i total     = _mm_setzero_si128();
  __m128i ones      = _mm_setzero_si128();
  __m128i twos      = _mm_setzero_si128();
  __m128i fours     = _mm_setzero_si128();
  __m128i eights    = _mm_setzero_si128();
  __m128i sixteens  = _mm_setzero_si128();
  __m128i twosA, twosB, foursA, foursB, eightsA, eightsB;

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

    total = _mm_add_epi64(total, popcount(sixteens));
  }


  total = _mm_slli_epi64(total, 4);     // * 16
  total = _mm_add_epi64(total, _mm_slli_epi64(popcount(eights), 3)); // += 8 * ...
  total = _mm_add_epi64(total, _mm_slli_epi64(popcount(fours),  2)); // += 4 * ...
  total = _mm_add_epi64(total, _mm_slli_epi64(popcount(twos),   1)); // += 2 * ...
  total = _mm_add_epi64(total, popcount(ones));

  for(; i < size; i++)
    total += popcount(data[i]);

  return _mm_extract_epi64(total,0) + _mm_extract_epi64(total,1);
}


int sse_harley_seal_bitset64_weight(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m128i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  int total;
  if(size >= minvit) {
    total = popcnt((const __m128i*) data, size / wordspervector);
    for (size_t i = size - size % wordspervector; i < size; i++) {
      total += _mm_popcnt_u64(data[i]);
    }
    return total;
  }
  total = 0;
  for (size_t i = size - size % minvit; i < size; i++) {
    total += _mm_popcnt_u64(data[i]);
  }
  return total;
}



#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>
#include "config.h"

#ifdef HAVE_AVX512CD_INSTRUCTIONS

#include "avx512cd_hamming_weight.h"

// ------------------------------


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


void dump_hex(const uint8_t* buf, size_t num) {
    for (size_t i=0; i < num; i++) {
        printf("%02x", buf[i]);
    }
}


void dump(const char* name, __m512i xmm) {
    
    uint8_t buf[64];
    _mm512_storeu_si512((__m512*)(buf), xmm);
    printf("%-10s:", name);
    dump_hex(buf, 64);
    putchar('\n');
}



static uint64_t popcnt_naive(const __m512i* data, const uint64_t size)
{
    __m512i zeros = _mm512_setzero_si512();
    __m512i total = _mm512_setzero_si512();
    __m512i one   = _mm512_set1_epi32(1);

    for (uint64_t i=0; i < size; i++) {
        __mmask16 m = 0;
        __mmask16 z = 0;
        __m512i   v = data[i];

        m = _mm512_cmpneq_epi32_mask(v, zeros);
        z = m;
        while (z) {
            
            __m512i lz = _mm512_mask_lzcnt_epi32(zeros, z, v);
            m = _mm512_cmpneq_epi32_mask(lz, _mm512_set1_epi32(32));
            z = _mm512_kand(m, z);

            total = _mm512_mask_add_epi32(total, z, total, one);

#if 1
            lz = _mm512_mask_add_epi32(lz, z, lz, one);
            v = _mm512_sllv_epi32(v, lz);
#else
            v = _mm512_sllv_epi32(v, lz);
            v = _mm512_and_si512(v, _mm512_set1_epi32(0x7fffffffu));
#endif
        }
    }


    return avx512_sum_epu32(total);
}


// --- public -------------------------------------------------


uint64_t avx512cd_naive(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_naive((const __m512i*) data, size / wordspervector);
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


#endif // HAVE_AVX512CD_INSTRUCTIONS


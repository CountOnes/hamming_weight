#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>
#include "config.h"
#ifdef HAVE_AVX512BW_INSTRUCTIONS

#include "avx512bw_hamming_weight.h"
#include "small_table.c"


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

#endif // HAVE_AVX512BW_INSTRUCTIONS

// ------------------------------

#ifdef HAVE_AVX512VBMI_INSTRUCTIONS
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


uint64_t popcnt_vperm2b_ver2(const __m512i* data, const uint64_t count) {
    
    __m512i  v_result = _mm512_setzero_si512();
    uint64_t s_result = 0;

    const __m512i lookup0 = _mm512_loadu_si512(small_table);
    const __m512i lookup1 = _mm512_loadu_si512(small_table + 64);
    __m512i v, t;

    uint64_t i;
    for (i=0; i < count; i++) {
        v = _mm512_load_si512(data + i);

        // lookup 6 lower bits
        t = _mm512_permutex2var_epi8(lookup0, v, lookup1);

        // separately accumulate the 7th bit
        s_result = s_result + _mm_popcnt_u64(_mm512_movepi8_mask(v));

        v_result = _mm512_add_epi64(v_result, _mm512_sad_epu8(t, _mm512_setzero_si512()));
    }

    return avx512_sum_epu64(v_result) + s_result;
}


uint64_t popcnt_vperm2b_ver2_unrolled(const __m512i* data, const uint64_t count) {
    
    __m512i  v_result = _mm512_setzero_si512();
    uint64_t s_result = 0;

    const __m512i lookup0 = _mm512_loadu_si512(small_table);
    const __m512i lookup1 = _mm512_loadu_si512(small_table + 64);
    __m512i v0;
    __m512i v1;
    __m512i v2;
    __m512i v3;
    __m512i t;
    __m512i t0;
    __m512i t1;
    __m512i t2;
    __m512i t3;

    uint64_t i;
    for (i=0; i < count; i += 4) {
        v0 = _mm512_load_si512(data + i + 0);
        v1 = _mm512_load_si512(data + i + 1);
        v2 = _mm512_load_si512(data + i + 2);
        v3 = _mm512_load_si512(data + i + 3);

        t0 = _mm512_permutex2var_epi8(lookup0, v0, lookup1);
        t1 = _mm512_permutex2var_epi8(lookup0, v1, lookup1);
        t2 = _mm512_permutex2var_epi8(lookup0, v2, lookup1);
        t3 = _mm512_permutex2var_epi8(lookup0, v3, lookup1);

        t0 = _mm512_add_epi8(t0, t1);
        t2 = _mm512_add_epi8(t2, t3);
        t  = _mm512_add_epi8(t0, t2);

        s_result = s_result + _mm_popcnt_u64(_mm512_movepi8_mask(v0));
        s_result = s_result + _mm_popcnt_u64(_mm512_movepi8_mask(v1));
        s_result = s_result + _mm_popcnt_u64(_mm512_movepi8_mask(v2));
        s_result = s_result + _mm_popcnt_u64(_mm512_movepi8_mask(v3));

        v_result = _mm512_add_epi64(v_result, _mm512_sad_epu8(t, _mm512_setzero_si512()));
    }

    for (i=4*(count/4); i < count; i++) {
        v0 = _mm512_load_si512(data + i);
        t0 = _mm512_permutex2var_epi8(lookup0, v0, lookup1);

        s_result = s_result + _mm_popcnt_u64(_mm512_movepi8_mask(v0));
        v_result = _mm512_add_epi64(v_result, _mm512_sad_epu8(t0, _mm512_setzero_si512()));
    }

    return avx512_sum_epu64(v_result) + s_result;
}

// --- public -------------------------------------------------

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


uint64_t avx512_vperm2b_ver2(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_vperm2b_ver2((const __m512i*) data, size / wordspervector);
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


uint64_t avx512_vperm2b_ver2_unrolled(const uint64_t * data, size_t size) {
  const unsigned int wordspervector = sizeof(__m512i) / sizeof(uint64_t);
  const unsigned int minvit = 16 * wordspervector;
  uint64_t total;
  size_t i;

  if (size >= minvit) {
    total = popcnt_vperm2b_ver2_unrolled((const __m512i*) data, size / wordspervector);
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


#endif


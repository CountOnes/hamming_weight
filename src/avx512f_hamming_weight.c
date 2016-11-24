#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>

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


static uint64_t __attribute__((always_inline)) _mm512_popcnt(const __m512i v) {

    uint64_t tmp[8] __attribute__((aligned(64)));

    _mm512_store_si512(tmp, v);

    return _mm_popcnt_u64(tmp[0])
         + _mm_popcnt_u64(tmp[1])
         + _mm_popcnt_u64(tmp[2])
         + _mm_popcnt_u64(tmp[3])
         + _mm_popcnt_u64(tmp[4])
         + _mm_popcnt_u64(tmp[5])
         + _mm_popcnt_u64(tmp[6])
         + _mm_popcnt_u64(tmp[7]);
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


#define USE_INLINE_ASM 1

#if defined(USE_INLINE_ASM)
register __m512i d0 asm("zmm15");
register __m512i d1 asm("zmm16");
register __m512i d2 asm("zmm17");
register __m512i d3 asm("zmm18");
register __m512i d4 asm("zmm19");
register __m512i d5 asm("zmm20");
register __m512i d6 asm("zmm21");
register __m512i d7 asm("zmm22");
register __m512i d8 asm("zmm23");
register __m512i d9 asm("zmm24");
register __m512i dA asm("zmm25");
register __m512i dB asm("zmm26");
register __m512i dC asm("zmm27");
register __m512i dD asm("zmm28");
register __m512i dE asm("zmm29");
register __m512i dF asm("zmm30");
#endif

static uint64_t popcnt_harley_seal__hardware_popcnt_2(const __m512i* data, const uint64_t size)
{
  uint64_t total = 0;
  __m512i ones      = _mm512_setzero_si512();
  __m512i twos      = _mm512_setzero_si512();
  __m512i fours     = _mm512_setzero_si512();
  __m512i eights    = _mm512_setzero_si512();
  __m512i sixteens  = _mm512_setzero_si512();
  __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

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

  uint64_t tmp[8] __attribute__((aligned(64)));

  for(/**/; i < limit; i += 16)
  {
#if !defined(USE_INLINE_ASM)
    _mm512_store_si512(tmp, sixteens);
    CSA(&twosA, &ones, ones, data[i+0], data[i+1]);
    total += _mm_popcnt_u64(tmp[0]);
    CSA(&twosB, &ones, ones, data[i+2], data[i+3]);
    total += _mm_popcnt_u64(tmp[1]);
    CSA(&foursA, &twos, twos, twosA, twosB);
    total += _mm_popcnt_u64(tmp[2]);
    CSA(&twosA, &ones, ones, data[i+4], data[i+5]);
    total += _mm_popcnt_u64(tmp[3]);
    CSA(&twosB, &ones, ones, data[i+6], data[i+7]);
    total += _mm_popcnt_u64(tmp[4]);
    CSA(&foursB, &twos, twos, twosA, twosB);
    total += _mm_popcnt_u64(tmp[5]);
    CSA(&eightsA,&fours, fours, foursA, foursB);
    total += _mm_popcnt_u64(tmp[6]);
    CSA(&twosA, &ones, ones, data[i+8], data[i+9]);
    total += _mm_popcnt_u64(tmp[7]);
    CSA(&twosB, &ones, ones, data[i+10], data[i+11]);
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, data[i+12], data[i+13]);
    CSA(&twosB, &ones, ones, data[i+14], data[i+15]);
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsB, &fours, fours, foursA, foursB);
    CSA(&sixteens, &eights, eights, eightsA, eightsB);
#else
    
    /*
    
    inline, naive assembly -- the first attempt, translation of following code

        CSA(&t0, &ones, ones, data[i+0], data[i+1]);
        CSA(&t1, &ones, ones, data[i+2], data[i+3]);
        CSA(&t2, &ones, ones, data[i+4], data[i+5]);
        CSA(&t3, &ones, ones, data[i+6], data[i+7]);
        CSA(&t4, &ones, ones, data[i+8], data[i+9]);
        CSA(&t5, &ones, ones, data[i+10], data[i+11]);
        CSA(&t6, &ones, ones, data[i+12], data[i+13]);
        CSA(&t7, &ones, ones, data[i+14], data[i+15]);

        CSA(&t0, &twos, twos, t0, t1);
        CSA(&t2, &twos, twos, t2, t3);
        CSA(&t4, &twos, twos, t4, t5);
        CSA(&t6, &twos, twos, t6, t7);

        CSA(&t0, &fours, fours, t0, t2);
        CSA(&t4, &fours, fours, t4, t6);
        
        CSA(&sixteens, &eights, eights, t0, t4);
    
    There are three asm statements, I couldn't put everything into
    one statement due to error: "error: more than 30 operands in ‘asm’".
    */

    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
    uint64_t r0, r1, r2, r3;
    _mm512_store_si512(tmp, sixteens);

    __asm__ volatile (
        "vmovdqa64      0x0000(%[data]), %%zmm15                    \n"
        "popcnt         0x00(%[tmp]), %[r0]                         \n"
        "vmovdqa64      0x0040(%[data]), %%zmm16                    \n"
        "popcnt         0x08(%[tmp]), %[r1]                         \n"
        "vmovdqa64      0x0080(%[data]), %%zmm17                    \n"
        "popcnt         0x10(%[tmp]), %[r2]                         \n"
        "vmovdqa64      0x00c0(%[data]), %%zmm18                    \n"
        "popcnt         0x18(%[tmp]), %[r3]                         \n"
        "vmovdqa64      0x0100(%[data]), %%zmm19                    \n"
        "addq           %[r1], %[r0]                                \n"
        "vmovdqa64      0x0140(%[data]), %%zmm20                    \n"
        "addq           %[r3], %[r2]                                \n"
        "vmovdqa64      0x0180(%[data]), %%zmm21                    \n"
        "addq           %[r0], %[total]                             \n"
        "vmovdqa64      0x01c0(%[data]), %%zmm22                    \n"
        "addq           %[r2], %[total]                             \n"
        "vmovdqa64      0x0200(%[data]), %%zmm23                    \n"
        "popcnt         0x20(%[tmp]), %[r0]                         \n"
        "vmovdqa64      0x0240(%[data]), %%zmm24                    \n"
        "popcnt         0x28(%[tmp]), %[r1]                         \n"
        "vmovdqa64      0x0280(%[data]), %%zmm25                    \n"
        "popcnt         0x30(%[tmp]), %[r2]                         \n"
        "vmovdqa64      0x02c0(%[data]), %%zmm26                    \n"
        "popcnt         0x38(%[tmp]), %[r2]                         \n"
        "vmovdqa64      0x0300(%[data]), %%zmm27                    \n"
        "addq           %[r1], %[r0]                                \n"
        "vmovdqa64      0x0340(%[data]), %%zmm28                    \n"
        "addq           %[r3], %[r2]                                \n"
        "vmovdqa64      0x0380(%[data]), %%zmm29                    \n"
        "addq           %[r0], %[total]                             \n"
        "vmovdqa64      0x03c0(%[data]), %%zmm30                    \n"
        "addq           %[r2], %[total]                             \n"

        "vmovdqa64      %[ones], %[t0]                              \n"
        "vpternlogd     $0x96, %%zmm15, %%zmm16, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm15, %%zmm16, %[t0]                  \n"

        "vmovdqa64      %[ones], %[t1]                              \n"
        "vpternlogd     $0x96, %%zmm17, %%zmm18, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm17, %%zmm18, %[t1]                  \n"

        "vmovdqa64      %[ones], %[t2]                              \n"
        "vpternlogd     $0x96, %%zmm19, %%zmm20, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm19, %%zmm20, %[t2]                  \n"

        "vmovdqa64      %[ones], %[t3]                              \n"
        "vpternlogd     $0x96, %%zmm21, %%zmm22, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm21, %%zmm22, %[t3]                  \n"

        "vmovdqa64      %[ones], %[t4]                              \n"
        "vpternlogd     $0x96, %%zmm23, %%zmm24, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm23, %%zmm24, %[t4]                  \n"

        "vmovdqa64      %[ones], %[t5]                              \n"
        "vpternlogd     $0x96, %%zmm25, %%zmm26, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm25, %%zmm26, %[t5]                  \n"

        "vmovdqa64      %[ones], %[t6]                              \n"
        "vpternlogd     $0x96, %%zmm27, %%zmm28, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm27, %%zmm28, %[t6]                  \n"

        "vmovdqa64      %[ones], %[t7]                              \n"
        "vpternlogd     $0x96, %%zmm29, %%zmm30, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm29, %%zmm30, %[t7]                  \n"

        // outputs
        : [ones]  "+x" (ones)
        , [t0] "=x" (t0)
        , [t1] "=x" (t1)
        , [t2] "=x" (t2)
        , [t3] "=x" (t3)
        , [t4] "=x" (t4)
        , [t5] "=x" (t5)
        , [t6] "=x" (t6)
        , [t7] "=x" (t7)
        , [t8] "+x" (t8)
        , [t9] "+x" (t9)
        , [total] "+r" (total)
        , [r0] "+r" (r0)
        , [r1] "+r" (r1)
        , [r2] "+r" (r2)
        , [r3] "+r" (r3)

        // input
        : [data] "r" (data + i)
        , [tmp] "r" (tmp)
    );

    __asm__ volatile (
        // from this point t0 .. t7 are processed

        "vmovdqa64      %[twos], %[t8]                              \n"
        "vpternlogd     $0x96, %[t0], %[t1], %[twos]                \n"
        "vpternlogd     $0xe8, %[t8], %[t1], %[t0]                  \n"

        "vmovdqa64      %[twos], %[t8]                              \n"
        "vpternlogd     $0x96, %[t2], %[t3], %[twos]                \n"
        "vpternlogd     $0xe8, %[t8], %[t3], %[t2]                  \n"

        "vmovdqa64      %[twos], %[t8]                              \n"
        "vpternlogd     $0x96, %[t4], %[t5], %[twos]                \n"
        "vpternlogd     $0xe8, %[t8], %[t5], %[t4]                  \n"

        "vmovdqa64      %[twos], %[t8]                              \n"
        "vpternlogd     $0x96, %[t6], %[t7], %[twos]                \n"
        "vpternlogd     $0xe8, %[t8], %[t7], %[t6]                  \n"

        // outputs
        : [twos] "+x" (twos)
        , [t0] "+x" (t0)
        , [t1] "+x" (t1)
        , [t2] "+x" (t2)
        , [t3] "+x" (t3)
        , [t4] "+x" (t4)
        , [t5] "+x" (t5)
        , [t6] "+x" (t6)
        , [t7] "+x" (t7)
        , [t8] "+x" (t8)
        , [t9] "+x" (t9)
    );

    __asm__ volatile (
        // from this point t0, t2, t4, t6 are processed

        "vmovdqa64      %[fours], %[t8]                             \n"
        "vpternlogd     $0x96, %[t0], %[t2], %[fours]               \n"
        "vpternlogd     $0xe8, %[t8], %[t2], %[t0]                  \n"

        "vmovdqa64      %[fours], %[t8]                             \n"
        "vpternlogd     $0x96, %[t4], %[t6], %[fours]               \n"
        "vpternlogd     $0xe8, %[t8], %[t6], %[t4]                  \n"

        // from this point t0, t4, are processed

        "vmovdqa64      %[eights], %[sixteens]                      \n"
        "vpternlogd     $0x96, %[t0], %[t4], %[eights]              \n"
        "vpternlogd     $0xe8, %[t0], %[t4], %[sixteens]            \n"

        // outputs
        : [fours] "+x" (fours)
        , [eights] "+x" (eights)
        , [sixteens] "+x" (sixteens)
        , [t0] "+x" (t0)
        , [t2] "+x" (t2)
        , [t4] "+x" (t4)
        , [t6] "+x" (t6)
        , [t8] "+x" (t8)
    );

#endif // !defined(USE_INLINE_ASM)
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


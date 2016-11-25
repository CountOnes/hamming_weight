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


static uint64_t popcnt_harley_seal__hardware_popcnt_2(const __m512i* data, const uint64_t size)
{
  uint64_t total = 0;
  __m512i ones      = _mm512_setzero_si512();
  __m512i twos      = _mm512_setzero_si512();
  __m512i fours     = _mm512_setzero_si512();
  __m512i eights    = _mm512_setzero_si512();
  __m512i sixteens  = _mm512_setzero_si512();
  __m512i thirtytwos = _mm512_setzero_si512();

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  uint64_t tmp[8] __attribute__((aligned(64)));

  for(; i < limit; i += 32)
  {
    /*
    
    inline, naive assembly -- the first attempt, translation of following code
    (now doubled, to handle thirtytwos)

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

    uint64_t block_total;

    __asm__ volatile (
        "vmovdqa64      0x0000(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0040(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm10                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm10                \n"

        // store tmp
        "vmovdqa64      %[thirtytwos], (%[tmp])                         \n"

        "vmovdqa64      0x0080(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x00c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm11                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm11                \n"

        "vmovdqa64      0x0100(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0140(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm12                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm12                \n"

        "vmovdqa64      0x0180(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x01c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm13                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm13                \n"

        "vmovdqa64      0x0200(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0240(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm14                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm14                \n"

        "vmovdqa64      0x0280(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x02c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm15                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm15                \n"

        "vmovdqa64      0x0300(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0340(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm16                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm16                \n"

        "vmovdqa64      0x0380(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x03c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm17                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm17                \n"

        "vmovdqa64      0x0400(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0440(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm18                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm18                \n"

        "vmovdqa64      0x0480(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x04c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm19                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm19                \n"

        "vmovdqa64      0x0500(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0540(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm20                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm20                \n"

        "vmovdqa64      0x0580(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x05c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm21                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm21                \n"

        "vmovdqa64      0x0600(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0640(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm22                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm22                \n"

        "vmovdqa64      0x0680(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x06c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm23                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm23                \n"

        "vmovdqa64      0x0700(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x0740(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm24                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm24                \n"

        "vmovdqa64      0x0780(%[data]), %%zmm30                        \n"
        "vmovdqa64      0x07c0(%[data]), %%zmm31                        \n"
        "vmovdqa64      %[ones], %%zmm25                                \n"
        "vpternlogd     $0x96, %%zmm30, %%zmm31, %[ones]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm31, %%zmm25                \n"

        // twos
        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm10, %%zmm11, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm11, %%zmm10                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm12, %%zmm13, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm13, %%zmm12                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm14, %%zmm15, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm15, %%zmm14                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm16, %%zmm17, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm17, %%zmm16                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm18, %%zmm19, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm19, %%zmm18                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm20, %%zmm21, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm21, %%zmm20                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm22, %%zmm23, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm23, %%zmm22                \n"

        "vmovdqa64      %[twos], %%zmm30                                \n"
        "vpternlogd     $0x96, %%zmm24, %%zmm25, %[twos]                \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm25, %%zmm24                \n"

#if 1
        "popcnt         0x00(%[tmp]), %%r8                              \n"
        "popcnt         0x08(%[tmp]), %%r9                              \n"
        "popcnt         0x10(%[tmp]), %%r10                             \n"
        "popcnt         0x18(%[tmp]), %%r11                             \n"
        "popcnt         0x20(%[tmp]), %%r12                             \n"
        "popcnt         0x28(%[tmp]), %%r13                             \n"
        "popcnt         0x30(%[tmp]), %%r14                             \n"
        "popcnt         0x38(%[tmp]), %%r15                             \n"

        "xorq           %[total], %[total]                              \n"
        "addq           %%r8,  %[total]                                 \n"
        "addq           %%r9,  %[total]                                 \n"
        "addq           %%r10, %[total]                                 \n"
        "addq           %%r11, %[total]                                 \n"
        "addq           %%r12, %[total]                                 \n"
        "addq           %%r13, %[total]                                 \n"
        "addq           %%r14, %[total]                                 \n"
        "addq           %%r15, %[total]                                 \n"
#endif

        // fours
        "vmovdqa64      %[fours], %%zmm30                               \n"
        "vpternlogd     $0x96, %%zmm10, %%zmm12, %[fours]               \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm12, %%zmm10                \n"

        "vmovdqa64      %[fours], %%zmm30                               \n"
        "vpternlogd     $0x96, %%zmm14, %%zmm16, %[fours]               \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm16, %%zmm14                \n"

        "vmovdqa64      %[fours], %%zmm30                               \n"
        "vpternlogd     $0x96, %%zmm18, %%zmm20, %[fours]               \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm20, %%zmm18                \n"

        "vmovdqa64      %[fours], %%zmm30                               \n"
        "vpternlogd     $0x96, %%zmm22, %%zmm24, %[fours]               \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm24, %%zmm22                \n"

        // eights
        "vmovdqa64      %[eights], %%zmm30                              \n"
        "vpternlogd     $0x96, %%zmm10, %%zmm14, %[eights]              \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm14, %%zmm10                \n"

        "vmovdqa64      %[eights], %%zmm30                              \n"
        "vpternlogd     $0x96, %%zmm18, %%zmm22, %[eights]              \n"
        "vpternlogd     $0xe8, %%zmm30, %%zmm22, %%zmm18                \n"

        // sixteens
        "vmovdqa64      %[sixteens], %[thirtytwos]                      \n"
        "vpternlogd     $0x96, %%zmm10, %%zmm18, %[sixteens]            \n"
        "vpternlogd     $0xe8, %%zmm10, %%zmm18, %[thirtytwos]          \n"

        // outputs
        : [ones]        "+x" (ones)
        , [twos]        "+x" (twos)
        , [fours]       "+x" (fours)
        , [eights]      "+x" (eights)
        , [sixteens]    "+x" (sixteens)
        , [thirtytwos]  "+x" (thirtytwos)
        , [total]       "=r" (block_total)

        // input
        : [data] "r" (data + i)
        , [tmp]  "r" (tmp)

        : "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19"
        , "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25"
        , "zmm30", "zmm31"
        , "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
    );

    total += block_total;
  }

  total += _mm512_popcnt(thirtytwos);
  total *= 32;
  total += 16 * _mm512_popcnt(sixteens);
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


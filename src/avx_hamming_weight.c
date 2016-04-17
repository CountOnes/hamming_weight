#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

#ifdef HAVE_AVX2_INSTRUCTIONS


#include "avx_hamming_weight.h"
#include "popcnt_hamming_weight.h"

// compute the Hamming weight of an array of 64-bit words using AVX2 instructions
int avx2_bitset64_weight(const uint64_t * array, size_t length) {
    // these are precomputed hamming weights (weight(0), weight(1)...)
    const __m256i shuf =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1,
                         1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i mask = _mm256_set1_epi8(0x0f);  // low 4 bits of each byte
    __m256i total = _mm256_setzero_si256();
    __m256i zero = _mm256_setzero_si256();
    const int inner = 4;  // length of the inner loop, could go up to 8 safely
    const int outer = length * sizeof(uint64_t) /
                      (sizeof(__m256i) * inner);  // length of outer loop
    for (int k = 0; k < outer; k++) {
        __m256i innertotal = _mm256_setzero_si256();
        for (int i = 0; i < inner; ++i) {
            __m256i ymm1 =
                _mm256_lddqu_si256((const __m256i *)array + k * inner + i);
            __m256i ymm2 =
                _mm256_srli_epi32(ymm1, 4);  // shift right, shiftingin zeroes
            ymm1 = _mm256_and_si256(ymm1, mask);  // contains even 4 bits
            ymm2 = _mm256_and_si256(ymm2, mask);  // contains odd 4 bits
            ymm1 = _mm256_shuffle_epi8(
                       shuf, ymm1);  // use table look-up to sum the 4 bits
            ymm2 = _mm256_shuffle_epi8(shuf, ymm2);
            innertotal = _mm256_add_epi8(innertotal, ymm1);  // inner total
            // values in each
            // byte are bounded
            // by 8 * inner
            innertotal = _mm256_add_epi8(innertotal, ymm2);  // inner total
            // values in each
            // byte are bounded
            // by 8 * inner
        }
        innertotal = _mm256_sad_epu8(zero, innertotal);  // produces 4 64-bit
        // counters (having
        // values in [0,8 *
        // inner * 4])
        total = _mm256_add_epi64(
                    total,
                    innertotal);  // add the 4 64-bit counters to previous counter
    }
    int leftoverwords =  length % (inner * sizeof(__m256i) / sizeof(uint64_t));
    int leftover = popcnt_bitset64_weight(array + length - leftoverwords , leftoverwords);
    return leftover + _mm256_extract_epi64(total, 0) + _mm256_extract_epi64(total, 1) +
           _mm256_extract_epi64(total, 2) + _mm256_extract_epi64(total, 3);
}


int avx_lauradoux_bitset64_weight(const uint64_t *input, size_t size) {
    const __m256i M1  = _mm256_set1_epi64x(UINT64_C(0x5555555555555555));
    const __m256i M2  = _mm256_set1_epi64x(UINT64_C(0x3333333333333333));
    const __m256i M4  = _mm256_set1_epi64x(UINT64_C(0x0F0F0F0F0F0F0F0F));
    size_t i, j;
    __m256i Bit_count = _mm256_setzero_si256();
    const int wordspervector = sizeof(__m256i)/sizeof(uint64_t);
    const __m256i * INPUT = (const __m256i *)input;
    for (i = 0; i + 12 * wordspervector <= size; i += 12 * wordspervector , INPUT += 12) {
        __m256i acc = _mm256_setzero_si256();
        for (j = 0; j < 12; j += 3) {
            __m256i count1, count2, half1, half2;
            count1  =  _mm256_lddqu_si256(INPUT + j );
            count2  =  _mm256_lddqu_si256(INPUT + j + 1);
            half1   =  _mm256_lddqu_si256(INPUT + j + 2);
            half2   =  half1;
            half1  =  _mm256_and_si256(half1,M1);
            half2   = _mm256_and_si256(_mm256_srli_epi64(half2,1),M1);
            count1 = _mm256_sub_epi64(count1,_mm256_and_si256(_mm256_srli_epi64(count1,1),M1));
            count2 = _mm256_sub_epi64(count2,_mm256_and_si256(_mm256_srli_epi64(count2,1),M1));
            count1 =  _mm256_add_epi64(count1,half1);
            count2 =  _mm256_add_epi64(count2,half2);
            count1  = _mm256_add_epi64(_mm256_and_si256(count1, M2) , _mm256_and_si256(_mm256_srli_epi64(count1,2), M2));
            count1 = _mm256_add_epi64(count1,_mm256_add_epi64(_mm256_and_si256(count2 , M2) , _mm256_and_si256(_mm256_srli_epi64(count2,2), M2)));
            acc    = _mm256_add_epi64(acc,_mm256_add_epi64(_mm256_and_si256(count1 ,M4) , _mm256_and_si256(_mm256_srli_epi64(count1,4) , M4)));
        }
        const uint64_t m8  = UINT64_C(0x00FF00FF00FF00FF);
        const __m256i M8  = _mm256_set1_epi64x(m8);
        acc = _mm256_add_epi64(_mm256_and_si256(acc , M8) , _mm256_and_si256(_mm256_srli_epi64(acc,8)  , M8));
        const uint64_t m16 = UINT64_C(0x0000FFFF0000FFFF);
        acc = _mm256_and_si256(_mm256_add_epi64(acc,  _mm256_srli_epi64(acc,16)) , _mm256_set1_epi64x(m16));
        acc =  _mm256_add_epi64(acc,  _mm256_srli_epi64(acc,32));
        Bit_count = _mm256_add_epi64(Bit_count, acc);
    }
    int bit_count = _mm256_extract_epi32(Bit_count,0) + _mm256_extract_epi32(Bit_count,2) + _mm256_extract_epi32(Bit_count,4) + _mm256_extract_epi32(Bit_count,6);
    for (; i < size; i++) {
        bit_count += _mm_popcnt_u64(input[i]);
    }
    return bit_count;
}


#endif // HAVE_AVX2_INSTRUCTIONS

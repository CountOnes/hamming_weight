#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

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

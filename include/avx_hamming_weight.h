#ifndef _AVX_HAMMING_WEIGHT_H_
#define _AVX_HAMMING_WEIGHT_H_

#include <stdint.h>

#if defined(HAVE_AVX2_INSTRUCTIONS)

// compute the Hamming weight of an array of 64-bit words using AVX2 instructions
int avx2_bitset64_weight(const uint64_t * array, size_t length);

#endif // HAVE_AVX2_INSTRUCTIONS

#endif

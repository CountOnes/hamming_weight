#ifndef _AVX_HAMMING_WEIGHT_H_
#define _AVX_HAMMING_WEIGHT_H_

#include <stdint.h>

// compute the Hamming weight of an array of 64-bit words using AVX2 instructions
int avx2_bitset64_weight(const uint64_t * array, size_t length);

#endif

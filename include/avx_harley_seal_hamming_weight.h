#ifndef _AVX_HARLEY_SEAL_HAMMING_WEIGHT_H_
#define _AVX_HARLEY_SEAL_HAMMING_WEIGHT_H_

#include <stdint.h>
#include "config.h"

#ifdef HAVE_AVX2_INSTRUCTIONS

// compute the Hamming weight of an array of 64-bit words using AVX2 instructions and Harley Seal
int avx2_harley_seal_bitset64_weight(const uint64_t * data, size_t size);

#endif // HAVE_AVX2_INSTRUCTIONS

#endif

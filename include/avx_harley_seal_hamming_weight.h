#ifndef _AVX_HARLEY_SEAL_HAMMING_WEIGHT_H_
#define _AVX_HARLEY_SEAL_HAMMING_WEIGHT_H_

#include <stdint.h>
#include "config.h"

#ifdef HAVE_AVX2_INSTRUCTIONS

// compute the Hamming weight of an array of 64-bit words using AVX2 instructions and Harley Seal
int avx2_harley_seal_bitset64_weight(const uint64_t * data, size_t size);

// compute the Hamming weight of an array of 64-bit words using popcnt instruction and Harley Seal
int avx2_harley_seal_hardware_popcnt(const uint64_t * data, size_t size);

// compute the Hamming weight of an array of 64-bit words using popcnt instruction and Harley Seal, uses a small inner loop
int avx2_harley_seal_eights_hardware_popcnt(const uint64_t * data, size_t size);

// compute the Hamming weight of an array of 64-bit words using popcnt instruction and Harley Seal
int avx2_harley_seal_hardware_buffer_popcnt(const uint64_t * data, size_t size);

// compute the Hamming weight of an array of 64-bit words using AVX2 instructions and Harley Seal (version using a particular optimization proposed by N. Kurz)
int avx2_harley_seal_nate_bitset64_weight(const uint64_t * data, size_t size);


// same as avx2_harley_seal_bitset64_weight, but with more aggressive unrolling
int avx2_harley_seal_bitset64_weight_unrolled_twice(const uint64_t * data, size_t size);

#endif // HAVE_AVX2_INSTRUCTIONS

#endif

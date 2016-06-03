#ifndef _SCALAR_HAMMING_WEIGHT_H_
#define _SCALAR_HAMMING_WEIGHT_H_

#include <stdint.h>



// compute the Hamming weight of the an array of 64-bit words using a scalar Hamming weight function
int scalar_bitset64_weight(const uint64_t * input, size_t length);

// Computes the hamming weight. Attributed to Cédric Lauradoux
int lauradoux_bitset64_weight(const uint64_t *input, size_t size);


/// Harley-Seal popcount, see "Hacker's Delight" 2nd edition.
int scalar_harley_seal_bitset64_weight(const uint64_t * data, size_t size);

// same as scalar_harley_seal_bitset64_weight but works over blocks of 8 words
int scalar_harley_seal8_bitset64_weight(const uint64_t * data, size_t size);

#endif

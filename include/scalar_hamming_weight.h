#ifndef _SCALAR_HAMMING_WEIGHT_H_
#define _SCALAR_HAMMING_WEIGHT_H_

#include <stdint.h>



// compute the Hamming weight of the an array of 64-bit words using a scalar Hamming weight function
int scalar_bitset64_weight(const uint64_t * input, size_t length);

// Computes the hamming weight. Attributed to Cédric Lauradoux
int lauradoux_bitset64_weight(const uint64_t *input, size_t size);

#endif

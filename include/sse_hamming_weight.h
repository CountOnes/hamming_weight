#ifndef _SSE_HAMMING_WEIGHT_H_
#define _SSE_HAMMING_WEIGHT_H_

#include <stdint.h>
#include "config.h"

// compute the Hamming weight of an array of 64-bit words using SSE instructions
int sse_bitset64_weight(const uint64_t * array, size_t length);


// compute the Hamming weight of an array of 64-bit words using SSE instructions, two counters
int sse_twocounters_bitset64_weight(const uint64_t * array, size_t length);

// compute the Hamming weight of an array of 64-bit words using SSE instructions and popcnt
int sse_morancho_bitset64_weight(const uint64_t * array, size_t length);

#endif

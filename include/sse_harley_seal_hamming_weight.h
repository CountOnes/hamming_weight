#ifndef _SSE_HARLEY_SEAL_HAMMING_WEIGHT_H_
#define _SSE_HARLEY_SEAL_HAMMING_WEIGHT_H_

#include <stdint.h>
#include "config.h"


// compute the Hamming weight of an array of 64-bit words using SSE instructions and Harley Seal
int sse_harley_seal_bitset64_weight(const uint64_t * data, size_t size);


#endif

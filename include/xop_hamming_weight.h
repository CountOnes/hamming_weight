#ifndef _XOP_HAMMING_WEIGHT_H_
#define _XOP_HAMMING_WEIGHT_H_

#include <stdint.h>
#include "config.h"

#ifdef HAVE_XOP_INSTRUCTIONS

// compute the Hamming weight of an array of 64-bit words using XOP instructions
int xop_bitset64_weight(const uint64_t * array, size_t length);

#endif // HAVE_XOP_INSTRUCTIONS

#endif

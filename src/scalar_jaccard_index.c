#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

#include "scalar_jaccard_index.h"


// straight out of wikipedia
static uint64_t scalar_hamming_weight(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555;
    x = ((x>>2)&0x3333333333333333) + (x&0x3333333333333333);
    x += x >> 4;
    x &= 0x0f0f0f0f0f0f0f0f;
    x *= 0x0101010101010101;
    return x >> 56;
}

// compute the Hamming weight of the an array of 64-bit words using a scalar Hamming weight function
void scalar_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t length, uint64_t* sum, uint64_t* inters) {
    *sum = 0;
    *inters = 0;
    for(size_t k = 0; k < length; k++) {
    	uint64_t A = dataA[k];
    	uint64_t B = dataB[k];
    	*sum += scalar_hamming_weight(A | B);
    	*inters += scalar_hamming_weight(A & B);
    }
}


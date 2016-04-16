#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#if defined(HAVE_POPCNT_INSTRUCTION)

#include "popcnt_hamming_weight.h"


// compute the Hamming weight of an array of 64-bit words using the popcnt instruction
int popcnt_bitset64_weight(const uint64_t * input, size_t length) {
    int card = 0;
    for(size_t k = 0; k < length; k++) {
        card += _mm_popcnt_u64(input[k]);
    }
    return card;
}


// compute the Hamming weight of an array of 64-bit words using unrolled popcnt instructions
int unrolled_popcnt_bitset64_weight(const uint64_t * input, size_t length) {
    int card = 0;
    for(size_t k = 0; k < length; k+=4) {
        card += _mm_popcnt_u64(input[k]);
        card += _mm_popcnt_u64(input[k+1]);
        card += _mm_popcnt_u64(input[k+2]);
        card += _mm_popcnt_u64(input[k+3]);
    }
    return card;
}

#endif // HAVE_POPCNT_INSTRUCTION

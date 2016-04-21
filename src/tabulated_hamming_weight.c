#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

#include "tabulated_hamming_weight.h"


#include "small_table.c"

// compute the Hamming weight of an array of 8-bit words using the small table look-ups
int table_bitset8_weight(const uint8_t * input, size_t length) {
    int card = 0;
    for(size_t k = 0; k < length; k++) {
        card += small_table[input[k]];
    }
    return card;
}

#include "bigtable.c"


// compute the Hamming weight of an array of 16-bit words using the big table look-ups
int table_bitset16_weight(const uint16_t * input, size_t length) {
    int card = 0;
    for(size_t k = 0; k < length; k++) {
        card += big_table[input[k]];
    }
    return card;
}

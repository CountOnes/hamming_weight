#include "scalar_bitset.h"

#include <stdint.h>
#include <stddef.h>


// straight out of wikipedia
static uint64_t scalar_hamming_weight(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555;
    x = ((x>>2)&0x3333333333333333) + (x&0x3333333333333333);
    x += x >> 4;
    x &= 0x0f0f0f0f0f0f0f0f;
    x *= 0x0101010101010101;
    return x >> 56;
}

#define BITSET_CONTAINER_FN(opname, opsymbol)               \
int scalar_##opname(const uint64_t* array_1, const uint64_t* array_2,     \
		size_t length, uint64_t*out) {                                    \
    int32_t sum = 0;                                                      \
    size_t i = 0;                                                         \
    for (; i + 1 < length; i += 2) {                                      \
        const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]),         \
                       word_2 = (array_1[i + 1])opsymbol(array_2[i + 1]); \
        out[i] = word_1;                                                  \
        out[i + 1] = word_2;                                              \
        sum += scalar_hamming_weight(word_1);                             \
        sum += scalar_hamming_weight(word_2);                             \
    }                                                                     \
    if ( i  < length ) {                                                  \
        const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]);         \
        out[i] = word_1;                                                  \
        sum += scalar_hamming_weight(word_1);                             \
    }                                                                     \
    return sum;                                                           \
}

BITSET_CONTAINER_FN(and, &)

#undef BITSET_CONTAINER_FN






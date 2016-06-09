#ifdef HAVE_POPCNT_INSTRUCTION

#include "popcnt_bitset.h"
#include <stdint.h>
#include <stddef.h>
#include <x86intrin.h>

#define BITSET_CONTAINER_FN(opname, opsymbol)               \
int popcnt_##opname(const uint64_t* array_1, const uint64_t* array_2,     \
		size_t length, uint64_t*out) {                                    \
    int32_t sum = 0;                                                      \
    size_t i = 0;                                                         \
    for (; i + 1 < length; i += 2) {                                      \
        const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]),         \
                       word_2 = (array_1[i + 1])opsymbol(array_2[i + 1]); \
        out[i] = word_1;                                                  \
        out[i + 1] = word_2;                                              \
        sum += _mm_popcnt_u64(word_1);                                    \
        sum += _mm_popcnt_u64(word_2);                                    \
    }                                                                     \
    if ( i  < length ) {                                                  \
        const uint64_t word_1 = (array_1[i])opsymbol(array_2[i]);         \
        out[i] = word_1;                                                  \
        sum += _mm_popcnt_u64(word_1);                                    \
    }                                                                     \
    return sum;                                                           \
}

BITSET_CONTAINER_FN(and, &)

#undef BITSET_CONTAINER_FN

#endif

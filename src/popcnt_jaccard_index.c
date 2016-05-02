#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#ifdef HAVE_POPCNT_INSTRUCTION

#include "popcnt_jaccard_index.h"


// compute the Jaccard index of an array of 64-bit words using the popcnt instruction
int popcnt_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* sum, uint64_t* inters) {
    uint64_t s = 0;
    uint64_t i = 0;
    for(size_t k = 0; k < n; k++) {
        
        s += _mm_popcnt_u64(dataA[k] | dataB[k]);
        i += _mm_popcnt_u64(dataA[k] & dataB[k]);
    }

    *sum    = s;
    *inters = i;

    return 0;
}

// compute the Jaccard index of an array of 64-bit words using unrolled popcnt instructions
int unrolled_popcnt_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* sum, uint64_t* inters) {
    uint64_t s = 0;
    uint64_t i = 0;

    size_t k =0;
    for(; k + 7 < n; k+=8) {
        i += _mm_popcnt_u64(dataA[k+0] & dataB[k+0]);
        s += _mm_popcnt_u64(dataA[k+0] | dataB[k+0]);
        i += _mm_popcnt_u64(dataA[k+1] & dataB[k+1]);
        s += _mm_popcnt_u64(dataA[k+1] | dataB[k+1]);
        i += _mm_popcnt_u64(dataA[k+2] & dataB[k+2]);
        s += _mm_popcnt_u64(dataA[k+2] | dataB[k+2]);
        i += _mm_popcnt_u64(dataA[k+3] & dataB[k+3]);
        s += _mm_popcnt_u64(dataA[k+3] | dataB[k+3]);
        i += _mm_popcnt_u64(dataA[k+4] & dataB[k+4]);
        s += _mm_popcnt_u64(dataA[k+4] | dataB[k+4]);
        i += _mm_popcnt_u64(dataA[k+5] & dataB[k+5]);
        s += _mm_popcnt_u64(dataA[k+5] | dataB[k+5]);
        i += _mm_popcnt_u64(dataA[k+6] & dataB[k+6]);
        s += _mm_popcnt_u64(dataA[k+6] | dataB[k+6]);
        i += _mm_popcnt_u64(dataA[k+7] & dataB[k+7]);
        s += _mm_popcnt_u64(dataA[k+7] | dataB[k+7]);
    }
    for(; k + 3 < n; k+=4) {
        i += _mm_popcnt_u64(dataA[k+0] & dataB[k+0]);
        s += _mm_popcnt_u64(dataA[k+0] | dataB[k+0]);
        i += _mm_popcnt_u64(dataA[k+1] & dataB[k+1]);
        s += _mm_popcnt_u64(dataA[k+1] | dataB[k+1]);
        i += _mm_popcnt_u64(dataA[k+2] & dataB[k+2]);
        s += _mm_popcnt_u64(dataA[k+2] | dataB[k+2]);
        i += _mm_popcnt_u64(dataA[k+3] & dataB[k+3]);
        s += _mm_popcnt_u64(dataA[k+3] | dataB[k+3]);
    }
    for(; k < n; k++) {
        i += _mm_popcnt_u64(dataA[k] & dataB[k]);
        s += _mm_popcnt_u64(dataA[k] | dataB[k]);
    }

    *sum    = s;
    *inters = i;

    return 0;
}

#endif // HAVE_POPCNT_INSTRUCTION

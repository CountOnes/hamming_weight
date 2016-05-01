#ifndef _SSE_JACCARD_INDEX_H_
#define _SSE_JACCARD_INDEX_H_

#include <stdint.h>
#include "config.h"

/*
    Compute Jaccard index coefficients:

    sum    := popcount(A | B)
    inters := popcount(A & B)

    Returns 0.
*/
int sse_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t length, uint64_t* sum, uint64_t* inters);

#endif

#ifndef _SSE_JACCARD_INDEX_H_
#define _SSE_JACCARD_INDEX_H_

#include <stdint.h>
#include "config.h"

/*
    Compute Jaccard index coefficients:

    j_union := popcount(A | B)
    j_inter := popcount(A & B)
*/
void sse_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t length, uint64_t* j_union, uint64_t* j_inter);

#endif

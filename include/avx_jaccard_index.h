#ifndef _AVX_JACCARD_INDEX_H_
#define _AVX_JACCARD_INDEX_H_

#include <stdint.h>
#include "config.h"

#ifdef HAVE_AVX2_INSTRUCTIONS

// compute the Jaccard index of an array of 64-bit words using AVX2 instructions
void avx2_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t length, uint64_t* sum, uint64_t* inters);

#endif // HAVE_AVX2_INSTRUCTIONS

#endif

#ifndef _POPCNT_JACCARD_INDEX_H_
#define _POPCNT_JACCARD_INDEX_H_

#include <stdint.h>
#include "config.h"


#ifdef HAVE_POPCNT_INSTRUCTION

// compute the Jaccard index of an array of 64-bit words using the popcnt instruction
void popcnt_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* j_union, uint64_t* j_inter);

// compute the Jaccard index of an array of 64-bit words using the popcnt instruction
void slightly_unrolled_popcnt_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* j_union, uint64_t* j_inter);

// compute the Hamming weight of an array of 64-bit words using unrolled popcnt instructions
void unrolled_popcnt_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t n, uint64_t* j_union, uint64_t* j_inter);

#endif // HAVE_POPCNT_INSTRUCTION

#endif

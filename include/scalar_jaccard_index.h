#ifndef SCALAR_JACCARD_INDEX_H
#define SCALAR_JACCARD_INDEX_H

#include <stdint.h>


/*
    Compute Jaccard index coefficients:

    j_union := popcount(A | B)
    j_inter := popcount(A & B)
*/

// compute the jaccard index coefficients of the an array of 64-bit words using a scalar Hamming weight function
void scalar_jaccard_index(const uint64_t* dataA, const uint64_t* dataB, size_t length, uint64_t* j_union, uint64_t* j_inter);


#endif

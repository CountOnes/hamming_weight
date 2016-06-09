#ifndef INCLUDE_AVX_BITSET_H_
#define INCLUDE_AVX_BITSET_H_

#include <stdint.h>
#include <stddef.h>
#include "config.h"

#ifdef HAVE_AVX2_INSTRUCTIONS

/* Computes the intersection of bitsets `src_1' and `src_2' into `dst' and
 * return the cardinality. */
int avx_lookup_and(const uint64_t* dataA, const uint64_t* dataB,size_t length,
		uint64_t*dst);

/* Computes the intersection of bitsets `src_1' and `src_2' into `dst' and
 * return the cardinality. */
int avx_harley_seal_and(const uint64_t* dataA, const uint64_t* dataB,size_t length,
		uint64_t*dst);

#endif
#endif /* INCLUDE_AVX_BITSET_H_ */

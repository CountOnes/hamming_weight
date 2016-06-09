#ifndef INCLUDE_POPCNT_BITSET_H_
#define INCLUDE_POPCNT_BITSET_H_


#include <stdint.h>
#include <stddef.h>
#include "config.h"

#ifdef HAVE_POPCNT_INSTRUCTION

/* Computes the intersection of bitsets `src_1' and `src_2' into `dst' and
 * return the cardinality. */
int popcnt_and(const uint64_t* dataA, const uint64_t* dataB, size_t length,
		uint64_t*dst);

#endif

#endif /* INCLUDE_POPCNT_BITSET_H_ */

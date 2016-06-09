/*
 * scalar_bitset.h
 *
 *  Created on: Jun 9, 2016
 *      Author: lemire
 */

#ifndef INCLUDE_SCALAR_BITSET_H_
#define INCLUDE_SCALAR_BITSET_H_



#include <stdint.h>
#include <stddef.h>
#include "config.h"


/* Computes the intersection of bitsets `src_1' and `src_2' into `dst' and
 * return the cardinality. */
int scalar_and(const uint64_t* dataA, const uint64_t* dataB, size_t length,
		uint64_t*dst);



#endif /* INCLUDE_SCALAR_BITSET_H_ */

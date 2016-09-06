#ifndef _POPCNT_HAMMING_WEIGHT_H_
#define _POPCNT_HAMMING_WEIGHT_H_

#include <stdint.h>
#include "config.h"


#ifdef HAVE_POPCNT_INSTRUCTION

// compute the Hamming weight of an array of 64-bit words using the popcnt instruction
int popcnt_bitset64_weight(const uint64_t * input, size_t length);


// compute the Hamming weight of an array of 64-bit words using unrolled popcnt instructions
int unrolled_popcnt_bitset64_weight(const uint64_t * input, size_t length);

// compute Hamming weight using popcnt instruction through assembly
// This code is from Alex Yee.
int yee_popcnt_bitset64_weight(const uint64_t* buf, size_t len);

// compute Hamming weight using popcnt instruction through assembly
// This code was originally designed by Nathan Kurz.
int nate_popcnt_bitset64_weight(const uint64_t* buf, size_t len);

#endif // HAVE_POPCNT_INSTRUCTION

#endif

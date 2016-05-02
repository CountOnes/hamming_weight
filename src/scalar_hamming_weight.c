#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>

#include "scalar_hamming_weight.h"


// straight out of wikipedia
static uint64_t scalar_hamming_weight(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555;
    x = ((x>>2)&0x3333333333333333) + (x&0x3333333333333333);
    x += x >> 4;
    x &= 0x0f0f0f0f0f0f0f0f;
    x *= 0x0101010101010101;
    return x >> 56;
}

/**
 * Computes the hamming weight. Attributed to Cédric Lauradoux.
 */
int lauradoux_bitset64_weight(const uint64_t *input, size_t size) {
    const uint64_t m1  = UINT64_C(0x5555555555555555);
    const uint64_t m2  = UINT64_C(0x3333333333333333);
    const uint64_t m4  = UINT64_C(0x0F0F0F0F0F0F0F0F);
    const uint64_t m8  = UINT64_C(0x00FF00FF00FF00FF);
    const uint64_t m16 = UINT64_C(0x0000FFFF0000FFFF);
    const uint64_t h01 = UINT64_C(0x0101010101010101);

    uint64_t count1, count2, half1, half2, acc;
    uint64_t x;
    size_t i, j;
    size_t limit = size - size % 12;
    int bit_count = 0;
    for (i = 0; i < limit; i += 12, input += 12) {
        acc = 0;
        for (j = 0; j < 12; j += 3) {
            count1  =  input[j + 0];
            count2  =  input[j + 1];
            half1   =  input[j + 2];
            half2   =  input[j + 2];
            half1  &=  m1;
            half2   = (half2  >> 1) & m1;
            count1 -= (count1 >> 1) & m1;
            count2 -= (count2 >> 1) & m1;
            count1 +=  half1;
            count2 +=  half2;
            count1  = (count1 & m2) + ((count1 >> 2) & m2);
            count1 += (count2 & m2) + ((count2 >> 2) & m2);
            acc    += (count1 & m4) + ((count1 >> 4) & m4);
        }
        acc = (acc & m8) + ((acc >>  8)  & m8);
        acc = (acc       +  (acc >> 16)) & m16;
        acc =  acc       +  (acc >> 32);
        bit_count += (int) acc;
    }
    for (i = 0; i < size - limit; i++) {
        x = input[i];
        x =  x       - ((x >> 1)  & m1);
        x = (x & m2) + ((x >> 2)  & m2);
        x = (x       +  (x >> 4)) & m4;
        bit_count += (int) ((x * h01) >> 56);
    }
    return bit_count;
}



// compute the Hamming weight of the an array of 64-bit words using a scalar Hamming weight function
int scalar_bitset64_weight(const uint64_t * input, size_t length) {
    int card = 0;
    for(size_t k = 0; k < length; k++) {
        card += scalar_hamming_weight(input[k]);
    }
    return card;
}


/// Carry-save adder (CSA).
/// @see Chapter 5 in "Hacker's Delight".
///
static inline void CSA(uint64_t* h, uint64_t* l, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t u = a ^ b;
  *h = (a & b) | (u & c);
  *l = u ^ c;
}

/// Harley-Seal popcount (4th iteration).
/// The Harley-Seal popcount algorithm is one of the fastest algorithms
/// for counting 1 bits in an array using only integer operations.
/// This implementation uses only 5.69 instructions per 64-bit word.
/// @see Chapter 5 in "Hacker's Delight" 2nd edition.
///
int scalar_harley_seal_bitset64_weight(const uint64_t * data, size_t size)  {
  uint64_t total = 0;
  uint64_t ones = 0, twos = 0, fours = 0, eights = 0, sixteens = 0;
  uint64_t twosA, twosB, foursA, foursB, eightsA, eightsB;
  uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for(; i < limit; i += 16)
  {
    CSA(&twosA, &ones, ones, data[i+0], data[i+1]);
    CSA(&twosB, &ones, ones, data[i+2], data[i+3]);
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, data[i+4], data[i+5]);
    CSA(&twosB, &ones, ones, data[i+6], data[i+7]);
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsA, &fours, fours, foursA, foursB);
    CSA(&twosA, &ones, ones, data[i+8], data[i+9]);
    CSA(&twosB, &ones, ones, data[i+10], data[i+11]);
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, data[i+12], data[i+13]);
    CSA(&twosB, &ones, ones, data[i+14], data[i+15]);
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsB, &fours, fours, foursA, foursB);
    CSA(&sixteens, &eights, eights, eightsA, eightsB);

    total += scalar_hamming_weight(sixteens);
  }

  total *= 16;
  total += 8 * scalar_hamming_weight(eights);
  total += 4 * scalar_hamming_weight(fours);
  total += 2 * scalar_hamming_weight(twos);
  total += 1 * scalar_hamming_weight(ones);

  for(; i < size; i++)
    total += scalar_hamming_weight(data[i]);

  return total;
}

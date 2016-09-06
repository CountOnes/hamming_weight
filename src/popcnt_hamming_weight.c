#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <x86intrin.h>


#ifdef HAVE_POPCNT_INSTRUCTION

#include "popcnt_hamming_weight.h"


// compute the Hamming weight of an array of 64-bit words using the popcnt instruction
int popcnt_bitset64_weight(const uint64_t * input, size_t length) {
    int card = 0;
    for(size_t k = 0; k < length; k++) {
        card += _mm_popcnt_u64(input[k]);
    }
    return card;
}

// compute the Hamming weight of an array of 64-bit words using unrolled popcnt instructions
int unrolled_popcnt_bitset64_weight(const uint64_t * input, size_t length) {
    int card = 0;
    size_t k =0;
    for(; k + 7 < length; k+=8) {
        card += _mm_popcnt_u64(input[k]);
        card += _mm_popcnt_u64(input[k+1]);
        card += _mm_popcnt_u64(input[k+2]);
        card += _mm_popcnt_u64(input[k+3]);
        card += _mm_popcnt_u64(input[k+4]);
        card += _mm_popcnt_u64(input[k+5]);
        card += _mm_popcnt_u64(input[k+6]);
        card += _mm_popcnt_u64(input[k+7]);
    }
    for(; k + 3 < length; k+=4) {
        card += _mm_popcnt_u64(input[k]);
        card += _mm_popcnt_u64(input[k+1]);
        card += _mm_popcnt_u64(input[k+2]);
        card += _mm_popcnt_u64(input[k+3]);
    }
    for(; k < length; k++) {
        card += _mm_popcnt_u64(input[k]);
    }
    return card;
}

#define ASM_SUM_POPCNT(total, tmp1, tmp2, tmp3, tmp4, neg, end)         \
    __asm volatile ("1:\n"                                              \
                    "popcnt 0(%6,%5,1), %1\n"                           \
                    "popcnt 8(%6,%5,1), %2\n"                           \
                    "popcnt 16(%6,%5,1), %3\n"                          \
                    "popcnt 24(%6,%5,1), %4\n"                          \
                    "add %1, %0\n"                                      \
                    "add %2, %0\n"                                      \
                    "add %3, %0\n"                                      \
                    "add %4, %0\n"                                      \
                    "add $32, %5\n"                                     \
                    "jnz 1b\n" :                                        \
                    "+&r" (total),                                      \
                    "=&r" (tmp1),                                       \
                    "=&r" (tmp2),                                       \
                    "=&r" (tmp3),                                       \
                    "=&r" (tmp4),                                       \
                    "+&r" (neg) :                                       \
                    "r" (end)                                           \
                    )


// designed originally by Nathan Kurz, adapted by D. Lemire.
int nate_popcnt_bitset64_weight(const uint64_t* buf, size_t len) {
    uint64_t count1, count2, count3, count4;
    uint64_t total_count = 0;
    size_t len4 = len / 4 * 4;// a simple mask would do
    for (size_t i = 0; i < (len % 4); ++i) {
        total_count += _mm_popcnt_u64(buf[i]);
    }
    const uint8_t *end = (const uint8_t *) buf + len * sizeof(uint64_t);
    int64_t neg = -(len4 * sizeof(uint64_t));
    ASM_SUM_POPCNT(total_count, count1, count2, count3, count4, neg, end);
    return total_count;
}

// compute Hamming weight using popcnt instruction through assembly
// This code is from Alex Yee.
int yee_popcnt_bitset64_weight(const uint64_t* buf, size_t len) {
  uint64_t cnt[4];
  for (int i = 0; i < 4; ++i) {
    cnt[i] = 0;
  }
  size_t i = 0;
  for (; i + 3 < len; i+=4) {
    __asm__ __volatile__(
	    "popcnt %4, %4  \n\t"
	    "add %4, %0     \n\t"
	    "popcnt %5, %5  \n\t"
	    "add %5, %1     \n\t"
	    "popcnt %6, %6  \n\t"
	    "add %6, %2     \n\t"
	    "popcnt %7, %7  \n\t"
	    "add %7, %3     \n\t"
	    : "+r" (cnt[0]), "+r" (cnt[1]), "+r" (cnt[2]), "+r" (cnt[3])
	    : "r"  (buf[i]), "r"  (buf[i+1]), "r"  (buf[i+2]), "r"  (buf[i+3])
		);
  }
  int answer =  cnt[0] + cnt[1] + cnt[2] + cnt[3];
  for(;i < len; ++i) {
    answer += _mm_popcnt_u64(buf[i]);
  }
  return answer;
}

#endif // HAVE_POPCNT_INSTRUCTION

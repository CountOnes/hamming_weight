#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>


#include "benchmark.h"
#include "jaccard_index.h"

void *aligned_malloc(size_t alignment, size_t size) {
    void *mem;
    if (posix_memalign(&mem, alignment, size)) exit(1);
    return mem;
}


bool compare(uint64_t jaccard_sum, uint64_t jaccard_int,
		uint64_t jaccard_sum_correct, uint64_t jaccard_int_correct) {
    return (jaccard_sum == jaccard_sum_correct) && (jaccard_int == jaccard_int_correct);
}

void demo(int size) {
    printf("size = %d words or %lu bytes \n",size,  size*sizeof(uint64_t));
    int repeat = 500;
    uint64_t * prec = aligned_malloc(32,size * sizeof(uint64_t));
    uint64_t * prec2 = aligned_malloc(32,size * sizeof(uint64_t));
    uint64_t jaccard_sum, jaccard_int;
    uint64_t jaccard_sum_correct, jaccard_int_correct;

    for(int k = 0; k < size; ++k) {
        prec[k]  = -k;
        prec2[k] = k;
    }
    scalar_jaccard_index(prec,prec2,size,&jaccard_sum_correct,&jaccard_int_correct);

    BEST_TIME_CHECK(scalar_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),
    		compare(jaccard_sum,jaccard_int,jaccard_sum_correct,jaccard_int_correct),, repeat, size);

#if defined(HAVE_POPCNT_INSTRUCTION)
    BEST_TIME_CHECK(popcnt_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),
    		compare(jaccard_sum,jaccard_int,jaccard_sum_correct,jaccard_int_correct),, repeat, size);
    BEST_TIME_CHECK(slightly_unrolled_popcnt_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),
    		compare(jaccard_sum,jaccard_int,jaccard_sum_correct,jaccard_int_correct),, repeat, size);
    BEST_TIME_CHECK(unrolled_popcnt_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),
    		compare(jaccard_sum,jaccard_int,jaccard_sum_correct,jaccard_int_correct),, repeat, size);
    BEST_TIME_CHECK(sse_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),
    		compare(jaccard_sum,jaccard_int,jaccard_sum_correct,jaccard_int_correct),, repeat, size);
#else
    printf("no popcnt instruction\n");
#endif

#if defined(HAVE_AVX2_INSTRUCTIONS)
    BEST_TIME_CHECK(avx2_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),
    		compare(jaccard_sum,jaccard_int,jaccard_sum_correct,jaccard_int_correct),, repeat, size);
#else
    printf("no AVX2 instructions\n");
#endif

    free(prec);
    free(prec2);
    printf("\n");
}

int main() {
    for(int w = 8; w <= 8192; w *= 2) {
      demo(w);
    }
    return 0;
}

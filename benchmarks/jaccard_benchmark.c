#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>


#include "benchmark.h"
#include "sse_jaccard_index.h"
#include "popcnt_jaccard_index.h"
void *aligned_malloc(size_t alignment, size_t size) {
    void *mem;
    if (posix_memalign(&mem, alignment, size)) exit(1);
    return mem;
}

void demo(int size) {
    printf("size = %d words or %lu bytes \n",size,  size*sizeof(uint64_t));
    int repeat = 500;
    uint64_t * prec = aligned_malloc(32,size * sizeof(uint64_t));
    uint64_t * prec2 = aligned_malloc(32,size * sizeof(uint64_t));
    uint64_t jaccard_sum, jaccard_int;
    for(int k = 0; k < size; ++k) {
        prec[k]  = -k;
        prec2[k] = k;
    }
#if defined(HAVE_POPCNT_INSTRUCTION)
    BEST_TIME(popcnt_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),0,, repeat, size);
    printf("%ld/%ld\n", jaccard_sum, jaccard_int);
    BEST_TIME(unrolled_popcnt_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),0,, repeat, size);
    printf("%ld/%ld\n", jaccard_sum, jaccard_int);
    BEST_TIME(sse_jaccard_index(prec,prec2,size,&jaccard_sum,&jaccard_int),0,, repeat, size);
    printf("%ld/%ld\n", jaccard_sum, jaccard_int);
#else
    printf("no popcnt instruction\n");
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

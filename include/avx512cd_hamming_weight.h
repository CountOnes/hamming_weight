#ifndef _AVX512CD_HAMMING_WEIGHT_
#define _AVX512CD_HAMMING_WEIGHT_


#ifdef HAVE_AVX512CD_INSTRUCTIONS

// naive, loop-based counting
uint64_t avx512cd_naive(const uint64_t * data, size_t size);

#endif // HAVE_AVX512CD_INSTRUCTIONS

#endif // _AVX512CD_HAMMING_WEIGHT_

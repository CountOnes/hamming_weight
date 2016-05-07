#ifndef _AVX512F_HAMMING_WEIGHT_
#define _AVX512F_HAMMING_WEIGHT_


#ifdef HAVE_AVX512F_INSTRUCTIONS

// AVX512F version of Harley-Seal, using only "foundation" instruction
uint64_t avx512f_harley_seal(const uint64_t * data, size_t size);

#endif // HAVE_AVX512F_INSTRUCTIONS

#endif // _AVX512F_HAMMING_WEIGHT_

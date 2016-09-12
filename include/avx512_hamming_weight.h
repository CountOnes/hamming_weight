#ifndef _AVX512_HAMMING_WEIGHT_
#define _AVX512_HAMMING_WEIGHT_


#ifdef HAVE_AVX512_INSTRUCTIONS

// register-level popcount using vpermb instruction (lookup in a ZMM register)
uint64_t avx512_vpermb(const uint64_t * data, size_t size);

// register-level popcount using vperm2b instruction (lookup in two ZMM registers)
uint64_t avx512_vperm2b(const uint64_t * data, size_t size);

#endif // HAVE_AVX512_INSTRUCTIONS

#endif // _AVX512_HAMMING_WEIGHT_

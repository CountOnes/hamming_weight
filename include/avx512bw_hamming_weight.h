#ifndef _AVX512BW_HAMMING_WEIGHT_
#define _AVX512BW_HAMMING_WEIGHT_


#ifdef HAVE_AVX512BW_INSTRUCTIONS

// register-level popcount using vpermb instruction (lookup in a ZMM register)
uint64_t avx512_vpermb(const uint64_t * data, size_t size);

// register-level popcount using vperm2b instruction (lookup in two ZMM registers)
uint64_t avx512_vperm2b(const uint64_t * data, size_t size);

// register-level popcount using vperm2b instruction (lookup in two ZMM registers)
// another way of handling the 7th bits
uint64_t avx512_vperm2b_ver2(const uint64_t * data, size_t size);

#endif // HAVE_AVX512BW_INSTRUCTIONS

#endif // _AVX512_HAMMING_WEIGHT_

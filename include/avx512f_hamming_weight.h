#ifndef _AVX512F_HAMMING_WEIGHT_
#define _AVX512F_HAMMING_WEIGHT_


#ifdef HAVE_AVX512F_INSTRUCTIONS

// AVX512F version of Harley-Seal, using only "foundation" instruction
uint64_t avx512f_harley_seal(const uint64_t * data, size_t size);

// AVX512F version of Harley-Seal, using only "foundation" instruction + popcnt instruction
uint64_t avx512f_harley_seal__hardware_popcnt(const uint64_t * data, size_t size);

// AVX512F version of Harley-Seal, using only "foundation" instruction + popcnt instruction
// changed order of computations
uint64_t avx512f_harley_seal__hardware_popcnt_2(const uint64_t * data, size_t size);

// AVX512F using gather instruction
uint64_t avx512f_gather(const uint64_t * data, size_t size);
void avx512f_gather_init();

#endif // HAVE_AVX512F_INSTRUCTIONS

#endif // _AVX512F_HAMMING_WEIGHT_

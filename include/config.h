#ifndef _CONFIG_H_
#define _CONFIG_H_

#ifdef __AVX2__
#define HAVE_AVX2_INSTRUCTIONS 1
#endif //__AVX2__

#ifdef __AVX512BW__
#define HAVE_AVX512BW_INSTRUCTIONS 1
#endif // __AVX512BW__

#ifdef __AVX512F__
#define HAVE_AVX512F_INSTRUCTIONS 1
#endif // __AVX512F__

#ifdef __POPCNT__
#define HAVE_POPCNT_INSTRUCTION 1
#endif

#endif //_CONFIG_H_

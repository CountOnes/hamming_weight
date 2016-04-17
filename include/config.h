#ifndef _CONFIG_H_
#define _CONFIG_H_

#ifdef __AVX2__
#define HAVE_AVX2_INSTRUCTIONS 1
#endif //__AVX2__

#ifdef __POPCNT__
#define HAVE_POPCNT_INSTRUCTION 1
#endif

#endif //_CONFIG_H_

# hamming_weight
C library to compute the Hamming weight of arrays. The Hamming weight is the number
of ones in a stream of bits. Computing this count quickly has important applications
in indexing, machine learning and so forth.


Usage
-------

```bash
make
./basic_benchmark
```

If a CPU doesn't support AVX2 define SSE. If a CPU doesn't support popcnt instruction
then define NOPOPCNT.

```
export SSE=1 NOPOPCNT=1; make
./basic_benchmark
```

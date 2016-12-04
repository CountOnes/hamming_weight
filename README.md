# hamming_weight
C library to compute the Hamming weight of arrays. The Hamming weight is the number
of ones in a stream of bits. Computing this count quickly has important applications
in indexing, machine learning, cryptography and so forth.

Library has several highly optimized implementations, which use `popcnt`, SSE,
AVX2, and AVX512 instructions.

Please note that an AVX512F variant is under development, AVX512BW is just a
proof of concept (no hardware support yet), and AVX512CD was written to show
it's feasible.

Paper
------

[Faster Population Counts using AVX2 Instructions](https://arxiv.org/abs/1611.07612)


Usage
-------

```bash
make
./unit
./basic_benchmark
```

Building
---------

It's assumed that target CPU supports both AVX2 and ``popcnt``.

* If a CPU doesn't support AVX2 define **SSE**.
* If a CPU doesn't support popcnt instruction then define **NOPOPCNT**.
* If you want to build AVX512 variants, define **AVX512F** or **AVX512BW** or **AVX512CD**.
* If you want to build for AMD XOP, define **XOP**.

For example:

```
export SSE=1 NOPOPCNT=1; make
./basic_benchmark

export AVX512F=1; make
./basic_benchmark
```

#!/bin/bash

function compile {
    make clean
    make || exit
}

for compiler in gcc-5 clang icc
do
    if command -v $compiler > /dev/null
    then
        export CC=$compiler
        echo "selecting compiler $compiler"

        echo "AVX2 build"
        export SSE=0
        export AVX512=0
        export AVX512F=0
        compile $compiler

        echo "SSE build"
        export SSE=1
        export AVX512=0
        export AVX512F=0
        compile $compiler

        echo "AVX512 build"
        export SSE=0
        export AVX512=1
        export AVX512F=0
        compile $compiler

        echo "AVX512F build"
        export SSE=0
        export AVX512=0
        export AVX512F=1
        compile $compiler
    fi
done

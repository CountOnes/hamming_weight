#!/bin/bash

function compile {
    if command -v $1 > /dev/null
    then
        echo "selecting compiler $1"
        make clean
        export CC=$1
        make || exit
        ./basic_benchmark   | ./format.py > popcount.$1.txt
        ./jaccard_benchmark | ./format.py > jaccard.$1.txt
    fi
}

for compiler in gcc clang icc
do
    compile $compiler
done

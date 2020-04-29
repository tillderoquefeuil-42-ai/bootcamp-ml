#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./createExDir original_dir number_of_dir"
elif [ -z "$2" ]; then
    echo "Usage: ./createExDir original_dir number_of_dir"
else
    for ((i=0;i<=$2;i++)); do
        name=$(printf "ex%02d" $i)
        mkdir -p $1/$name
        touch $1/$name/test.py
    done
fi
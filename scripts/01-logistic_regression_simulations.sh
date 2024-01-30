#!/bin/bash
conda activate env
cd ../src

for dgp in {0..2} 
do
    for N in {0..3}
    do
        for P in {0..2}
        do
            python _03_simulations.py --dgp $dgp --n $N --p $P
        done
    done
done

#!/bin/bash

#python batchmatmul.py 2>&1 | tee tuninglog

[ -d energyresults ] && rm -rf energyresults
mkdir energyresults

for i in {0..9}
do
    echo "Running measure.py with --I $i" | tee energyresults/"top$((i+1))"
    python measure_a100.py --I "$i" 2>&1 | tee -a energyresults/"top$((i+1))"
done
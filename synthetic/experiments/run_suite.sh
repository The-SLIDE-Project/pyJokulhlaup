#!/bin/bash
# The actual (serial) computation, for a default model run

row_count=$(($(wc -l < './suite_parameters.csv') - 1))
for i in $(seq 28 $row_count); do
    eval "python3.12 -m src.run_job './suite_parameters.csv' $i"
done


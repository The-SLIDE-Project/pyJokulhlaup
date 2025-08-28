#!/bin/bash
# The actual (serial) computation, for a default model run

row_count=$(($(wc -l < './main_figure_parameters.csv') - 1))
for i in $(seq 1 $row_count); do
    eval "python3.12 -m src.run_job './main_figure_parameters.csv' $i"
done

sleep 60m
osascript -e 'tell app "System Events" to shut down'
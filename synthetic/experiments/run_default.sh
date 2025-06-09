#!/bin/bash
# The actual (serial) computation, for a default model run

eval "python3.12 -m src.run_job './suite_parameters.csv' 1"


#!/bin/bash

# 
sbatch --array=1,2,3,4,5,6,7 ./script/eval.sbatch $1
#!/bin/bash

# 
sbatch --array=1,2,3,4,7 ./script/eval.sbatch $1 $2

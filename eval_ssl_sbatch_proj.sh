#!/bin/bash

# 
sbatch --array=1,2,3,4,7 ./script/eval_ssl_proj.sbatch $1 $2 $3 $4 $5

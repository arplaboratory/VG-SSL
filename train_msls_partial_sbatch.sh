#!/bin/bash

# PRETRAIN
sbatch --export=ALL,PRETRAIN=imagenet,SEED=0 ./script/train_msls_partial.sbatch 
sbatch --export=ALL,PRETRAIN=imagenet,SEED=1 ./script/train_msls_partial.sbatch 
sbatch --export=ALL,PRETRAIN=imagenet,SEED=2 ./script/train_msls_partial.sbatch 

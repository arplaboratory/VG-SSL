#!/bin/bash

# PRETRAIN
sbatch --export=ALL,SSL=byol,SEED=0 ./script/train_msls_ssl.sbatch 
sbatch --export=ALL,SSL=simsiam,SEED=0 ./script/train_msls_ssl.sbatch 

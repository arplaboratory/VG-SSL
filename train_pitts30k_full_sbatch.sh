#!/bin/bash

sbatch ./script/train_pitts30k_full.sbatch --export=ALL,PRETRAIN=simclr,SEED=0
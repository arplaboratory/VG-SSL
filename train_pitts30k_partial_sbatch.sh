#!/bin/bash

# PRETRAIN
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=simclr,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=simclr,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=simclr,SEED=2
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=moco,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=moco,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=moco,SEED=2
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=mocov2,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=mocov2,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=mocov2,SEED=2
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=byol,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=byol,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=byol,SEED=2
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=vicreg,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=vicreg,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=vicreg,SEED=2
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=swav,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=swav,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=swav,SEED=2
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=bt,SEED=0
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=bt,SEED=1
sbatch ./script/train_pitts30k_partial.sbatch --export=ALL,PRETRAIN=bt,SEED=2

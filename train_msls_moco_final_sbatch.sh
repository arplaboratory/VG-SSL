#!/bin/bash

# PRETRAIN
sbatch --export=ALL,SSL=simclr,SEED=0,LR=1e-5,BATCH=64,PROJ=2048 ./script/train_msls_ssl_long_simclr.sbatch
sbatch --export=ALL,SSL=mocov2,SEED=0,LR=1e-5,BATCH=64,PROJ=2048 ./script/train_msls_ssl_long_simclr.sbatch
sbatch --export=ALL,SSL=simclr,SEED=0,LR=1e-5,BATCH=64,PROJ=65536 ./script/train_msls_ssl_long_simclr.sbatch
sbatch --export=ALL,SSL=mocov2,SEED=0,LR=1e-5,BATCH=64,PROJ=65536 ./script/train_msls_ssl_long_simclr.sbatch
sbatch --export=ALL,SSL=simclr,SEED=0,LR=1e-5,BATCH=64,PROJ=2048 ./script/train_msls_ssl_long_simclr_neg1.sbatch
sbatch --export=ALL,SSL=mocov2,SEED=0,LR=1e-5,BATCH=64,PROJ=2048 ./script/train_msls_ssl_long_simclr_neg1.sbatch
sbatch --export=ALL,SSL=simclr,SEED=0,LR=1e-5,BATCH=64,PROJ=65536 ./script/train_msls_ssl_long_simclr_neg1.sbatch
sbatch --export=ALL,SSL=mocov2,SEED=0,LR=1e-5,BATCH=64,PROJ=65536 ./script/train_msls_ssl_long_simclr_neg1.sbatch

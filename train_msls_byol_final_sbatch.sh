#!/bin/bash

sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=1.0 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=1.0 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=0.5 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=0.5 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=0.25 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=0.25 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=0.0 ./script/train_msls_ssl_long_byol_neg.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=0.0 ./script/train_msls_ssl_long_byol_neg.sbatch

sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-3,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-3,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-5,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-5,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch

sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=3 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=3 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=byol,SEED=1,LR=1e-4,BATCH=64,PROJ=2048,LAY=1 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=1,LR=1e-4,BATCH=64,PROJ=2048,LAY=1 ./script/train_msls_ssl_long_byol_neg025.sbatch

sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=1024,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=1024,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=4096,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=4096,LAY=2 ./script/train_msls_ssl_long_byol_neg025.sbatch

sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025_128.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025_128.sbatch

sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025_256.sbatch
sbatch --export=ALL,SSL=simsiam,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2 ./script/train_msls_ssl_long_byol_neg025_256.sbatch
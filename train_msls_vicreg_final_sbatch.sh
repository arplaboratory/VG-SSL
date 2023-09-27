#!/bin/bash

# None vs random vs partial
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=0,FC=1024,NEGQ=0 ./script/train_msls_ssl_long_vicreg_neg_random_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=0,FC=1024,NEGQ=0 ./script/train_msls_ssl_long_vicreg_neg_random_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_random_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_random_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=1024,NEGQ=1 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=1024,NEGQ=1 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch

# # Different number of layers
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=0,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=0,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=1,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=1,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=3,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=3,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch

# # Different number of proj dim
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=4096,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=4096,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=1024,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=1024,LAY=2,NEG=1000,FC=1024,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch

# # Different number of fc dim
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=512,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=512,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=vicreg,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=2048,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
sbatch --export=ALL,SSL=bt,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=1000,FC=2048,NEGQ=10 ./script/train_msls_ssl_long_vicreg_neg_partial_pair.sbatch
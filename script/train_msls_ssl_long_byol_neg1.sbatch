#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_ssl
#SBATCH --output=train_msls_ssl.out

eval "$(conda shell.bash hook)"
conda activate VG_SSL

srun python3 train_ssl.py --dataset_name=msls --mining random --datasets_folder=./datasets --epochs_num 1000 --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr $LR --optim adam --train_batch_size $BATCH --infer_batch_size 256 --use_best_positive --projection_size $PROJ --num_workers 8 --database_negatives_ratio 1.0 --n_layers $LAY

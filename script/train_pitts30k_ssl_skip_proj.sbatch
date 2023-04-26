#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=90GB  
#SBATCH --job-name=train_pitts30k_ssl
#SBATCH --output=train_pitts30k_ssl.out     
eval "$(conda shell.bash hook)"
conda activate VPR_SSL


srun python3 train_ssl.py --dataset_name=pitts30k --mining random --datasets_folder=/scratch/jx1190/VPR_SSL/datasets/ --epochs_num 200 --seed $SEED --ssl_method $SSL --method pair --backbone resnet18conv4 --lr 1e-4 --optim adam --train_batch_size 64 --infer_batch_size 64 --use_best_positive --database_negatives_ratio 0.1 --aug_epsilon g0.5
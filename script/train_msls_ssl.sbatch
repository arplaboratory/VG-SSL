#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_partial
#SBATCH --output=train_msls_partial.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 train.py --dataset_name=msls --datasets_folder=./datasets --seed $SEED --ssl_method $SSL --backbone resnet50conv4 --lr 0.2 --optim lars --cosine_scheduler
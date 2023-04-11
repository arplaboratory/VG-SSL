#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --job-name=train_pitts30k_partial
#SBATCH --output=train_pitts30k_partial.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 train.py --dataset_name=pitts30k --mining=partial --datasets_folder=./datasets --seed $SEED --pretrain $PRETRAIN --backbone resnet50conv4 --use_faiss_gpu
#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=24                # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --job-name=train_pitts30k_full
#SBATCH --output=train_pitts30k_full.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 train.py --dataset_name=pitts30k --mining=full --datasets_folder=/scratch/jx1190/VPR_SSL/datasets
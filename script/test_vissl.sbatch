#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --job-name=eval

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 run_distributed_engines.py config=quick_1gpu_resnet50_simclr config.DATA.TRAIN.DATA_SOURCES=[synthetic]
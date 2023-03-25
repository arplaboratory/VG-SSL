#!/bin/bash 
#SBATCH --nodes=2                        # requests 3 compute servers
#SBATCH --ntasks-per-node=4              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=train_pitts30k_partial
#SBATCH --output=train_pitts30k_partial.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

srun python3 train_ssl.py --dataset_name=msls --mining random --datasets_folder=./datasets --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr 0.2 --optim lars --cosine_scheduler --train_batch_size 32 --infer_batch_size 128
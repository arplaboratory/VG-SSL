#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=4              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=90GB  
#SBATCH --job-name=train_pitts30k_ssl
#SBATCH --output=train_pitts30k_ssl.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL


srun python3 train_ssl.py --dataset_name=pitts30k --mining random --datasets_folder=/scratch/jx1190/VPR_SSL/datasets/ --epochs_num 500 --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr 0.3 --optim lars --cosine_scheduler --train_batch_size 16 --infer_batch_size 128 --use_best_positive --num_workers 4 


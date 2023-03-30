#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=4              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=train_pitts30k_ssl
#SBATCH --output=train_pitts30k_ssl.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

opt_batch_size=`python3 find_batch_size.py --dataset_name=pitts30k --mining random --datasets_folder=./datasets --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr 0.2 --optim lars --cosine_scheduler --infer_batch_size 128`
rm .scale_batch_size_*
srun python3 train_ssl.py --dataset_name=pitts30k --mining random --datasets_folder=./datasets --epochs_num 1000 --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr 0.2 --optim lars --cosine_scheduler --train_batch_size $opt_batch_size --infer_batch_size 128

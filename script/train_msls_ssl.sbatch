#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=4              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=train_msls_ssl
#SBATCH --output=train_msls_ssl.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

opt_batch_size=`python3 find_batch_size.py --dataset_name=msls --mining random --datasets_folder=./datasets --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr $LR --optim adam --infer_batch_size 128 --matching $MATCH`
rm .scale_batch_size_*
srun python3 train_ssl.py --dataset_name=msls --mining random --datasets_folder=./datasets --epochs_num 1000 --seed $SEED --ssl_method $SSL --method pair --backbone resnet50conv4 --lr $LR --optim adam --train_batch_size $opt_batch_size --infer_batch_size 128 --matching $MATCH
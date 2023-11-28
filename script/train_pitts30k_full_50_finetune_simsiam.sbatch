#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_pitts30k_full
#SBATCH --output=train_pitts30k_full.out

singularity exec --nv \
                 --overlay /vast/jx1190/pitts30k.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train.py  --dataset_name pitts30k --backbone resnet50conv5 --aggregation gem --mining full --datasets_folder ./datasets --save_dir global_retrieval --lr 0.00001 --fc_output_dim 1024 --train_batch_size 16 --infer_batch_size 64 --num_workers 16 --epochs_num 500 --patience 50 --negs_num_per_query 2 --queries_per_epoch 5000 --cache_refresh_rate 1000 --resume logs/global_retrieval/msls-2023-10-10_05-34-16-cd63bcf3-b7f0-42e1-ae37-59cb30dc969b/best_model.pth"
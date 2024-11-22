#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=12               # uses 1 compute core per task
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_rerank
#SBATCH --output=train_msls_rerank.out

singularity exec --nv \
                 --overlay /vast/jx1190/pitts30k.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_reranker.py --fix 0 --dataset_name pitts30k --backbone deit --aggregation=gem --mining full --datasets_folder ./datasets  --save_dir pitts30k_finetune --lr 0.00001 --cosine_scheduler --fc_output_dim 256 --num_workers 12 --warmup 5 --optim adamw --epochs_num 50 --patience 10 --negs_num_per_query 1 --queries_per_epoch 5000 --cache_refresh_rate 1000 --train_batch_size 16 --infer_batch_size 256 --rerank_batch_size 4 --unfreeze --resume logs/finetune/msls-2023-11-21_08-44-37-0d51e02b-2756-40f0-995b-ae6832d9c211/best_model.pth"
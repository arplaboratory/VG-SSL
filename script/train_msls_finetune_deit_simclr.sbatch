#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=12               # uses 1 compute core per task
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_rerank
#SBATCH --output=train_msls_rerank.out

singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_reranker.py --fix 0 --dataset_name msls --backbone deit --aggregation=gem --mining partial --datasets_folder ./datasets  --save_dir finetune --lr 0.00001 --cosine_scheduler --fc_output_dim 256 --num_workers 12 --warmup 5 --optim adamw --epochs_num 50 --patience 10 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000 --train_batch_size 32 --infer_batch_size 256 --rerank_batch_size 20 --save_best 0 --unfreeze --resume logs/rerank/msls-2023-11-03_21-57-57-c941f4a0-6656-4d7e-b883-247d0c6465e4/best_model.pth"
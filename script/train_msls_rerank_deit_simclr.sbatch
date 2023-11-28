#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_rerank
#SBATCH --output=train_msls_rerank.out

singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_reranker.py  --dataset_name msls --backbone deit --aggregation=gem --mining global --neg_hardness 100 --datasets_folder ./datasets  --save_dir rerank --lr 0.0005 --cosine_scheduler --fc_output_dim 256 --num_workers 8 --warmup 5 --optim adamw --epochs_num 50 --patience 25 --negs_num_per_query 1 --queries_per_epoch 50000 --cache_refresh_rate 10000 --train_batch_size 128 --infer_batch_size 256 --rerank_batch_size 8 --save_best 0 --ssl_method simclr --resume logs/global_retrieval/msls-2023-11-15_02-02-22-f74a3d19-d8b1-4bf1-9f16-284d5c39b511/best_model.pth"
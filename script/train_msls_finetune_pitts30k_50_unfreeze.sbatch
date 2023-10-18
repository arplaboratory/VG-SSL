#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=12               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_rerank
#SBATCH --output=train_msls_rerank.out

singularity exec --nv \
                 --overlay /vast/jx1190/pitts30k.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_reranker.py --fix 0 --dataset_name pitts30k --backbone resnet50conv5 --aggregation=gem --mining full --datasets_folder ./datasets  --save_dir pitts30k_finetune --lr 0.00001 --cosine_scheduler --fc_output_dim 1024 --num_workers 12 --warmup 5 --optim adamw --epochs_num 50 --patience 10 --negs_num_per_query 1 --queries_per_epoch 5000 --cache_refresh_rate 1000 --train_batch_size 16 --infer_batch_size 256 --rerank_batch_size 4 --unfreeze --resume logs/finetune/msls-2023-10-06_16-50-05-f13305a1-fbf0-44d5-8aaa-e9f8ae61d292/best_model.pth"
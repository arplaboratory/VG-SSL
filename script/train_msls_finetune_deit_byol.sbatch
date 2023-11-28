#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=12               # uses 1 compute core per task
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=150GB
#SBATCH --job-name=train_msls_rerank
#SBATCH --output=train_msls_rerank.out

singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_reranker.py --fix 0 --dataset_name msls --backbone deit --aggregation=gem --mining partial --datasets_folder ./datasets  --save_dir finetune --lr 0.00001 --fc_output_dim 256 --num_workers 12 --optim adam --epochs_num 100 --patience 100 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000 --train_batch_size 64 --infer_batch_size 256 --rerank_batch_size 16 --save_best 0 --unfreeze --resume logs/rerank/msls-2023-11-18_22-13-30-43172d05-e42d-4946-94b6-5a337403f2b3/best_model.pth"
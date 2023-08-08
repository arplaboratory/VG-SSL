#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_partial
#SBATCH --output=train_msls_partial.out

singularity exec --nv \
                 --overlay ../mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train.py  --dataset_name msls --backbone deit --aggregation gem --mining partial --datasets_folder ./datasets --save_dir global_retrieval --lr 0.00001 --fc_output_dim 256 --train_batch_size 16 --infer_batch_size 1024 --num_workers 16 --epochs_num 100 --patience 10 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000"
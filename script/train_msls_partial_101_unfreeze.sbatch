#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=24               # uses 1 compute core per task
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_partial
#SBATCH --output=train_msls_partial.out

singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train.py  --dataset_name msls --backbone resnet101conv5 --aggregation gem --mining partial --datasets_folder ./datasets --save_dir global_retrieval --lr 0.00001 --fc_output_dim 1024 --train_batch_size 16 --infer_batch_size 64 --num_workers 24 --epochs_num 100 --patience 10 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000 --unfreeze"
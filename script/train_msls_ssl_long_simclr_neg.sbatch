#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16                # uses 1 compute core per task
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=train_msls_ssl
#SBATCH --output=train_msls_ssl.out

singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_ssl.py --dataset_name msls --backbone resnet50conv5 --aggregation gem --mining random --ssl_method $SSL --method pair --datasets_folder ./datasets --save_dir global_retrieval --lr $LR --fc_output_dim 1024 --train_batch_size $BATCH --infer_batch_size 256 --num_workers 16 --epochs_num 100 --patience 10 --negs_num_per_query 1 --queries_per_epoch 10000 --unfreeze --n_layers $LAY --projection_size $PROJ --neg_samples_num $NEG"

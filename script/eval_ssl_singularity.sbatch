#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --job-name=eval
#SBATCH --output=eval_ssl.out

eval "$(conda shell.bash hook)"
conda activate VG_SSL

singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 eval_reranker.py --resume='logs/global_retrieval/msls-2023-09-27_22-53-44-12156346-aaf5-4e52-a190-d2c78fdceedb/best_model.pth' --dataset_name msls --datasets_folder ./datasets --backbone resnet50conv5 --ssl_method simclr --fc_output_dim 1024 --aggregation gem --infer_batch_size 256"

# singularity exec --nv \
#                  --overlay /vast/jx1190/mapillary_sls.sqf:ro \
#                  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
#                  /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 eval_reranker.py --resume='logs/global_retrieval/msls-2023-09-27_22-53-44-12156346-aaf5-4e52-a190-d2c78fdceedb/best_model.pth' --dataset_name msls --datasets_folder ./datasets --backbone deit --ssl_method simclr --fc_output_dim 256 --aggregation gem --infer_batch_size 256 --rerank_batch_size 8"
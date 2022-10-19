#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=eval
#SBATCH --output=eval_%j_%a.out

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

if [ ${SLURM_ARRAY_TASK_ID} == 1 ]
then
    python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=st_lucia --datasets_folder ./datasets > eval_st_lucia.out
fi
if [ ${SLURM_ARRAY_TASK_ID} == 2 ]
then
    python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=pitts30k --datasets_folder ./datasets > eval_pitts30k.out
fi
if [ ${SLURM_ARRAY_TASK_ID} == 3 ]
then
    python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=msls     --datasets_folder ./datasets > eval_msls.out
fi
if [ ${SLURM_ARRAY_TASK_ID} == 4 ]
then
    python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=tokyo247 --datasets_folder ./datasets > eval_tokyo247.out
fi
if [ ${SLURM_ARRAY_TASK_ID} == 5 ]
then
    python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=san_francisco --datasets_folder ./datasets > eval_san_francisco
fi
if [ ${SLURM_ARRAY_TASK_ID} == 6 ]
then
    python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=pitts250k --datasets_folder ./datasets > eval_pitts250k
fi
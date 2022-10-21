#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=eynsham --datasets_folder ./datasets > eval_eynsham.out
python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=st_lucia --datasets_folder ./datasets > eval_st_lucia.out
python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=pitts30k --datasets_folder ./datasets > eval_pitts30k.out
python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=msls     --datasets_folder ./datasets > eval_msls.out
python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=tokyo247 --datasets_folder ./datasets > eval_tokyo247.out
python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=san_francisco --datasets_folder ./datasets > eval_san_francisco.out
python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=pitts250k --datasets_folder ./datasets > eval_pitts250k.out

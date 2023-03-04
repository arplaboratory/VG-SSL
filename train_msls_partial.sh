#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 train.py --dataset_name=msls --mining=partial --datasets_folder=./datasets
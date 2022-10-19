#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 train.py --dataset_name=pitts30k --mining=full --datasets_folder=datasets
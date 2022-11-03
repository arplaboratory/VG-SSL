#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 run_distributed_engines.py config=quick_1gpu_resnet50_simclr config.DATA.TRAIN.DATA_SOURCES=[synthetic]
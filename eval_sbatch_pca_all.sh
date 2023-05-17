#!/bin/bash
./eval_sbatch_pca.sh msls-2023-04-23_22-41-52-97371a43-c35e-4ccd-ab42-3483b184fc56 resnet50conv4 1024 # imagenet-0
./eval_sbatch_pca.sh msls-2023-04-23_22-41-50-acb2607f-f131-4591-937b-b09018abd88b resnet50conv4 1024 # imagenet-1
./eval_sbatch_pca.sh msls-2023-04-23_22-41-50-0b84dc59-71e6-4c28-b140-36b4b9f7c232 resnet50conv4 1024 # imagenet-2
./eval_sbatch_pca.sh msls-2023-04-23_22-41-52-97371a43-c35e-4ccd-ab42-3483b184fc56 resnet50conv4 4096 # imagenet-0
./eval_sbatch_pca.sh msls-2023-04-23_22-41-50-acb2607f-f131-4591-937b-b09018abd88b resnet50conv4 4096 # imagenet-1
./eval_sbatch_pca.sh msls-2023-04-23_22-41-50-0b84dc59-71e6-4c28-b140-36b4b9f7c232 resnet50conv4 4096 # imagenet-2


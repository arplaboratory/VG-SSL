#!/bin/bash

./eval_ssl_sbatch_proj.sh msls-2023-05-10_18-57-13-0a102599-1507-4d62-b76b-7f94a42b39f1 resnet50conv4 2048 simclr 1
./eval_ssl_sbatch_proj.sh msls-2023-05-10_22-58-30-fec8f4c5-fb9f-4100-9522-edacfeee51d3 resnet50conv4 2048 mocov2 1
./eval_ssl_sbatch_proj.sh msls-2023-05-13_08-56-04-357b4fec-293b-4fd3-9ca7-99938e7e6f0e resnet50conv4 2048 byol 2
./eval_ssl_sbatch_proj.sh msls-2023-05-13_08-56-04-6edd3058-491c-4c9f-a9af-7ff3aa6f3a56 resnet50conv4 2048 simsiam 2
./eval_ssl_sbatch_proj.sh msls-2023-05-11_22-57-19-856fdee1-796e-4290-b895-80a3a3c2941d resnet50conv4 2048 bt 2 #
./eval_ssl_sbatch_proj.sh msls-2023-05-11_22-57-19-baa1f032-d8b9-4485-abec-bc2a0d42b19d resnet50conv4 2048 vicreg 2
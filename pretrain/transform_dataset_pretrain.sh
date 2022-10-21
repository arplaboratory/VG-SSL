# #!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VPR_SSL

# GLDv2
python h5_transformer_pretrain.py --read_path ./datasets/GLDv2 --save_path  ./datasets/GLDv2/pretrain.h5 --compress --dataset_name GLDv2 &

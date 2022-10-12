#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VPR_SSL

# eynsham
python h5_reader.py --read_path ~/datasets_vg/datasets/eynsham/

# msls
python h5_reader.py --read_path ~/datasets_vg/datasets/msls/

# nordland
python h5_reader.py --read_path ~/datasets_vg/datasets/nordland

# # pitts30k
python h5_reader.py --read_path ~/datasets_vg/datasets/pitts30k

# # pitts250k
python h5_reader.py --read_path ~/datasets_vg/datasets/pitts250k

# #san_francisco
python h5_reader.py --read_path ~/datasets_vg/datasets/san_francisco

# #st_lucia
python h5_reader.py --read_path ~/datasets_vg/datasets/st_lucia/

# #tokyo247
python h5_reader.py --read_path ~/datasets_vg/datasets/tokyo247


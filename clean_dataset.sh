#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VPR_SSL

# # eynsham
rm ./datasets/eynsham/*.h5

# # msls
rm ./datasets/msls/*.h5

# # nordland
rm ./datasets/nordland/*.h5

# # # pitts30k
rm ./datasets/pitts30k/*.h5

# # # pitts250k
rm ./datasets/pitts250k/*.h5

# # #san_francisco
rm ./datasets/san_francisco/*.h5

# # #st_lucia
rm ./datasets/st_lucia/*.h5

# # #tokyo247
rm ./datasets/tokyo247/*.h5


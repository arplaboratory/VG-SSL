import h5py
import numpy as np
import os
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, help='The path of the folder that contains jpg or png images')
parser.add_argument('--compress', action='store_true')
args = parser.parse_args()

for h5_file_name in os.listdir(args.read_path):
    extension = h5_file_name.split('.')[-1]
    if extension == 'h5':
        print('====================================')
        try:
            hf = h5py.File(os.path.join(args.read_path, h5_file_name), 'r') # open a hdf5 file
            print(os.path.join(args.read_path, h5_file_name))
            print(hf['image_name'][0])
            print(hf['image_size'][0])
            print(hf['image_data'].shape)
            print(hf['image_data'][0,0:10,0:10,0])
            hf.close()  # close the hdf5 file
        except Exception as e:
            print(f'Reading error for {os.path.join(args.read_path, h5_file_name)}')
            print(f'{e}')
        print('====================================')
import h5py
import numpy as np
import os
from PIL import Image
import argparse
from tqdm import tqdm

ALLOWED_EXT = ['jpeg', 'jpg', 'png']
parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, help='The path of the folder that contains jpg or png images')
parser.add_argument('--save_path', type=str, help='The path of the folder that contains jpg or png images')
parser.add_argument('--resize_height', type=int, default=480)
parser.add_argument('--resize_width', type=int, default=640)
parser.add_argument('--compress', action='store_true')
args = parser.parse_args()

# To create h5 file, we must resize the images
if os.path.isfile(args.save_path):
    os.remove(args.save_path)
hf = h5py.File(args.save_path, 'a') # open a hdf5 file
start = False
for img_name in tqdm(os.listdir(args.read_path)):
    extension = img_name.split('.')[-1]
    if extension in ALLOWED_EXT:
        img = Image.open(os.path.join(args.read_path, img_name)).convert("RGB")
        new_img = img.resize((args.resize_width, args.resize_height))
        img_np = np.array(new_img)
        img_np = np.expand_dims(img_np, axis = 0)
        size_np = np.expand_dims(np.array([img.height, img.width]), axis=0)
        if not start:
            if args.compress:
                hf.create_dataset('image_data', data=img_np, chunks=True, maxshape=(None, 640, 640, 3), compression='lzf')  # write the data to hdf5 file
                hf.create_dataset('image_size', data=size_np, chunks=True, maxshape=(None, 2), compression='lzf')
            else:
                hf.create_dataset('image_data', data=img_np, chunks=True, maxshape=(None, 640, 640, 3))  # write the data to hdf5 file
                hf.create_dataset('image_size', data=size_np, chunks=True, maxshape=(None, 2))
            start = True
        else:
            hf['image_data'].resize(hf['image_data'].shape[0] + img_np.shape[0], axis=0)
            hf['image_data'][-img_np.shape[0]:] = img_np
            hf['image_size'].resize(hf['image_size'].shape[0] + size_np.shape[0], axis=0)
            hf['image_size'][-size_np.shape[0]:] = size_np
    else:
        continue
hf.close()  # close the hdf5 file
print('hdf5 file size: %d bytes'%os.path.getsize(args.save_path))
import h5py
import numpy as np
import os
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd


def create_h5_file(args, resize_option=False):
    # Always use resize for GLDv2
    if os.path.isfile(args.save_path):
        os.remove(args.save_path)
    with h5py.File(args.save_path, "a") as hf:
        start = False
        img_names = []
        labels = []
        if args.dataset_name == "GLDv2":
            df = pd.read_csv(os.path.join(args.read_path, "train.csv"))
            img_names = df["id"].values
            labels = df["landmark_id"].values
            labels = np.array(labels)
        for img_name in tqdm(img_names):
            img_path = os.path.join(
                args.read_path, img_name[0], img_name[1], img_name[2], img_name + ".jpg"
            )
            img = Image.open(img_path).convert("RGB")
            size_np = np.expand_dims(np.array([img.height, img.width]), axis=0)
            if resize_option is True:
                new_img = img.resize((args.resize_width, args.resize_height))
            else:
                new_img = img
            img_np = np.array(new_img)
            img_np = np.expand_dims(img_np, axis=0)
            if not start:
                if args.compress:
                    hf.create_dataset(
                        "image_data",
                        data=img_np,
                        chunks=True,
                        maxshape=(None, None, None, 3),
                        compression="lzf",
                    )  # write the data to hdf5 file
                    hf.create_dataset(
                        "image_size",
                        data=size_np,
                        chunks=True,
                        maxshape=(None, 2),
                        compression="lzf",
                    )
                else:
                    hf.create_dataset(
                        "image_data",
                        data=img_np,
                        chunks=True,
                        maxshape=(None, None, None, 3),
                    )  # write the data to hdf5 file
                    hf.create_dataset(
                        "image_size", data=size_np, chunks=True, maxshape=(None, 2)
                    )
                start = True
            else:
                hf["image_data"].resize(
                    hf["image_data"].shape[0] + img_np.shape[0], axis=0
                )
                hf["image_data"][-img_np.shape[0] :] = img_np
                hf["image_size"].resize(
                    hf["image_size"].shape[0] + size_np.shape[0], axis=0
                )
                hf["image_size"][-size_np.shape[0] :] = size_np
        t = h5py.string_dtype(encoding="utf-8")
        if args.compress:
            hf.create_dataset("label", data=labels, compression="lzf")
            hf.create_dataset("image_name", data=img_names, dtype=t, compression="lzf")
        else:
            hf.create_dataset("label", data=labels)
            hf.create_dataset("image_name", data=img_names, dtype=t)
        print("hdf5 file size: %d bytes" % os.path.getsize(args.save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--read_path",
        type=str,
        help="The path of the folder that contains jpg or png images",
    )
    parser.add_argument(
        "--save_path", type=str, help="The path of the folder that stores h5 files"
    )
    parser.add_argument("--resize_height", type=int, default=224)
    parser.add_argument("--resize_width", type=int, default=224)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--dataset_name", type=str, choices=["GLDv2"])
    args = parser.parse_args()

    # First round if not variable image size, use original size. Else, use default size.
    try:
        if args.dataset_name == "GLDv2":
            create_h5_file(args, resize_option=True)
    except Exception as e:
        print(f"{args.save_path} extraction error")
        print(f"{e}")
        print(
            f"Try resizing all images to ({args.resize_width}, {args.resize_height}) (WxH)"
        )
        create_h5_file(args, resize_option=True)

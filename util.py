import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA
from tqdm import tqdm

import datasets_ws


def get_flops(model, input_shape=(480, 640)):
    # """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    # assert (
    #     len(input_shape) == 2
    # ), f"input_shape should have len==2, but it's {input_shape}"
    # module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    # output = torchscan.utils.format_info(module_info)
    # return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]
    pass


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith("module"):
        state_dict = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict.items()}
        )
    model.load_state_dict(state_dict)
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    if "epoch_num" in checkpoint.keys(): # train.py
        start_epoch_num = checkpoint["epoch_num"]
        if args.backbone.startswith('deit') and 'module.backbone.cls_token' not in checkpoint["model_state_dict"]:
            for key in list(checkpoint["model_state_dict"].keys()):
                checkpoint["model_state_dict"][key.replace('module','module.backbone')] = checkpoint["model_state_dict"][key]
                del(checkpoint["model_state_dict"][key])
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_r5 = checkpoint["best_r5"]
        recalls = checkpoint["recalls"]
        not_improved_num = checkpoint["not_improved_num"]
        logging.debug(
            f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
            f"current_best_R@5 = {best_r5:.1f}, "
            f"recalls = {recalls}"
        )
    else: # train_ssl.py
        start_epoch_num = checkpoint["epoch"]
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith('ssl_model'):
                del checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace('backbone','module.backbone')] = checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace('aggregation','module.aggregation')] = checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            if not key.startswith('module'):
                del checkpoint["state_dict"][key]
        if not args.backbone.startswith('deit'):
            try:
                checkpoint["state_dict"]["module.aggregation.0.1.p"] =checkpoint["state_dict"]["module.aggregation.0.0.1.p"]
                checkpoint["state_dict"]["module.aggregation.1.weight"] =checkpoint["state_dict"]["module.aggregation.0.1.weight"]
                checkpoint["state_dict"]["module.aggregation.1.bias"] =checkpoint["state_dict"]["module.aggregation.0.1.bias"]
            except Exception:
                logging.debug("No projection layer found!")
                
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_states"])
        best_r5 = checkpoint['callbacks']["ModelCheckpoint{'monitor': 'val_recall5', 'mode': 'max', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_score'].item()
        not_improved_num = 0
        logging.debug(
            f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
            f"current_best_R@5 = {best_r5:.1f}"
        )
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

def resume_train_ssl(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_states"])
    best_r5 = checkpoint['callbacks']["ModelCheckpoint{'monitor': 'val_recall5', 'mode': 'max', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_score'].item()
    not_improved_num = 0
    logging.debug(
        f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
        f"current_best_R@5 = {best_r5:.1f}"
    )
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

def resume_train_pitts30k(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch"]
    for key in list(checkpoint["state_dict"].keys()):
        if key.startswith('ssl_model'):
            del checkpoint["state_dict"][key]
    if not args.backbone.startswith('deit'):
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace('backbone','module.backbone')] = checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace('aggregation','module.aggregation')] = checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            if not key.startswith('module'):
                del checkpoint["state_dict"][key]
        try:
            checkpoint["state_dict"]["module.aggregation.0.1.p"] =checkpoint["state_dict"]["module.aggregation.0.0.1.p"]
            checkpoint["state_dict"]["module.aggregation.1.weight"] =checkpoint["state_dict"]["module.aggregation.0.1.weight"]
            checkpoint["state_dict"]["module.aggregation.1.bias"] =checkpoint["state_dict"]["module.aggregation.0.1.bias"]
        except Exception:
            logging.debug("No projection layer found!")
    else:
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace('backbone.','')] = checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith('backbone'):
                del checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith('aggregation'):
                del checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"]['module.' + key] = checkpoint["state_dict"][key]
        for key in list(checkpoint["state_dict"].keys()):
            if not key.startswith('module'):
                del checkpoint["state_dict"][key]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    best_r5 = checkpoint['callbacks']["ModelCheckpoint{'monitor': 'val_recall5', 'mode': 'max', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_score'].item()
    not_improved_num = 0
    logging.debug(
        f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
        f"current_best_R@5 = {best_r5:.1f}"
    )
    return model, optimizer, 0, 0, 0

def compute_pca(args, model, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(
        args, args.datasets_folder, args.pca_dataset_folder)
    dl = torch.utils.data.DataLoader(
        pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    logging.info("Computing PCA")
    with torch.no_grad():
        for i, images in tqdm(enumerate(dl), ncols=100):
            if i * args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[
                i * args.infer_batch_size: (i * args.infer_batch_size) + len(features)
            ] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca
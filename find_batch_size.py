from model import network
import datasets_ws
import commons
import parser
import test
import util
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join, isdir
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.tuner import Tuner
from uuid import uuid4
import os
import copy

def trial_run_for_tuning_batch_size(args):
    args.num_nodes = 1
    args.num_devices = 1

    # Creation of Datasets
    logging.debug(
        f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

    train_ds = None
    if args.method == 'pair':
        train_ds = datasets_ws.PairsDataset(
            args, args.datasets_folder, args.dataset_name, "train"
        )
    else:
        raise NotImplementedError('Unknown method is used')

    logging.info(f"Trial Train query set: {train_ds}")

    val_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.dataset_name, "val")
    logging.info(f"Trial Val set: {val_ds}")

    test_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Trial Test set: {test_ds}")
    # Initialize model
    model = network.SSLGeoLocalizationNet(copy.deepcopy(args), [train_ds, val_ds, test_ds])

    trainer = pl.Trainer(
        accelerator = "gpu",
        num_nodes = args.num_nodes,
        devices = args.num_devices,
        max_epochs = 1,
        sync_batchnorm = True,
        reload_dataloaders_every_n_epochs = 1,
        check_val_every_n_epoch = 1,
        num_sanity_val_steps = 0,
        enable_checkpointing=False
    )
    
    # Need manual setup
    model.setup(stage = "validate")
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="power")

    logging.info("Tuning batch size with 1 GPU finished")
    batch_size = copy.deepcopy(model.batch_size)
    del model
    del trainer
    torch.cuda.empty_cache()
    return batch_size

def main():
    # Initial setup: parser, logging...
    args = parser.parse_arguments()
    print(trial_run_for_tuning_batch_size(args))

if __name__ == "__main__":
    main()
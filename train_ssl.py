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
from uuid import uuid4
import os

def main():
    # Initial setup: parser, logging...
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join(
        "logs",
        args.save_dir,
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
    )
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    wandb_logger = WandbLogger(project="vg-ssl", entity="vg-ssl")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(
        f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs"
    )

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

    logging.info(f"Train query set: {train_ds}")

    val_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.dataset_name, "val")
    logging.info(f"Val set: {val_ds}")

    test_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    try:
        args.num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        args.num_devices = int(os.environ["SLURM_NTASKS_PER_NODE"])
    except:
        if args.visualize_input:
            args.num_nodes = 1
            args.num_devices = 1
        else:
            args.num_nodes = 1
            args.num_devices = torch.cuda.device_count()

    # Initialize model
    model = network.SSLGeoLocalizationNet(args, [train_ds, val_ds, test_ds])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_recall5",
        dirpath=args.save_dir,
        filename="best_model",
        save_last=True,
        mode="max",
        verbose=True)
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last_model"
    checkpoint_callback.FILE_EXTENSION = ".pth"
    bar = RichProgressBar()
    lrmoniter = LearningRateMonitor(logging_interval = "step")

    trainer = pl.Trainer(
        accelerator = "gpu",
        num_nodes = args.num_nodes,
        devices = args.num_devices,
        max_epochs = args.epochs_num,
        sync_batchnorm = True,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, bar, lrmoniter],
        check_val_every_n_epoch = 10,
        num_sanity_val_steps = 0
    )
    logging.debug(model)
    if trainer.is_global_zero:
        wandb_logger.experiment.config.update(vars(args))
    trainer.validate(model)
    trainer.fit(model)

if __name__ == "__main__":
    main()

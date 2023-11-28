#  Copyright [2023] [Bytedance]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os,sys
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"  #
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from uuid import uuid4
torch.backends.cudnn.benchmark=True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
from model import network
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint
from model.vicreg.utils import adjust_learning_rate
import wandb

def run_train():
    #### Initial setup: parser, logging...
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
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    #### Creation of Datasets
    logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

    try:
        val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
        logging.info(f"Val set: {val_ds}")
    except Exception:
        logging.info("Val set not found!")

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    #### Initialize model
    model = network.GeoLocalizationNetRerank(args)
    model = model.to(args.device)
    if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
        if not args.resume:
            triplets_ds.is_inference = True
            model.aggregation.initialize_netvlad_layer(args, triplets_ds, model.backbone)
        args.features_dim *= args.netvlad_clusters

    model = torch.nn.DataParallel(model)

    #### Resume model, optimizer, and other training parameters
    if args.resume:
        if args.aggregation != 'crn':
            # model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
            model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
        else:
            # CRN uses pretrained NetVLAD, then requires loading with strict=False and
            # does not load the optimizer from the checkpoint file.
            model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
        logging.info(f"Resuming from epoch {start_epoch_num} with best recall {best_r5:.1f}")
        best_r5 = start_epoch_num = not_improved_num = 0
    else:
        best_r5 = start_epoch_num = not_improved_num = 0

    # if args.backbone.startswith('vit'):
    #     logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}")
    # else:
    #     logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}")

    if torch.cuda.device_count() >= 2:
        # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        model = convert_model(model)
        model = model.cuda()

    if args.test:
        ds = test_ds
    else:
        ds = val_ds
        
    recalls, recalls_str = test.test_rerank(args, ds, model, test_method=args.test_method,rerank_bs=args.rerank_batch_size, num_local=args.num_local, rerank_dim=(args.local_dim+3), reg_top=args.reg_top)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")

if __name__=='__main__':
    run_train()

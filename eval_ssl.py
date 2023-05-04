import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
from model import network

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
model_name = args.resume.split('/')[-2]
args.save_dir = join(
    "test",
    args.save_dir,
    model_name,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.GeoLocalizationNet(args)
logging.debug(model)
model = model.to(args.device)

if args.aggregation in ["netvlad", "crn"]:
    if args.projection_size != -1 and args.compress_fc:
        args.features_dim = args.projection_size
    else:
        args.features_dim *= args.netvlad_clusters

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model_ssl(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(
        args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

from model.functional import sare_ind, sare_joint
from model.sync_batchnorm import convert_model
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
from uuid import uuid4

torch.backends.cudnn.benchmark = True  # Provides a speedup


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

# Initialize model
model = network.SSLGeoLocalizationNet(args)
model.setup(args)

# if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
#     if not args.resume:
#         train_ds.is_inference = True
#         model.aggregation.initialize_netvlad_layer(
#             args, train_ds, model.backbone)
#     args.features_dim *= args.netvlad_clusters


# Setup Optimizer and Loss
if args.aggregation == "crn":
    crn_params = list(model.module.aggregation.crn.parameters())
    net_params = list(model.module.backbone.parameters()) + list(
        [
            m[1]
            for m in model.module.aggregation.named_parameters()
            if not m[0].startswith("crn")
        ]
    )
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": crn_params, "lr": args.lr_crn_layer},
                {"params": net_params, "lr": args.lr_crn_net},
            ]
        )
        logging.info("You're using CRN with Adam, it is advised to use SGD")
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": crn_params,
                    "lr": args.lr_crn_layer,
                    "momentum": 0.9,
                    "weight_decay": 0.001,
                },
                {
                    "params": net_params,
                    "lr": args.lr_crn_net,
                    "momentum": 0.9,
                    "weight_decay": 0.001,
                },
            ]
        )
else:
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001
        )

if args.method == "pair":
    # TODO: Add pair loss criterion here. If the model return loss, then skip
    if model.module.return_loss == False:
        raise NotImplementedError("Criterion not found for pairs!")
    else:
        criterion_pairs = None
else:
    raise NotImplementedError()

# Resume model, optimizer, and other training parameters
if args.resume:
    if args.aggregation != "crn":
        (
            model,
            optimizer,
            best_r5,
            start_epoch_num,
            not_improved_num,
        ) = util.resume_train(args, model, optimizer)
    else:
        # CRN uses pretrained NetVLAD, then requires loading with strict=False and
        # does not load the optimizer from the checkpoint file.
        model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(
            args, model, strict=False
        )
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}"
    )
else:
    best_r5 = start_epoch_num = not_improved_num = 0

if args.backbone.startswith("vit"):
    logging.info(f"Output dimension of the model is {args.features_dim}")
else:
    logging.info(
        f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}"
    )

# Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        if args.method == 'pairs':
            # Compute pairs to use in the pair loss
            train_ds.is_inference = True
            train_ds.compute_pairs(args, model)
            train_ds.is_inference = False
            pairs_dl = DataLoader(
                dataset=train_ds,
                num_workers=args.num_workers,
                batch_size=args.train_batch_size,
                collate_fn=datasets_ws.collate_fn,
                pin_memory=(args.device == "cuda"),
                drop_last=True,
            )
        else:
            raise NotImplementedError()

        torch.cuda.empty_cache()
        model = model.train()

        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # pairs_local_indexes shape
        if args.method == "pairs":
            for images, pairs_local_indexes, _ in tqdm(pairs_dl, ncols=100):

                # Flip all pairs or none
                if args.horizontal_flip:
                    images = transforms.RandomHorizontalFlip()(images)

                # Compute features of all images (images contains queries, positives and negatives)
                if criterion_pairs is None:
                    loss = model(images.to(args.device))
                    loss_pairs = loss
                else:
                    features = model(images.to(args.device))
                    loss_pairs = 0
                    del features

                loss_pairs /= args.train_batch_size

                optimizer.zero_grad()
                loss_pairs.backward()
                optimizer.step()

                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss_pairs.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del loss_pairs

        logging.debug(
            f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): "
            + f"current batch pair  loss = {batch_loss:.4f}, "
            + f"average epoch pair loss = {epoch_losses.mean():.4f}"
        )

    logging.info(
        f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch pair loss = {epoch_losses.mean():.4f}"
    )

    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    is_best = recalls[1] > best_r5

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(
        args,
        {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r5": best_r5,
            "not_improved_num": not_improved_num,
        },
        is_best,
        filename="last_model.pth",
    )

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
        )
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
        )
        if not_improved_num >= args.patience:
            logging.info(
                f"Performance did not improve for {not_improved_num} epochs. Stop training."
            )
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(
    f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}"
)

# Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
    "model_state_dict"
]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test.test(
    args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")
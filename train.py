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
import wandb
from uuid import uuid4
from model.Deit import deit_small_distilled_patch16_224, deit_base_distilled_patch16_384
from model.vicreg.utils import adjust_learning_rate

torch.backends.cudnn.benchmark = True  # Provides a speedup


# Initial setup: parser, logging...
args = parser.parse_arguments()
if args.ssl_method != "none":
    raise ValueError("Training with triplets should set ssl_method to none")
start_time = datetime.now()
args.save_dir = join(
    "logs",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
wandb.init(project="vg-ssl", entity="vg-ssl", config=vars(args))
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(
    f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs"
)

# Creation of Datasets
logging.debug(
    f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

train_ds = None
if args.method == 'triplet':
    train_ds = datasets_ws.TripletsDataset(
        args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query
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
if args.backbone == "deitBase":
    args.features_dim = args.fc_output_dim
    model = deit_base_distilled_patch16_384(img_size=args.resize, num_classes=args.features_dim)
elif args.backbone == "deit":
    args.features_dim = args.fc_output_dim
    model = deit_small_distilled_patch16_224(img_size=args.resize, num_classes=args.features_dim)
else:
    model = network.GeoLocalizationNet(args)
model = model.to(args.device)
if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
    if not args.resume:
        train_ds.is_inference = True
        model.aggregation.initialize_netvlad_layer(
            args, train_ds, model.backbone)
    args.features_dim *= args.netvlad_clusters

model = torch.nn.DataParallel(model)

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
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.03,
                                        amsgrad=False)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    else:
        raise NotImplementedError()

if args.method == "triplet":
    if args.criterion == "triplet":
        criterion_triplet = nn.TripletMarginLoss(
            margin=args.margin, p=2, reduction="sum")
    elif args.criterion == "sare_ind":
        criterion_triplet = sare_ind
    elif args.criterion == "sare_joint":
        criterion_triplet = sare_joint
    else:
        raise NotImplementedError("Criterion not found for triplets!")
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
        ) = util.resume_train_pitts30k(args, model, optimizer, strict=True)
    else:
        # CRN uses pretrained NetVLAD, then requires loading with strict=False and
        # does not load the optimizer from the checkpoint file.
        model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train_pitts30k(
            args, model, strict=False
        )
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}"
    )
else:
    best_r5 = start_epoch_num = not_improved_num = 0

# if args.backbone.startswith("vit"):
#     logging.info(f"Output dimension of the model is {args.features_dim}")
# else:
#     logging.info(
#         f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}"
#     )


if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

recalls, recalls_str = test.test(args, val_ds, model)
logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

# Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    if args.optim == 'adamw':
        adjust_learning_rate(optimizer, epoch_num, args)

    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        if args.method == "triplet":
            # Compute triplets to use in the triplet loss
            train_ds.is_inference = True
            train_ds.compute_triplets(args, model)
            train_ds.is_inference = False
            triplets_dl = DataLoader(
                dataset=train_ds,
                num_workers=args.num_workers,
                batch_size=args.train_batch_size,
                collate_fn=datasets_ws.collate_fn,
                pin_memory=(args.device == "cuda"),
                drop_last=True,
            )
        else:
            raise NotImplementedError()

        model.train()

        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        if args.method == "triplet":
            for images, triplets_local_indexes, _, _ in tqdm(triplets_dl, ncols=100):

                # Flip all triplets or none
                if args.horizontal_flip:
                    images = transforms.RandomHorizontalFlip()(images)

                # Compute features of all images (images contains queries, positives and negatives)
                features = model(images.to(args.device))
                loss_triplet = 0

                if args.criterion == "triplet":
                    triplets_local_indexes = torch.transpose(
                        triplets_local_indexes.view(
                            args.train_batch_size, args.negs_num_per_query, 3
                        ),
                        1,
                        0,
                    )
                    for triplets in triplets_local_indexes:
                        queries_indexes, positives_indexes, negatives_indexes = triplets.T
                        loss_triplet += criterion_triplet(
                            features[queries_indexes],
                            features[positives_indexes],
                            features[negatives_indexes],
                        )
                elif args.criterion == "sare_joint":
                    # sare_joint needs to receive all the negatives at once
                    triplet_index_batch = triplets_local_indexes.view(
                        args.train_batch_size, 10, 3
                    )
                    for batch_triplet_index in triplet_index_batch:
                        q = features[batch_triplet_index[0, 0]].unsqueeze(
                            0
                        )  # obtain query as tensor of shape 1xn_features
                        p = features[batch_triplet_index[0, 1]].unsqueeze(
                            0
                        )  # obtain positive as tensor of shape 1xn_features
                        n = features[
                            batch_triplet_index[:, 2]
                        ]  # obtain negatives as tensor of shape 10xn_features
                        loss_triplet += criterion_triplet(q, p, n)
                elif args.criterion == "sare_ind":
                    for triplet in triplets_local_indexes:
                        # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
                        q_i, p_i, n_i = triplet
                        loss_triplet += criterion_triplet(
                            features[q_i: q_i + 1],
                            features[p_i: p_i + 1],
                            features[n_i: n_i + 1],
                        )

                del features
                loss_triplet /= args.train_batch_size * args.negs_num_per_query

                optimizer.zero_grad()
                loss_triplet.backward()
                optimizer.step()

                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss_triplet.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del loss_triplet

        logging.debug(
            f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): "
            + f"current batch triplet loss = {batch_loss:.4f}, "
            + f"average epoch triplet loss = {epoch_losses.mean():.4f}"
        )

    logging.info(
        f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch triplet loss = {epoch_losses.mean():.4f}"
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

    wandb.log({
        "epoch_num": epoch_num,
        "recall1": recalls[0],
        "recall5": recalls[1],
        "best_r5": recalls[1] if is_best else best_r5,
        "sum_loss": epoch_losses.mean(),
    },)

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
wandb.log({
    "final_recall1": recalls[0],
    "final_recall5": recalls[1],
},)

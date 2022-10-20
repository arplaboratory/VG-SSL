import logging
from os.path import join
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

import util
import parser
import commons
from model import network, metrics
from dataset import GoogleLandmarkDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup


# ----  Initial setup: parser, logging... ----
args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = join("runs", args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))

commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
device = args.device

# ---- Dataloaders ----
# Read the csv files containing the names and labels of the training images of GLDv2
df = pd.read_csv(args.gldv2_csv)

train_dataset = GoogleLandmarkDataset(
    data_path=args.data_path,
    image_list=df['id'].values,
    class_ids=df['landmark_id'].values,
    resize_shape=(int(args.resize_shape[0]),
                  int(args.resize_shape[1])),
    )

train_dl = DataLoader(dataset=train_dataset,
                      batch_size=args.train_batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=True,
                      drop_last=True)

num_classes = df['landmark_id'].max() + 1   # Classes go from 0 to max()
args.features_dim = 512

# ---- Model ----
# instantiate the model, using ImageNet pretrained nets from torchvision
model = network.LandmarkNet(args=args, num_classes=num_classes)
model = model.to(device)
model = torch.nn.DataParallel(model)

logging.info(f"Number of classes for dataset {args.dataset_name} is {num_classes}")
logging.info(f"Training Samples   : {len(train_dl.dataset)}")

# ---- Setup Optimizer and Loss ----
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

criterion = nn.CrossEntropyLoss()

# ---- Resume the model, optimizer, training_parameters ----
if args.resume:
    model, optimizer, best_acc, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best accuracy {best_acc:.4f}")
else:
    not_improved = start_epoch_num = not_improved_num = best_acc = 0

# ---- Training Loop ----
for epoch_num in range(start_epoch_num, args.epochs_num):
    #  ---- Training Step ----
    logging.info(f"Start training epoch: {epoch_num:02d}/{args.epochs_num}")
    epoch_start_time = datetime.now()
    # ---- Loss & Metric ----
    train_loss = metrics.AverageMeter()
    train_acc  = metrics.AverageMeter()

    model.train()

    for i, (images, labels) in tqdm(enumerate(train_dl),
                                    total=len(train_dl),
                                    miniters=None, ncols=100):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.shape[0]

        outputs = model(images, labels)  # with loss_module != 'arcface', labels are not used in the model
        loss = criterion(outputs, labels)
        acc = metrics.accuracy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batch_size)
        train_acc.update(acc, batch_size)

    logging.info(f"Training: Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch loss = {train_loss.avg:.4f}, "
                 f"average epoch accuracy = {train_acc.avg:.4f}")

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "accuracy": train_acc.avg,
                                "not_improved_num": not_improved_num}, False, filename="last_model.pth")

logging.info(f"Trained for {epoch_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

import logging
from os.path import join
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

# from model.metrics import gap

import util
import parser
import commons
import dataset
from model import network, metrics

torch.backends.cudnn.benchmark = True  # Provides a speedup


# ----  Initial setup: parser, logging... ----
args = parser.parse_arguments('places')
start_time = datetime.now()
args.output_folder = join("runs", args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))

commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
device = args.device

# ---- Dataloaders ----
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.Places365(root=args.data_path,
                                   split="train-standard",
                                   transform=transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize
                                   ]))
valid_dataset = datasets.Places365(root=args.data_path,
                                   split="val",
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize]))

train_dl = DataLoader(dataset=train_dataset,
                      batch_size=args.train_batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=True)
valid_dl = DataLoader(dataset=valid_dataset,
                      batch_size=args.eval_batch_size,
                      shuffle=False,
                      num_workers=args.num_workers,
                      pin_memory=True)

num_classes = 365  # Places365 contains 365 classes
args.features_dim = 365  # For places the ArcFaceLoss module is not needed

# ---- Model ----
# instantiate the model, using ImageNet pretrained nets from torchvision
model = network.LandmarkNet(args=args,
                            num_classes=num_classes,  # dynamically obtained from the dataset
                            )
model = model.to(device)
model = torch.nn.DataParallel(model)

logging.info(f"Number of classes for dataset {args.dataset_name} is {num_classes}")
logging.info(f"Training Samples   : {len(train_dl.dataset)}")
logging.info(f"Validation Samples : {len(valid_dl.dataset)}")

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

        outputs = model(images)
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

    # ---- Validation Step ----
    logging.info(f"Start validation epoch: {epoch_num:02d}/{args.epochs_num}")
    epoch_start_time = datetime.now()
    # ---- Loss & Metric ----
    valid_loss = metrics.AverageMeter()
    valid_acc = metrics.AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(valid_dl),
                                        total=len(valid_dl),
                                        miniters=None, ncols=100):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]

            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            acc = metrics.accuracy(outputs, labels)

            valid_loss.update(loss.item(), batch_size)
            valid_acc.update(acc, batch_size)

        logging.info(f"Validation : Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch loss = {valid_loss.avg:.4f}, "
                     f"average epoch accuracy = {valid_acc.avg:.4f}")

    is_best = valid_acc.avg > best_acc
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "accuracy": valid_acc.avg,
                                "not_improved_num": not_improved_num}, is_best, filename="last_model.pth")

    # If accuracy did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best accuracy = {best_acc:.4f}, current accuracy = {valid_acc.avg:.4f}")
        best_acc = valid_acc.avg
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best accuracy = {best_acc:.4f}, current accuracy = {valid_acc.avg:.4f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best acc: {best_acc:.4f}")
logging.info(f"Trained for {epoch_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
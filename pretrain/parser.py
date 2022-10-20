import socket
import argparse
import multiprocessing

def parse_arguments(dataset_name):
    parser = argparse.ArgumentParser(description="Pretrain for Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- Generic Arguments ----
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")

    # ---- Dataset and DataLoader Arguments ----
    if dataset_name == 'gldv2':
        parser.add_argument("--gldv2_csv", type=str, default=None,
                            help="csv  file containing the metadata of GLDv2")
    elif dataset_name == 'places':
        parser.add_argument("--eval_batch_size", type=int, default=64,
                            help="Number of images in the eval batch size.")

    parser.add_argument("--data_path", type=str, default="",
                        help="Directory of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Number of images in the train batch size.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used to generate the splits in train/val set.")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(),
                        help="num_workers for all dataloaders")
    parser.add_argument('--resize_shape', type=int, default=[224, 224], nargs=2,
                        help="Resizing shape for images (HxW).")

    # ---- loss_module Arguments ----
    if dataset_name == 'gldv2':
        parser.add_argument("--loss_module", type=str, default="", help="loss_module",
                            choices=["arcface", ""])
        parser.add_argument('--arcface_s', type=float, default=30, help="s parameter of arcface loss")
        parser.add_argument('--arcface_margin', type=float, default=0.3, help="margin of arcface loss")
        parser.add_argument('--arcface_ls_eps', type=float, default=0.0, help="ls_eps of arcface loss. (label_smoothing)")

    # ---- Training and Optimizer Arguments ----
    parser.add_argument("--epochs_num", type=int, default=1000,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])

    # ----  Model Arguments ----
    parser.add_argument("--arch", type=str, default="r18",
                        choices=["vgg16",
                                 "r18",
                                 "r50",
                                 "r101"],
                        help="_")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")

    args = parser.parse_args()

    args.dataset_name = dataset_name
    if args.dataset_name == 'gldv2' and args.gldv2_csv is None:
        raise ValueError("With datasets GLDv2 the csv file with images id and landmark_id must be passed using "
                         "parameter --gldv2_csv")

    return args

# https://raw.githubusercontent.com/facebookresearch/vicreg/main/main_vicreg.py
from torch import nn, optim
import torch
import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cosine_scheduler:  # cosine lr schedule
        if epoch < args.warmup:
            alpha = epoch / args.warmup
            warmup_factor = 0.1 * (1.0 - alpha) + alpha
            lr *= warmup_factor
        else:
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs_num))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('current lr:', lr)
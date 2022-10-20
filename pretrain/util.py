import re
import torch
import shutil
import logging


def save_checkpoint(args, state, is_best, filename):
    model_path = f"{args.output_folder}/{filename}"
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, f"{args.output_folder}/best_model.pth")


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    accuracy = checkpoint['accuracy']
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, " \
                  f"current_accuracy = {accuracy:.4f}")
    shutil.copy(args.resume, args.output_folder)
    return model, optimizer, accuracy, start_epoch_num, not_improved_num


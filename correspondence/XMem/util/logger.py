"""
Dumps things to wandb and console
"""

import os
import warnings

import torchvision.transforms as transforms
import wandb
import numpy as np


def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def fix_width_trunc(x):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))

class WandbLogger:
    def __init__(self, short_id, id, git_info, config=None):
        self.short_id = short_id
        if self.short_id == 'NULL':
            self.short_id = 'DEBUG'

        if id is None:
            self.no_log = True
            warnings.warn('Logging has been disabled.')
        else:
            self.no_log = False

            self.inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

            self.inv_seg_trans = transforms.Normalize(
                mean=[-0.5/0.5],
                std=[1/0.5])

            # Initialize wandb
            wandb.init(
                project="xmem-egoexo",
                name=id,
                config=config,
                tags=["xmem", "egoexo", "correspondence"],
                settings=wandb.Settings(git_commit=None, git_remote=None, git_root=None)
            )

        self.log_string('git', git_info)

    def log_scalar(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        wandb.log({tag: x}, step=step)

    def log_metrics(self, l1_tag, l2_tag, val, step, f=None):
        tag = l1_tag + '/' + l2_tag
        text = '{:s} - It {:6d} [{:5s}] [{:13}]: {:s}'.format(self.short_id, step, l1_tag.upper(), l2_tag, fix_width_trunc(val))
        print(text)
        if f is not None:
            f.write(text + '\n')
            f.flush()
        self.log_scalar(tag, val, step)

    def log_im(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_im_trans(x)
        x = tensor_to_numpy(x)
        # Convert to HWC format for wandb
        x = x.transpose(1, 2, 0)
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_cv2(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        # x is already in HWC format (cv2 format)
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_seg(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_seg_trans(x)
        x = tensor_to_numpy(x)
        # Convert to HWC format for wandb
        x = x.transpose(1, 2, 0)
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_gray(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = tensor_to_numpy(x)
        # Convert to HWC format for wandb
        x = x.transpose(1, 2, 0)
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_string(self, tag, x):
        print(tag, x)
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        # Log as text in wandb
        wandb.log({f"{tag}_text": x})
        
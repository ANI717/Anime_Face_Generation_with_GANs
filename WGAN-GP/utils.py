#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import torch
import config
from pathlib import Path
from torchvision.utils import save_image


# Load Checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Save Checkpoint
def save_checkpoint(model, optimizer, filename='checkpoint/my_checkpoint.pth.tar'):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


# Save Sample Image
def save_sample_image(real, fake, epoch, step):
    Path(config.RESULTS).mkdir(parents=True, exist_ok=True)
    save_image(real, os.path.join(config.RESULTS, 'real.png'))
    save_image(fake, os.path.join(config.RESULTS, f'fake_epoch{epoch}_step{step}.png'))


# Function to impose Gradient Penalty while Training
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
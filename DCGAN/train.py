#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from pathlib import Path

import config
from datagen import Datagen
from models import Discriminator, Generator, initialize_weights
from utils import load_checkpoint, save_checkpoint, save_sample_image


# Load Data
dataset = Datagen('../Dataset/data.csv', '../Dataset/images', transform=config.TRANSFORMS)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
fixed_noise = torch.randn(32, config.Z_DIM, 1, 1).to(config.DEVICE)


# Initialize Model
model_gen = Generator(config.Z_DIM, config.IMG_CHANNELS, config.GEN_FEATURES).to(config.DEVICE)
model_disc = Discriminator(config.IMG_CHANNELS, config.DISC_FEATURES).to(config.DEVICE)
initialize_weights(model_gen)
initialize_weights(model_disc)


# Initialize Optimizer and Loss
optimizer_gen = optim.Adam(model_gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(model_disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# Load Checkpoint
if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_GEN, model_gen, optimizer_gen, config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_DISC, model_disc, optimizer_disc, config.LEARNING_RATE)


# Test Block
print(next(iter(dataset)).shape)
import sys
sys.exit()


# # Writer for Tensorboard
# writer_real = SummaryWriter('logs/real')
# writer_fake = SummaryWriter('logs/fake')
step = 0


# Training
for epoch in range(config.NUM_EPOCHS):
    for batch_idx, real in enumerate(tqdm(loader, leave=True)):
        real = real.to(config.DEVICE)
        noise = torch.randn((real.shape[0], config.Z_DIM, 1, 1)).to(config.DEVICE)
        fake = model_gen(noise)
        
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = model_disc(real).reshape(-1)
        disc_fake = model_disc(fake).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/2
        
        model_disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        optimizer_disc.step()
        
        # Train Generator: min log(1 - D(G(z)))
        output = model_disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        
        model_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
        
        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = model_gen(fixed_noise)
                
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                
                # writer_real.add_image("Real", img_grid_real, global_step=step)
                # writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
                # Save Sample Generated Images
                save_sample_image(img_grid_real, img_grid_fake, epoch, step)
                
            step += 1
    
    # Save Model in Every Epoch
    if config.SAVE_MODEL:
        Path(config.CHECKPOINT_GEN.split('/')[0]).mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_gen, optimizer_gen, config.CHECKPOINT_GEN)
        
        Path(config.CHECKPOINT_GEN.split('/')[0]).mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_disc, optimizer_disc, config.CHECKPOINT_DISC)
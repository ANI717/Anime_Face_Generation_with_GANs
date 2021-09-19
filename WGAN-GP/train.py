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
from models import Generator, Critic, initialize_weights
from utils import load_checkpoint, save_checkpoint, save_sample_image, gradient_penalty


# Load Data
dataset = Datagen('../Dataset/data.csv', '../Dataset/images', transform=config.TRANSFORMS)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
fixed_noise = torch.randn(32, config.Z_DIM, 1, 1).to(config.DEVICE)


# Initialize Model
model_gen = Generator(config.Z_DIM, config.IMG_CHANNELS, config.GEN_FEATURES).to(config.DEVICE)
model_critic = Critic(config.IMG_CHANNELS, config.CRITIC_FEATURES).to(config.DEVICE)
initialize_weights(model_gen)
initialize_weights(model_critic)


# Initialize Optimizer and Loss
optimizer_gen = optim.Adam(model_gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))
optimizer_critic = optim.Adam(model_critic.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))
criterion = nn.BCELoss()


# Load Checkpoint
if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_GEN, model_gen, optimizer_gen, config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_CRITIC, model_critic, optimizer_critic, config.LEARNING_RATE)


# # Test Block
# print(next(iter(dataset)).shape)
# import sys
# sys.exit()


# # Writer for Tensorboard
# writer_real = SummaryWriter('logs/real')
# writer_fake = SummaryWriter('logs/fake')
step = 0


# Training
for epoch in range(config.NUM_EPOCHS):
    for batch_idx, real in enumerate(tqdm(loader)):
        real = real.to(config.DEVICE)
        
        # Train Discriminator:
        for _ in range(config.CRITIC_ITERATIONS):
            noise = torch.randn((real.shape[0], config.Z_DIM, 1, 1)).to(config.DEVICE)
            fake = model_gen(noise)
            
            critic_real = model_critic(real).reshape(-1)
            critic_fake = model_critic(fake).reshape(-1)
            
            gp = gradient_penalty(model_critic, real, fake, device=config.DEVICE)
            loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake)) + config.LAMBDA_GP*gp
            
            model_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()
        
        # Train Generator:
        output = model_critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        
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
        save_checkpoint(model_critic, optimizer_critic, config.CHECKPOINT_CRITIC)
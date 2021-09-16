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
import torchvision.transforms as transforms
from tqdm import tqdm

from model import Discriminator, Generator, initialize_weights
from datagen import Datagen


# Set Hyperparameters
IMG_SIZE = (64,64)
IMG_CHANNELS = 3
Z_DIM = 200

GEN_FEATURES = 128
DISC_FEATURES = 128

BATCH_SIZE = 128
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10


# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Data
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*IMG_CHANNELS, [0.5]*IMG_CHANNELS),
    ])

dataset = Datagen('../Dataset/data.csv', '../Dataset/images', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)


# Initialize Model
model_gen = Generator(Z_DIM, IMG_CHANNELS, GEN_FEATURES).to(device)
model_disc = Discriminator(IMG_CHANNELS, DISC_FEATURES).to(device)
initialize_weights(model_gen)
initialize_weights(model_disc)


# Initialize Optimizer and Loss
optimizer_gen = optim.Adam(model_gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(model_disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# # Test Block
# print(next(iter(dataset)).shape)
# import sys
# sys.exit()


# Writer for Tensorboard
writer_real = SummaryWriter('logs/real')
writer_fake = SummaryWriter('logs/fake')
step = 0


# Training
for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(tqdm(loader)):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
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
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
            step += 1
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

from model import Generator, Critic, initialize_weights
from datagen import Datagen


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


# Set Hyperparameters
IMG_SIZE = (64,64)
IMG_CHANNELS = 3
Z_DIM = 100

GEN_FEATURES = 64
CRITIC_FEATURES = 64

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


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
model_critic = Critic(IMG_CHANNELS, CRITIC_FEATURES).to(device)
initialize_weights(model_gen)
initialize_weights(model_critic)


# Initialize Optimizer and Loss
optimizer_gen = optim.Adam(model_gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
optimizer_critic = optim.Adam(model_critic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
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
        
        # Train Discriminator:
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((real.shape[0], Z_DIM, 1, 1)).to(device)
            fake = model_gen(noise)
            
            critic_real = model_critic(real).reshape(-1)
            critic_fake = model_critic(fake).reshape(-1)
            
            gp = gradient_penalty(model_critic, real, fake, device=device)
            loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
            
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
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
            step += 1
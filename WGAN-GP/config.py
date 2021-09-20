#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torchvision.transforms as transforms


# Hyperparameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (64,64)
IMG_CHANNELS = 3
Z_DIM = 100

GEN_FEATURES = 64
CRITIC_FEATURES = 64

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = 'checkpoint/generator.pth.tar'
CHECKPOINT_CRITIC = 'checkpoint/critic.pth.tar'
RESULTS = 'results'


# Transformations
TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*IMG_CHANNELS, [0.5]*IMG_CHANNELS),
])
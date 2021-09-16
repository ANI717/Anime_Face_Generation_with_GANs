#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torch.nn as nn


# Generator Model Class
class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, features):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(noise_channels, features*16, 4, 1, 0),  # 1x1 --> 4x4
            self._block(features*16, features*8, 4, 2, 1),  # 4x4 --> 8x8
            self._block(features*8, features*4, 4, 2, 1),  # 8x8 --> 16x16
            self._block(features*4, features*2, 4, 2, 1),  # 16x16 --> 32x32
            nn.ConvTranspose2d(features*2, img_channels, 4, 2, 1), # 32x32 --> 64x64
            nn.Tanh(),
            )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.ReLU(),
            )

    def forward(self, x):
        return self.net(x)


# Critic Model Class
class Critic(nn.Module):
    def __init__(self, img_channels, features):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(            
            nn.Conv2d(img_channels, features, 4, 2, 1), # n --> (n-k+s+2p)/s: 64x64 --> 32x32
            nn.LeakyReLU(0.2), # 32x32 --> 32x32
            self._block(features, features*2, 4, 2, 1), # 32x32 --> 16x16
            self._block(features*2, features*4, 4, 2, 1), # 16x16 --> 8x8
            self._block(features*4, features*8, 4, 2, 1), # 4 x 4
            nn.Conv2d(features*8, 1, 4, 2, 0), # 1 x 1
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
    
    def forward(self, x):
        return self.disc(x)



# Weight Initialization to Achieve Proper Result
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)





# Test Block
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    
    critic = Critic(in_channels, 8)
    initialize_weights(critic)
    assert critic(x).shape == (N, 1, 1, 1)
    
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")

if __name__ == "__main__":
    test()

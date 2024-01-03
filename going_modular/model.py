
import torch
import torchvision
from torch import nn
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, configs) :
        super().__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=configs.letent_size_z,
                               out_channels=configs.num_fmap_gen*8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(configs.num_fmap_gen*8),
            nn.ReLU(),
            #output : (batch, num_fmap_gen*8, 4, 4)

            nn.ConvTranspose2d(configs.num_fmap_gen*8, configs.num_fmap_gen*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_fmap_gen*4),
            nn.ReLU(),
            #output : (batch, num_fmap_gen*4, 8, 8)

            nn.ConvTranspose2d(configs.num_fmap_gen*4, configs.num_fmap_gen*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_fmap_gen*2),
            nn.ReLU(),
            #output : (batch, num_fmap_gen*2, 16, 16)

            nn.ConvTranspose2d(configs.num_fmap_gen*2, configs.num_fmap_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_fmap_gen),
            nn.ReLU(),
            #output : (batch, num_fmap_gen, 32, 32)

            nn.ConvTranspose2d(configs.num_fmap_gen, configs.num_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_channels),
            nn.Tanh()
            #output : (batch, num_channels, 64, 64)
            )
        
    def forward(self, x):
        return self.main(x)
    


class Discriminator(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=configs.num_channels,
                      out_channels=configs.num_fmap_dis,
                      kernel_size=4,
                      stride=2, 
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #out shape (batch, num_fmap_dis, 32, 32)

            nn.Conv2d(configs.num_fmap_dis, configs.num_fmap_dis*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_fmap_dis*2),
            nn.LeakyReLU(0.2, inplace=True),
            #(batch, configs.num_fmap_dis*2, 16, 16)

            nn.Conv2d(configs.num_fmap_dis*2, configs.num_fmap_dis*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_fmap_dis*4),
            nn.LeakyReLU(0.2, inplace=True),
            #(batch, configs.num_fmap_dis*4, 8, 8)

            nn.Conv2d(configs.num_fmap_dis*4, configs.num_fmap_dis*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(configs.num_fmap_dis*8),
            nn.LeakyReLU(0.2, inplace=True),
            #(batch, configs.num_fmap_dis*8, 4, 4)

            nn.Conv2d(configs.num_fmap_dis*8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
            #(batch, 1, 1, 1)
        )

    def forward(self, x):
        return self.main(x)
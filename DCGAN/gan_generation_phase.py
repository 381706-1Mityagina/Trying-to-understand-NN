import torch
import torch.nn as net
from torch.autograd import Variable
from torch.nn import functional as F

class Generator(net.Module):
    def __init__(self, batch_size, image_size, z_dim, embed_dim, reduced_dim):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.reduced_dim = reduced_dim

        self.reduced_dim = net.Linear(embed_dim, reduced_dim)
        self.concat = net.Linear(z_dim + reduced_dim, 64 * 8 * 4 * 4)

        # Архитектура
        self.d_net = net.Sequential(
            net.ReLU(),
            net.ConvTranspose2d(512, 256, 4, 2, 1),
            net.BatchNorm2d(256),
            net.ReLU(),
            net.ConvTranspose2d(256, 128, 4, 2, 1),
            net.BatchNorm2d(128),
            net.ReLU(),
            net.ConvTranspose2d(128, 64, 4, 2, 1),
            net.BatchNorm2d(64),
            net.ReLU(),
            net.ConvTranspose2d(64, 3, 4, 2, 1),
            net.Tanh()
        )

    def forward(self, text_discr, z):
        # (batch_size, reduced_dim)
        reduced_text = self.reduced_dim(text_discr)
        # (batch_size, reduced_dim + z_dim)
        concat = torch.cat((reduced_text, z), 1)
        # (batch_size, 64*8*4*4)
        concat = self.concat(concat)
        # (batch_size, 4, 4, 64*8)
        concat = torch.view(-1, 4, 4, 64 * 8)
        # (batch_size, 64, 64, 3)
        d_net_out = self.d_net(concat)
        # (batch_size, 64, 64, 3)
        output = d_net_out / 2. + 0.5

        # output : image (64, 64, 3)
        return output
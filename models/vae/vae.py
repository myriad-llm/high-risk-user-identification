import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.en_fc1 = nn.Linear(in_channels, 1024)
        self.en_fc2_mu = nn.Linear(1024, latent_dim)
        self.en_fc2_sigma = nn.Linear(1024, latent_dim)
        self.de_fc3 = nn.Linear(latent_dim, 1024)
        self.de_fc4 = nn.Linear(1024, in_channels)

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.en_fc1(x)
        x = torch.relu(x)
        return self.en_fc2_mu(x), self.en_fc2_sigma(x)

    def decode(self, z):
        z = self.de_fc3(z)
        z = torch.relu(z)
        z = self.de_fc4(z)
        z = torch.sigmoid(z)
        return z

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma


import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
    def __init__(self, input_dim=64 * 64 * 3, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, 3, 64, 64)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


if __name__ == "__main__":
    model = SimpleVAE()
    x = torch.randn(4, 3, 64, 64)
    recon, mu, logvar = model(x)

    print("Input shape:", x.shape)
    print("Reconstruction shape:", recon.shape)
    print("Latent mean shape:", mu.shape)
    print("Latent logvar shape:", logvar.shape)

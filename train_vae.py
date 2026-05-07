import os
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image


# ── Custom Dataset (no subfolder needed) ─────────────────────────────────────
class AnimeDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                     glob.glob(os.path.join(img_dir, "*.png"))
        assert len(self.paths) > 0, f"No images found in {img_dir}"
        print(f"Found {len(self.paths)} images in {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # 0 is a dummy label, VAE doesn't need labels


# ── Model ─────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 8  -> 4
            nn.ReLU(),
        )

        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4  -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8  -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32 -> 64
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ── Loss ──────────────────────────────────────────────────────────────────────
def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ── Utilities ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def save_samples(model, device, latent_dim, out_path, n=64):
    model.eval()
    z = torch.randn(n, latent_dim).to(device)
    utils.save_image(model.decode(z), out_path, nrow=8)


@torch.no_grad()
def save_interpolation(model, device, latent_dim, out_path, steps=10):
    model.eval()
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    alphas = torch.linspace(0, 1, steps).to(device)
    zs = torch.cat([(1 - a) * z1 + a * z2 for a in alphas], dim=0)
    utils.save_image(model.decode(zs), out_path, nrow=steps)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="data")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--image_size",  type=int,   default=64)
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Data
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset    = AnimeDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model     = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_recon = total_kl = 0

        for x, _ in dataloader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=args.beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl_loss.item()

        n = len(dataloader)
        print(
            f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Loss: {total_loss/n:.2f} | "
            f"Recon: {total_recon/n:.2f} | "
            f"KL: {total_kl/n:.2f}"
        )

        save_samples(model, device, args.latent_dim, f"results/generated_epoch_{epoch:03d}.png")

    # Final outputs
    save_interpolation(model, device, args.latent_dim, "results/latent_interpolation.png")
    torch.save(model.state_dict(), "checkpoints/vae_anime_faces.pt")
    print("Training finished. Results saved to results/")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image


# ── Model (must match train_vae.py) ───────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 256, 4, 4)
        return self.decoder(h)


# ── Load model ────────────────────────────────────────────────────────────────
LATENT_DIM = 128
CHECKPOINT = "checkpoints/vae_anime_faces.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Model loaded on {device}")


# ── Helper ────────────────────────────────────────────────────────────────────
def tensor_to_pil(t):
    arr = (t.cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))  # CHW -> HWC


# ── Tab 1: Random Generation ──────────────────────────────────────────────────
def generate_faces(n_images):
    with torch.no_grad():
        z = torch.randn(n_images, LATENT_DIM).to(device)
        imgs = model.decode(z)
    return [tensor_to_pil(imgs[i]) for i in range(n_images)]


# ── Tab 2: Latent Interpolation ───────────────────────────────────────────────
def interpolate_faces(steps):
    with torch.no_grad():
        z1 = torch.randn(1, LATENT_DIM).to(device)
        z2 = torch.randn(1, LATENT_DIM).to(device)
        alphas = torch.linspace(0, 1, steps).to(device)
        zs = torch.cat([(1 - a) * z1 + a * z2 for a in alphas], dim=0)
        imgs = model.decode(zs)
    return [tensor_to_pil(imgs[i]) for i in range(steps)]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Anime Face VAE Gallery") as demo:
    gr.Markdown("# 🎨 Anime Face Generation with VAE")
    gr.Markdown("Generate anime-style faces using a Variational Autoencoder trained on 63,565 images.")

    with gr.Tab("🎲 Random Generation"):
        gr.Markdown("Generate random anime faces by sampling from the latent space.")
        n_slider = gr.Slider(minimum=1, maximum=16, value=8, step=1, label="Number of faces")
        gen_btn  = gr.Button("Generate!", variant="primary")
        gallery  = gr.Gallery(label="Generated Faces", columns=4, height="auto")
        gen_btn.click(fn=generate_faces, inputs=n_slider, outputs=gallery)

    with gr.Tab("🔀 Latent Interpolation"):
        gr.Markdown("Smoothly interpolate between two random faces in latent space.")
        steps_slider = gr.Slider(minimum=3, maximum=12, value=8, step=1, label="Interpolation steps")
        interp_btn   = gr.Button("Interpolate!", variant="primary")
        interp_gallery = gr.Gallery(label="Interpolation", columns=8, height="auto")
        interp_btn.click(fn=interpolate_faces, inputs=steps_slider, outputs=interp_gallery)

demo.launch(share=True)

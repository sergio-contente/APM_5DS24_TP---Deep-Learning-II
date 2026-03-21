"""Bonus: Compare generative models (RBM, DBN, VAE, GAN, Diffusion) on MNIST.
All models have approximately the same number of parameters (~160k)."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import load_mnist
from rbm import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM
from dbn import init_DBN, train_DBN

# ============================================================
# Config
# ============================================================
N_IMAGES = 10
MNIST_SHAPE = (28, 28)
DEVICE = torch.device("cpu")

print("Loading MNIST...")
X_train, _, _, _ = load_mnist()
print(f"Train: {X_train.shape}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=128, shuffle=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 1. RBM (784 -> 200) ~ 157k params
# ============================================================
print("\n" + "=" * 50)
print("1. Training RBM (784 -> 200)")
print("=" * 50)
rbm = init_RBM(784, 200)
print(f"   Params: {784*200 + 784 + 200}")
train_RBM(rbm, X_train, epochs=100, lr=0.1, batch_size=128)

# Generate via Gibbs sampling
v = (np.random.rand(N_IMAGES, 784) > 0.5).astype(np.float64)
for _ in range(1000):
    h_prob = entree_sortie_RBM(rbm, v)
    h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float64)
    v_prob = sortie_entree_RBM(rbm, h_sample)
    v = (np.random.rand(*v_prob.shape) < v_prob).astype(np.float64)
images_rbm = v


# ============================================================
# 2. DBN (784 -> 200 -> 100) ~ 178k params
# ============================================================
print("\n" + "=" * 50)
print("2. Training DBN (784 -> 200 -> 100)")
print("=" * 50)
dbn = init_DBN([784, 200, 100])
print(f"   Params: {784*200+784+200 + 200*100+200+100}")
train_DBN(dbn, X_train, epochs=100, lr=0.1, batch_size=128)

# Generate: Gibbs at top RBM, then top-down
top_rbm = dbn[-1]
v_top = (np.random.rand(N_IMAGES, top_rbm["W"].shape[0]) > 0.5).astype(np.float64)
for _ in range(1000):
    h_p = entree_sortie_RBM(top_rbm, v_top)
    h_s = (np.random.rand(*h_p.shape) < h_p).astype(np.float64)
    v_p = sortie_entree_RBM(top_rbm, h_s)
    v_top = (np.random.rand(*v_p.shape) < v_p).astype(np.float64)
images_dbn = sortie_entree_RBM(dbn[0], v_top)


# ============================================================
# 3. VAE ~ 160k params
# ============================================================
print("\n" + "=" * 50)
print("3. Training VAE")
print("=" * 50)


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


vae = VAE().to(DEVICE)
print(f"   Params: {count_params(vae)}")
optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(50):
    total_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        recon, mu, logvar = vae(batch)
        bce = nn.functional.binary_cross_entropy(recon, batch, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (bce + kld) / batch.size(0)
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"  VAE Epoch {epoch+1}/50 - Loss: {total_loss / len(train_loader):.2f}")

with torch.no_grad():
    z = torch.randn(N_IMAGES, 10).to(DEVICE)
    images_vae = vae.decode(z).cpu().numpy()


# ============================================================
# 4. GAN ~ 160k params
# ============================================================
print("\n" + "=" * 50)
print("4. Training GAN")
print("=" * 50)

LATENT_DIM_GAN = 64


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM_GAN, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)
print(f"   Generator params: {count_params(gen)}")
print(f"   Discriminator params: {count_params(disc)}")
print(f"   Total params: {count_params(gen) + count_params(disc)}")

opt_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCELoss()

for epoch in range(100):
    total_g, total_d = 0, 0
    for (real,) in train_loader:
        real = real.to(DEVICE)
        bs = real.size(0)
        ones = torch.ones(bs, 1, device=DEVICE)
        zeros = torch.zeros(bs, 1, device=DEVICE)

        # Train Discriminator
        z = torch.randn(bs, LATENT_DIM_GAN, device=DEVICE)
        fake = gen(z).detach()
        loss_d = bce(disc(real), ones) + bce(disc(fake), zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Train Generator
        z = torch.randn(bs, LATENT_DIM_GAN, device=DEVICE)
        fake = gen(z)
        loss_g = bce(disc(fake), ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        total_d += loss_d.item()
        total_g += loss_g.item()
    if (epoch + 1) % 20 == 0:
        print(f"  GAN Epoch {epoch+1}/100 - D: {total_d/len(train_loader):.3f}, G: {total_g/len(train_loader):.3f}")

with torch.no_grad():
    z = torch.randn(N_IMAGES, LATENT_DIM_GAN, device=DEVICE)
    images_gan = gen(z).cpu().numpy()


# ============================================================
# 5. Diffusion Model (simple DDPM) ~ 160k params
# ============================================================
print("\n" + "=" * 50)
print("5. Training Diffusion Model")
print("=" * 50)

T_STEPS = 100
betas = torch.linspace(1e-4, 0.02, T_STEPS)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


class DenoisingNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, t_emb_dim=16):
        super().__init__()
        self.t_embed = nn.Sequential(nn.Linear(1, t_emb_dim), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(input_dim + t_emb_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        t_emb = self.t_embed(t.float().unsqueeze(-1) / T_STEPS)
        return self.net(torch.cat([x, t_emb], dim=-1))


diffusion = DenoisingNet().to(DEVICE)
print(f"   Params: {count_params(diffusion)}")
opt_diff = optim.Adam(diffusion.parameters(), lr=1e-3)

for epoch in range(50):
    total_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        bs = batch.size(0)
        t = torch.randint(0, T_STEPS, (bs,), device=DEVICE)
        noise = torch.randn_like(batch)
        ab = alpha_bar[t].unsqueeze(-1).to(DEVICE)
        x_noisy = torch.sqrt(ab) * batch + torch.sqrt(1 - ab) * noise
        pred_noise = diffusion(x_noisy, t)
        loss = nn.functional.mse_loss(pred_noise, noise)
        opt_diff.zero_grad()
        loss.backward()
        opt_diff.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"  Diffusion Epoch {epoch+1}/50 - Loss: {total_loss/len(train_loader):.4f}")

# Sample via reverse process
with torch.no_grad():
    x = torch.randn(N_IMAGES, 784, device=DEVICE)
    for t in reversed(range(T_STEPS)):
        t_batch = torch.full((N_IMAGES,), t, device=DEVICE)
        pred_noise = diffusion(x, t_batch)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        if t > 0:
            x += torch.sqrt(beta_t) * torch.randn_like(x)
    images_diff = torch.clamp(x, 0, 1).cpu().numpy()


# ============================================================
# Comparison figure
# ============================================================
print("\n" + "=" * 50)
print("Generating comparison figure...")
print("=" * 50)

all_images = {
    "RBM\n(784->200)": images_rbm,
    "DBN\n(784->200->100)": images_dbn,
    "VAE\n(784->100->10)": images_vae,
    "GAN\n(64->128->784)": images_gan,
    "Diffusion\n(DDPM, T=100)": images_diff,
}

n_models = len(all_images)
fig, axes = plt.subplots(n_models, N_IMAGES, figsize=(2 * N_IMAGES, 2.5 * n_models))

for row, (name, imgs) in enumerate(all_images.items()):
    for col in range(N_IMAGES):
        ax = axes[row, col]
        ax.imshow(imgs[col].reshape(MNIST_SHAPE), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(name, fontsize=11, rotation=0, labelpad=80, va="center")

plt.suptitle("Generative Models Comparison on MNIST (~160k params each)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("bonus_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone! Saved to bonus_comparison.png")

# Print param counts summary
print("\nParameter counts:")
print(f"  RBM:       {784*200 + 784 + 200:,}")
print(f"  DBN:       {784*200+784+200 + 200*100+200+100:,}")
print(f"  VAE:       {count_params(vae):,}")
print(f"  GAN:       {count_params(gen) + count_params(disc):,}")
print(f"  Diffusion: {count_params(diffusion):,}")

"""
Conditional WGAN-GP for Pyocyanin Voltammogram Generation

Why WGAN-GP?
------------
WGAN-GP was chosen for a few key reasons, mostly because standard GANs 
really struggle with our specific data distribution:

1. **Wasserstein distance for the win** — Classic GAN divergence (JS/KL) 
   tends to collapse when the real distribution is highly concentrated 
   (like having only 40 real signals tightly clustered per concentration). 
   The W-distance stays well-defined even when the generator and data 
   distributions don't overlap, giving us stable, meaningful gradients 
   the whole way through.

2. **Gradient Penalty > Weight clipping** — The original WGAN clipped 
   critic weights to enforce the Lipschitz constraint, which bottlenecks 
   model capacity and can cause vanishing gradients. The Gradient Penalty 
   enforces this softly, letting the critic use its full expressive power. 
   We absolutely need this to capture the subtle physical variations 
   (peak position, FWHM, skewness) that our validation gate looks for.

3. **Conditional generation** — Concentration spans three decades 
   (0.1 - 100 µM). By injecting log10(c) as a conditioning vector into 
   both the generator and the critic, the model learns the monotonic 
   Ip ∝ c^0.85 calibration curve we see in the data, while still allowing 
   for natural variation within the same concentration class.

4. **1-D convolutional setup** — Our 229-point signal is a structured 
   sequence, not just tabular data. Using 1D conv layers with causal-like 
   dilation lets us capture multi-scale structures: the broad baseline 
   slope, the actual peak shape, and the high-frequency instrument noise. 
   This beats an MLP hands-down when spatial locality matters.

What the Data Looks Like
------------------------
- 229 points, E ∈ [-0.600, 0.502] V, uniform Δ ≈ 4.84 mV.
- Zeroed boundaries: I[0] = I[228] = 0.0 (enforced by the architecture).
- The action happens in the peak region: E ∈ [-0.55, -0.25] (indices 11- 72).
- Peak current (Ip) scales almost linearly with concentration in log-log 
  space (R² > 0.99).
- Because we only have 40 real signals, we train on the combined dataset 
  (340 signals) to keep training stable.

Usage
-----
    python wgangp_training.py                     # Run the full training pipeline
    python wgangp_training.py --epochs 5000       # Set a custom epoch count
    python wgangp_training.py --eval-only         # Generate signals from a saved model

Output Files
------------
    models/wgangp_generator.pt   — saved generator weights
    models/wgangp_config.json    — saved hyperparameters (for the eval notebook)
    samples/wgangp_signals.csv   — the generated signals (formatted to match real data)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 0. HYPERPARAMETERS  (edit freely for fine-tuning)
HP = {
    # Data 
    "real_csv":       "raw/raw_signals_real.csv",
    "combined_csv":   "raw/raw_signals_combined.csv",
    "potential_csv":  "raw/raw_potential_grid.csv",
    "train_on":       "real",         # "real" | "combined"
    "signal_len":     229,                # number of current points per signal

    # Architecture
    "latent_dim":     128,                # noise vector size
    "cond_dim":       16,                 # concentration embedding size
    "gen_channels":  [256, 128, 64, 32],  # conv channels in generator blocks
    "crit_channels": [32,  64, 128, 256], # conv channels in critic blocks
    "kernel_size":    9,                  # conv kernel (must be odd)

    # Training 
    "epochs":         8000,
    "batch_size":     32,
    "lr_gen":         1e-4,
    "lr_crit":        1e-4,
    "betas":          (0.0, 0.9),         # Adam β — standard for WGAN
    "n_critic":       5,                  # critic steps per generator step
    "lambda_gp":      10.0,               # gradient penalty coefficient
    "clip_output":    True,               # clip generator output to [0, 40] µA

    # Regularisation 
    "dropout":        0.1,
    "spectral_norm":  True,               # spectral norm on critic conv layers

    #Logging 
    "log_every":      200,
    "save_every":     2000,
    "seed":           42,

    # Generation 
    "n_generate":     300,                # signals to generate after training
    "gen_conc_mode":  "log_uniform",      # "log_uniform" | "real_classes"
    "conc_min":       0.1,
    "conc_max":       100.0,
}


# 1. UTILITIES
def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"  Device: {dev}")
    return dev


def log10_conc_normalise(c: np.ndarray,
                          c_min: float = 0.1,
                          c_max: float = 100.0) -> np.ndarray:
    """
    Map concentration to [-1, 1] via log10 scaling.
    
    log10(0.1) = -1, log10(100) = 2  →  mapped to [-1, 1].
    Using log10 space because Ip ∝ c (linear in log-log), which means
    the generator receives a perceptually uniform conditioning signal.
    """
    log_c    = np.log10(np.clip(c, 1e-9, None))
    log_min  = np.log10(c_min)
    log_max  = np.log10(c_max)
    return 2.0 * (log_c - log_min) / (log_max - log_min) - 1.0  # → [-1, 1]


def log10_conc_denormalise(c_norm: np.ndarray,
                            c_min: float = 0.1,
                            c_max: float = 100.0) -> np.ndarray:
    """Inverse of log10_conc_normalise."""
    log_min = np.log10(c_min)
    log_max = np.log10(c_max)
    log_c   = (c_norm + 1.0) / 2.0 * (log_max - log_min) + log_min
    return 10.0 ** log_c


def load_data(hp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare training data.

    Returns
    -------
    X      : (N, signal_len) float32 current matrix
    y_norm : (N,) float32 log-normalised concentration labels in [-1, 1]
    E      : (signal_len,) float64 potential grid
    """
    csv_key = "combined_csv" if hp["train_on"] == "combined" else "real_csv"
    df  = pd.read_csv(hp[csv_key])
    E   = pd.read_csv(hp["potential_csv"])["potential_V"].values.astype(np.float64)

    I_cols = [c for c in df.columns if c.startswith("I_")]
    X      = df[I_cols].values.astype(np.float32)          # (N, 229)
    y_raw  = df["concentration"].values.astype(np.float32)

    assert X.shape[1] == hp["signal_len"], (
        f"Signal length mismatch: CSV has {X.shape[1]}, HP expects {hp['signal_len']}")

    y_norm = log10_conc_normalise(y_raw,
                                   hp["conc_min"],
                                   hp["conc_max"]).astype(np.float32)

    print(f"  Loaded {len(X)} signals  |  "
          f"conc range [{y_raw.min():.3f}, {y_raw.max():.3f}] µM  |  "
          f"signal len {X.shape[1]}")
    return X, y_norm, E


# 2. MODEL ARCHITECTURE
class ConditionalEmbedding(nn.Module):
    """
    Maps a scalar concentration (normalised, shape [B,1]) to a
    conditioning vector of size `cond_dim`.

    A simple 2-layer MLP with Tanh is sufficient; the conditioning signal
    is a single scalar so no complex embedding is needed.
    """
    def __init__(self, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.Tanh(),
            nn.Linear(cond_dim, cond_dim),
            nn.Tanh(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: (B, 1)  →  (B, cond_dim)
        return self.net(c)


class ResBlock1D(nn.Module):
    """
    1-D residual block with two conv layers and a skip connection.
    
    Residual connections allow gradients to flow through many layers
    without vanishing, which is important for the 229-point sequence.
    """
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2  # same-padding
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.InstanceNorm1d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.InstanceNorm1d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class Generator(nn.Module):
    """
    Conditional 1-D convolutional generator.

    Architecture:
        noise(z) + cond(c)  →  Linear  →  reshape  →  [UpBlock × 4]  →  tanh

    The generator starts from a dense representation and upsamples to the
    full sequence length via transposed convolutions (learned upsampling).

    Boundary enforcement:
        The first and last points of every real signal are exactly 0.
        We enforce this by multiplying the output by a pre-computed mask
        that ramps from 0 at the edges to 1 in the interior.  This ensures
        I[0] = I[228] = 0 without any gradient penalty.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.hp          = hp
        self.latent_dim  = hp["latent_dim"]
        self.cond_dim    = hp["cond_dim"]
        self.signal_len  = hp["signal_len"]
        channels         = hp["gen_channels"]   # [256, 128, 64, 32]
        ks               = hp["kernel_size"]

        # Conditioning embedding
        self.cond_emb = ConditionalEmbedding(self.cond_dim)

        # Project latent + cond to initial sequence
        # We start at length signal_len // 8 = 28 and upsample ×8 → 229
        self.init_len  = 29                       # ≈ 229/8, chosen for clean upsampling
        self.fc        = nn.Linear(self.latent_dim + self.cond_dim,
                                   channels[0] * self.init_len)

        # Upsample blocks: 29 → 57 → 115 → 229
        self.up_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            # Upsample by ×2 then refine with conv
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock1D(out_ch, ks, hp["dropout"]),
            )
            self.up_blocks.append(block)
            in_ch = out_ch

        # Final projection to single channel
        self.final_conv = nn.Sequential(
            nn.Conv1d(channels[-1], 16, ks, padding=ks // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 1, 1),   # pointwise to output
        )

        # Boundary mask: smooth taper at both ends (first and last 3 points → 0)
        # This is a fixed non-learnable constraint matching I[0]=I[228]=0
        mask            = torch.ones(1, 1, self.signal_len)
        mask[0, 0, :3]  = torch.linspace(0, 1, 3)
        mask[0, 0, -3:] = torch.linspace(1, 0, 3)
        self.register_buffer("boundary_mask", mask)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)  noise vector
        c : (B, 1)           log-normalised concentration in [-1, 1]

        Returns
        -------
        x : (B, 229)  generated current signal
        """
        c_emb  = self.cond_emb(c)                     # (B, cond_dim)
        h      = torch.cat([z, c_emb], dim=1)         # (B, latent+cond)
        h      = self.fc(h)                            # (B, ch[0]*init_len)
        h      = h.view(-1, self.hp["gen_channels"][0], self.init_len)  # (B, C, 29)

        for block in self.up_blocks:
            h = block(h)                               # (B, C_i, L_i)

        # Crop or pad to exact target length
        h = h[:, :, :self.signal_len]
        if h.shape[2] < self.signal_len:
            h = torch.nn.functional.pad(h, (0, self.signal_len - h.shape[2]))

        h = self.final_conv(h)                         # (B, 1, 229)

        # Activation: Softplus to enforce non-negative output in peak region.
        # Signals can have very small negative values in the baseline, so we
        # apply Softplus with a small bias shift rather than ReLU.
        h = torch.nn.functional.softplus(h) - 0.05    # slight negative allowed

        # Optional hard clamp to physical range [0, 40] µA
        if self.hp.get("clip_output", True):
            h = h.clamp(-1.0, 40.0)

        # Apply boundary mask (I[0] = I[228] = 0)
        h = h * self.boundary_mask                     # (B, 1, 229)

        return h.squeeze(1)                            # (B, 229)


class Critic(nn.Module):
    """
    Conditional 1-D convolutional WGAN-GP critic (discriminator).

    The critic must be Lipschitz-1.  We enforce this via gradient penalty
    rather than weight clipping.  Spectral normalisation is optionally
    added on top for additional stability.

    Architecture:
        signal(x) + cond(c)  →  [DownBlock × 4]  →  GlobalAvgPool  →  Linear(1)

    The conditioning vector is broadcast and concatenated channel-wise
    to the input signal, making it a fully conditional critic.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.hp       = hp
        self.cond_dim = hp["cond_dim"]
        channels      = hp["crit_channels"]   # [32, 64, 128, 256]
        ks            = hp["kernel_size"]

        # Conditioning embedding for critic
        self.cond_emb = ConditionalEmbedding(self.cond_dim)

        # First layer takes signal (1 ch) + broadcast cond (cond_dim ch)
        def maybe_sn(layer):
            if hp.get("spectral_norm", True):
                return nn.utils.spectral_norm(layer)
            return layer

        in_ch = 1 + self.cond_dim
        self.down_blocks = nn.ModuleList()
        for i, out_ch in enumerate(channels):
            stride = 2 if i < len(channels) - 1 else 1  # downsample ×2 in first 3 blocks
            pad    = ks // 2
            block  = nn.Sequential(
                maybe_sn(nn.Conv1d(in_ch, out_ch, ks, stride=stride, padding=pad)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(hp["dropout"]),
            )
            self.down_blocks.append(block)
            in_ch = out_ch

        # Output head
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global average pooling
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 229)   signal
        c : (B, 1)     log-normalised concentration

        Returns
        -------
        score : (B, 1)  unbounded Wasserstein score
        """
        c_emb = self.cond_emb(c)                     # (B, cond_dim)
        # Broadcast conditioning across sequence length: (B, cond_dim, 229)
        c_seq = c_emb.unsqueeze(2).expand(-1, -1, x.shape[1])
        h     = x.unsqueeze(1)                       # (B, 1, 229)
        h     = torch.cat([h, c_seq], dim=1)         # (B, 1+cond_dim, 229)

        for block in self.down_blocks:
            h = block(h)

        return self.output_head(h)                   # (B, 1)


# 3. GRADIENT PENALTY

def gradient_penalty(critic: Critic,
                     real:   torch.Tensor,
                     fake:   torch.Tensor,
                     c:      torch.Tensor,
                     device: torch.device,
                     lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Compute two-sided gradient penalty (Gulrajani et al., 2017).

    Samples a random convex combination  x̂ = εx_real + (1-ε)x_fake
    and penalises ‖∇_x̂ D(x̂)‖₂ deviating from 1.

    This enforces the 1-Lipschitz constraint on the critic without
    restricting its weight norms, preserving full model capacity.
    """
    B = real.shape[0]
    eps = torch.rand(B, 1, device=device, requires_grad=False)
    interpolated = (eps * real + (1 - eps) * fake).detach().requires_grad_(True)

    d_interp = critic(interpolated, c)

    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]                                               # (B, 229)

    grads_norm = grads.norm(2, dim=1)                  # (B,)
    gp = lambda_gp * ((grads_norm - 1.0) ** 2).mean()
    return gp


# 4. TRAINING LOOP
def train(hp: dict) -> tuple[Generator, list[dict]]:
    """
    Full WGAN-GP training loop.

    Returns the trained Generator and a list of per-epoch metrics.
    """
    set_seed(hp["seed"])
    device = get_device()

    # Data 
    X, y_norm, E = load_data(hp)
    X_t      = torch.tensor(X,      dtype=torch.float32)
    y_t      = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)  # (N,1)
    dataset  = TensorDataset(X_t, y_t)
    loader   = DataLoader(dataset,
                          batch_size=hp["batch_size"],
                          shuffle=True,
                          drop_last=True)

    # Models 
    G = Generator(hp).to(device)
    D = Critic(hp).to(device)

    n_params_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_params_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"  Generator params : {n_params_G:,}")
    print(f"  Critic params    : {n_params_D:,}")

    opt_G = torch.optim.Adam(G.parameters(),
                              lr=hp["lr_gen"],
                              betas=hp["betas"])
    opt_D = torch.optim.Adam(D.parameters(),
                              lr=hp["lr_crit"],
                              betas=hp["betas"])

    # LR schedulers — gentle cosine decay
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=hp["epochs"], eta_min=hp["lr_gen"] / 10)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=hp["epochs"], eta_min=hp["lr_crit"] / 10)

    history = []
    t0      = time.time()

    print(f"\n  Training WGAN-GP for {hp['epochs']} epochs …\n")

    for epoch in range(1, hp["epochs"] + 1):
        G.train(); D.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches    = 0

        for x_real, c_real in loader:
            x_real = x_real.to(device)   # (B, 229)
            c_real = c_real.to(device)   # (B, 1)
            B      = x_real.shape[0]

            # Critic update (n_critic times per generator step) 
            for _ in range(hp["n_critic"]):
                z    = torch.randn(B, hp["latent_dim"], device=device)
                x_fake = G(z, c_real).detach()          # no grad through G

                d_real = D(x_real, c_real)
                d_fake = D(x_fake, c_real)

                # Wasserstein loss: maximise E[D(real)] - E[D(fake)]
                # equivalently minimise -(E[D(real)] - E[D(fake)])
                w_dist = d_real.mean() - d_fake.mean()
                gp     = gradient_penalty(D, x_real, x_fake, c_real, device,
                                          hp["lambda_gp"])
                d_loss = -w_dist + gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # Generator update 
            z      = torch.randn(B, hp["latent_dim"], device=device)
            x_fake = G(z, c_real)
            g_loss = -D(x_fake, c_real).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches    += 1

        scheduler_G.step()
        scheduler_D.step()

        # Logging 
        avg_d = epoch_d_loss / n_batches
        avg_g = epoch_g_loss / n_batches
        history.append({"epoch": epoch, "d_loss": avg_d, "g_loss": avg_g})

        if epoch % hp["log_every"] == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:5d}/{hp['epochs']} | "
                  f"D loss: {avg_d:+.4f} | G loss: {avg_g:+.4f} | "
                  f"Time: {elapsed:.0f}s")

        if epoch % hp["save_every"] == 0:
            _save_checkpoint(G, hp, epoch)

    print(f"\n  Training complete in {time.time() - t0:.1f}s")
    return G, history


def _save_checkpoint(G: Generator, hp: dict, epoch: int) -> None:
    os.makedirs("/app/models", exist_ok=True)
    path = f"/app/models/wgangp_generator_ep{epoch}.pt"
    torch.save(G.state_dict(), path)
    print(f"    [ckpt] saved {path}")

# 5. SIGNAL GENERATION
def generate_signals(G:            Generator,
                     hp:            dict,
                     device:        torch.device,
                     n:             int  | None = None,
                     conc_values:   np.ndarray | None = None) -> pd.DataFrame:
    """
    Generate synthetic voltammogram signals using the trained generator.

    The output DataFrame has exactly the same column layout as
    raw_signals_real.csv:  ['concentration', 'I_0', 'I_1', ..., 'I_228']

    Parameters
    ----------
    G           : trained Generator
    hp          : hyperparameter dict
    device      : torch device
    n           : number of signals to generate (default: hp['n_generate'])
    conc_values : optional array of concentrations to condition on.
                  If None, samples log-uniformly from [conc_min, conc_max].
    """
    G.eval()
    n = n or hp["n_generate"]

    if conc_values is None:
        if hp["gen_conc_mode"] == "log_uniform":
            log_min     = np.log10(hp["conc_min"])
            log_max     = np.log10(hp["conc_max"])
            conc_values = 10 ** np.random.uniform(log_min, log_max, size=n)
        else:
            # Sample from the same discrete concentrations as real data
            df_real     = pd.read_csv(hp["real_csv"])
            real_concs  = df_real["concentration"].values
            conc_values = np.random.choice(real_concs, size=n, replace=True)

    c_norm = log10_conc_normalise(conc_values,
                                   hp["conc_min"],
                                   hp["conc_max"]).astype(np.float32)
    c_t    = torch.tensor(c_norm, dtype=torch.float32).unsqueeze(1).to(device)

    signals_list = []
    batch_size   = 64
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            z     = torch.randn(end - start, hp["latent_dim"], device=device)
            c_b   = c_t[start:end]
            x_gen = G(z, c_b).cpu().numpy()           # (batch, 229)
            signals_list.append(x_gen)

    X_gen   = np.vstack(signals_list)                  # (n, 229)
    I_cols  = [f"I_{i}" for i in range(hp["signal_len"])]
    df_out  = pd.DataFrame(X_gen, columns=I_cols)
    df_out.insert(0, "concentration", conc_values[:n])
    return df_out


# 6. SAVE / LOAD
def save_model(G: Generator, hp: dict) -> None:
    os.makedirs("/app/models", exist_ok=True)
    torch.save(G.state_dict(), "/app/models/wgangp_generator.pt")
    with open("/app/models/wgangp_config.json", "w") as f:
        # Convert tuples for JSON serialisation
        serialisable_hp = {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in hp.items()}
        json.dump(serialisable_hp, f, indent=2)
    print("  Saved: models/wgangp_generator.pt")
    print("  Saved: models/wgangp_config.json")


def load_model(config_path: str = "/app/models/wgangp_config.json",
               weights_path: str = "/app/models/wgangp_generator.pt") -> tuple[Generator, dict]:
    """Load a trained generator from disk."""
    with open(config_path) as f:
        hp = json.load(f)
    # Re-convert lists back to tuples where needed
    hp["betas"]        = tuple(hp["betas"])
    hp["gen_channels"] = list(hp["gen_channels"])
    hp["crit_channels"]= list(hp["crit_channels"])

    G = Generator(hp)
    G.load_state_dict(torch.load(weights_path, map_location="cpu"))
    G.eval()
    print(f"  Loaded generator from {weights_path}")
    return G, hp


# 7. ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="WGAN-GP Voltammogram Generator")
    parser.add_argument("--epochs",     type=int,  default=HP["epochs"])
    parser.add_argument("--batch-size", type=int,  default=HP["batch_size"])
    parser.add_argument("--latent-dim", type=int,  default=HP["latent_dim"])
    parser.add_argument("--n-generate", type=int,  default=HP["n_generate"])
    parser.add_argument("--train-on",   type=str,  default=HP["train_on"],
                        choices=["real", "combined"])
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, load saved model and generate")
    args = parser.parse_args()

    hp = dict(HP)
    hp["epochs"]    = args.epochs
    hp["batch_size"]= args.batch_size
    hp["latent_dim"]= args.latent_dim
    hp["n_generate"]= args.n_generate
    hp["train_on"]  = args.train_on

    print("=" * 60)
    print("  WGAN-GP — Pyocyanin Voltammogram Generator")
    print("=" * 60)

    device = get_device()

    if args.eval_only:
        print("\n  [Eval-only mode] Loading saved model …")
        G, hp = load_model()
        G = G.to(device)
    else:
        print(f"\n  Training on: {hp['train_on']} dataset")
        G, history = train(hp)
        save_model(G, hp)

        # Save loss history
        os.makedirs("/app/logs", exist_ok=True)
        pd.DataFrame(history).to_csv("/app/logs/wgangp_training_history.csv", index=False)
        print("  Saved: /app/logs/wgangp_training_history.csv")

    # Generate signals 
    print(f"\n  Generating {hp['n_generate']} synthetic signals …")
    df_gen = generate_signals(G, hp, device)

    os.makedirs("/app/samples", exist_ok=True)
    out_path = "/app/samples/wgangp_signals.csv"
    df_gen.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df_gen)} signals)")

    # Quick sanity check
    I_cols = [c for c in df_gen.columns if c.startswith("I_")]
    I_mat  = df_gen[I_cols].values
    print(f"\n  Sanity check:")
    print(f"    I[0] range   : [{I_mat[:, 0].min():.4f}, {I_mat[:, 0].max():.4f}]  (expect ≈ 0)")
    print(f"    I[228] range : [{I_mat[:,-1].min():.4f}, {I_mat[:,-1].max():.4f}]  (expect ≈ 0)")
    print(f"    I max overall: {I_mat.max():.4f} µA")
    print(f"    I min overall: {I_mat.min():.4f} µA")
    print(f"    conc range   : [{df_gen['concentration'].min():.3f}, "
          f"{df_gen['concentration'].max():.3f}] µM")
    print("\n  Done.")


if __name__ == "__main__":
    main()

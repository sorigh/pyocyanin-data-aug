"""
Conditional TimeGAN for Pyocyanin Voltammogram Generation

Why TimeGAN
------------
TimeGAN was chosen as the second architecture because it handles a major 
blind spot in standard GANs: time. 

1. **Catching the dynamics** — Instead of treating every data point as an 
   isolated event, TimeGAN maps raw signals into a latent space and trains 
   the GAN inside that space. For our voltammograms, this means it actually 
   learns the correlated rise and fall of the peak (the Faradaic process) 
   instead of just the global distribution.

2. **Keeping it grounded** — On top of the standard adversarial loss, TimeGAN 
   uses teacher-forcing to predict the next state. This prevents temporal 
   mode collapse, keeping the generated trajectories physically realistic 
   rather than just "smooth."

3. **Teaming up with WGAN-GP** — WGAN-GP is great at matching the marginal 
   distribution at individual points, while TimeGAN nails the joint temporal 
   structure. Together, they give us a really well-rounded evaluation picture.

4. **Built-in validation compatibility** — TimeGAN was actually the reference 
   architecture for Yoon et al.'s discriminative score metric (which sits 
   perfectly in Tier 4 of the validation gate). This means the generated 
   signals are being evaluated against their own design criteria.

Tweaks for This Domain
----------------------
- **Conditional generation**: We condition on the concentration (log10(c) ∈ [-1, 1]) 
  by concatenating the scalar to the GRU input at every single timestep.
- **Keeping it lean**: The sequence format is (B, 229, 1), where the single 
  feature is I(E). Since the potential (E) sits on a fixed, shared grid, it 
  doesn't carry any varying information between signals, so we leave it out.
- **Handling the small dataset**: With only 340 samples in the combined dataset, 
  I'm leaning on a high dropout rate (0.3) and weight decay to keep overfitting 
  in check.
- **Three-phase training**: Sticking to the original paper's approach: pre-training 
  the embedder (autoencoder) first, moving into adversarial training, and finishing 
  up with joint fine-tuning.

Usage
-----
    python timegan_training.py                    # Run the full pipeline
    python timegan_training.py --epochs 5000      # Set custom epochs per phase
    python timegan_training.py --eval-only        # Generate signals from a saved model

Output Files
------------
    models/timegan_embedder.pt    — encoder/decoder weights
    models/timegan_generator.pt   — temporal generator weights
    models/timegan_supervisor.pt  — supervisor network weights
    models/timegan_config.json    — saved hyperparameters
    samples/timegan_signals.csv   — generated signals (formatted to match the real data)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# 0. HYPERPARAMETERS
HP = {
    # Data
    "real_csv":       "raw/raw_signals_real.csv",
    "combined_csv":   "raw/raw_signals_combined.csv",
    "potential_csv":  "raw/raw_potential_grid.csv",
    "train_on":       "real",         # "real" | "combined"
    "signal_len":     229,                # sequence length T

    # Architecture
    "hidden_dim":     64,                 # GRU hidden size (H)
    "num_layers":     3,                  # GRU depth for embedder/supervisor/generator
    "cond_dim":       8,                  # concentration embedding size
    "noise_dim":      32,                 # noise input to generator (per timestep)
    "module":         "gru",              # "gru" | "lstm"

    # Training phases
    # Phase 1: Embedder pre-training (autoencoder)
    "phase1_epochs":  2000,
    # Phase 2: Supervised pre-training (supervisor)
    "phase2_epochs":  2000,
    # Phase 3: Joint GAN training
    "phase3_epochs":  4000,

    "batch_size":     32,
    "lr":             1e-3,               # single LR for all networks
    "weight_decay":   1e-4,

    # Loss weights (Phase 3) 
    "gamma":          1.0,                # weight on supervised loss in G update
    "eta":            10.0,               # weight on embedding reconstruction loss

    # Regularisation
    "dropout":        0.2,                # GRU dropout (applied between layers)

    # Logging 
    "log_every":      500,
    "seed":           42,

    # Generation
    "n_generate":     300,
    "gen_conc_mode":  "log_uniform",      # "log_uniform" | "real_classes"
    "conc_min":       0.1,
    "conc_max":       100.0,
}


# 1. UTILITIES
def set_seed(seed: int) -> None:
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
    """Map concentration to [-1, 1] via log10 scaling."""
    log_c   = np.log10(np.clip(c, 1e-9, None))
    log_min = np.log10(c_min)
    log_max = np.log10(c_max)
    return 2.0 * (log_c - log_min) / (log_max - log_min) - 1.0


def log10_conc_denormalise(c_norm: np.ndarray,
                            c_min: float = 0.1,
                            c_max: float = 100.0) -> np.ndarray:
    log_min = np.log10(c_min)
    log_max = np.log10(c_max)
    log_c   = (c_norm + 1.0) / 2.0 * (log_max - log_min) + log_min
    return 10.0 ** log_c


def normalise_signals(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalise signals to [0, 1] using global min/max.

    We use global (not per-signal) normalisation because the signal
    amplitude carries concentration information — per-signal normalisation
    would destroy the Ip ∝ c calibration.

    Returns (X_norm, X_min, X_max) for later denormalisation.
    """
    X_min = X.min()
    X_max = X.max()
    return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max


def denormalise_signals(X_norm: np.ndarray,
                         X_min:  float,
                         X_max:  float) -> np.ndarray:
    return X_norm * (X_max - X_min + 1e-8) + X_min


def load_data(hp: dict) -> tuple:
    csv_key = "combined_csv" if hp["train_on"] == "combined" else "real_csv"
    df  = pd.read_csv(hp[csv_key])
    E   = pd.read_csv(hp["potential_csv"])["potential_V"].values.astype(np.float64)

    I_cols = [c for c in df.columns if c.startswith("I_")]
    X      = df[I_cols].values.astype(np.float32)
    y_raw  = df["concentration"].values.astype(np.float32)

    assert X.shape[1] == hp["signal_len"]

    y_norm = log10_conc_normalise(y_raw, hp["conc_min"], hp["conc_max"]).astype(np.float32)

    # Normalise signals to [0, 1] for TimeGAN (it uses sigmoid outputs)
    X_norm, X_min, X_max = normalise_signals(X)

    print(f"  Loaded {len(X)} signals  |  signal len {X.shape[1]}")
    print(f"  Signal normalisation: [{X_min:.4f}, {X_max:.4f}] µA → [0, 1]")
    return X_norm, y_norm, E, X_min, X_max, y_raw


# 2. MODEL COMPONENTS
class RNNBlock(nn.Module):
    """
    Shared GRU (or LSTM) backbone used by all TimeGAN sub-networks.

    The `cond_input` flag determines whether the concentration conditioning
    scalar is concatenated to the input at every timestep.
    """
    def __init__(self,
                 input_size:  int,
                 hidden_dim:  int,
                 num_layers:  int,
                 dropout:     float,
                 module:      str = "gru",
                 cond_input:  bool = False,
                 cond_dim:    int = 0):
        super().__init__()
        self.cond_input = cond_input
        self.cond_dim   = cond_dim
        in_size = input_size + (cond_dim if cond_input else 0)

        rnn_cls = nn.LSTM if module == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size  = in_size,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

    def forward(self,
                x:     torch.Tensor,
                c_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        x     : (B, T, input_size)
        c_emb : (B, cond_dim) — conditioning, broadcast across timesteps
        returns: (B, T, hidden_dim)
        """
        if self.cond_input and c_emb is not None:
            c_seq = c_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, T, cond_dim)
            x = torch.cat([x, c_seq], dim=-1)

        out, _ = self.rnn(x)  # (B, T, hidden_dim)
        return out


class ConditionalEmbedding(nn.Module):
    """Maps scalar concentration to embedding vector."""
    def __init__(self, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.Tanh(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)   # (B, cond_dim)


class Embedder(nn.Module):
    """
    Encoder: maps real signals X (normalised) into latent space H.

    X (B, T, 1)  →  H (B, T, hidden_dim)

    The embedder learns a compact representation of the temporal dynamics.
    We use a sigmoid output to keep H in (0, 1), matching the generator.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.cond_emb = ConditionalEmbedding(hp["cond_dim"])
        self.rnn = RNNBlock(
            input_size = 1,
            hidden_dim = hp["hidden_dim"],
            num_layers = hp["num_layers"],
            dropout    = hp["dropout"],
            module     = hp["module"],
            cond_input = True,
            cond_dim   = hp["cond_dim"],
        )
        self.fc = nn.Linear(hp["hidden_dim"], hp["hidden_dim"])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1), c: (B, 1)  →  H: (B, T, hidden_dim)"""
        c_emb = self.cond_emb(c)
        h     = self.rnn(x, c_emb)
        return torch.sigmoid(self.fc(h))


class Recovery(nn.Module):
    """
    Decoder: reconstructs signals from latent space.

    H (B, T, hidden_dim)  →  X̂ (B, T, 1)

    Used only during pre-training to build a good embedding.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.cond_emb = ConditionalEmbedding(hp["cond_dim"])
        self.rnn = RNNBlock(
            input_size = hp["hidden_dim"],
            hidden_dim = hp["hidden_dim"],
            num_layers = hp["num_layers"],
            dropout    = hp["dropout"],
            module     = hp["module"],
            cond_input = True,
            cond_dim   = hp["cond_dim"],
        )
        self.fc = nn.Linear(hp["hidden_dim"], 1)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """H: (B, T, hidden_dim), c: (B, 1)  →  X̂: (B, T, 1)"""
        c_emb = self.cond_emb(c)
        out   = self.rnn(h, c_emb)
        return torch.sigmoid(self.fc(out))


class Supervisor(nn.Module):
    """
    Supervisor network: predicts next latent state given the current one.

    H_t  →  Ĥ_{t+1}

    The supervised loss ||H_{t+1} - Ĥ_{t+1}||² is the key innovation of
    TimeGAN: it enforces that the latent dynamics are temporally coherent,
    not just marginal-distribution-correct.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.cond_emb = ConditionalEmbedding(hp["cond_dim"])
        self.rnn = RNNBlock(
            input_size = hp["hidden_dim"],
            hidden_dim = hp["hidden_dim"],
            num_layers = max(hp["num_layers"] - 1, 1),   # one fewer layer
            dropout    = hp["dropout"],
            module     = hp["module"],
            cond_input = True,
            cond_dim   = hp["cond_dim"],
        )
        self.fc = nn.Linear(hp["hidden_dim"], hp["hidden_dim"])

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """H: (B, T, hidden_dim)  →  Ĥ: (B, T, hidden_dim)"""
        c_emb = self.cond_emb(c)
        out   = self.rnn(h, c_emb)
        return torch.sigmoid(self.fc(out))


class TemporalGenerator(nn.Module):
    """
    Generator: maps noise Z (B, T, noise_dim) → latent H̃ (B, T, hidden_dim).

    The generator operates in the latent space defined by the embedder.
    This is the key architectural difference from a standard GAN:
    the adversarial game is played in a learned temporal embedding space,
    not in the raw signal space, making training much more stable.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.hp       = hp
        self.cond_emb = ConditionalEmbedding(hp["cond_dim"])
        self.rnn = RNNBlock(
            input_size = hp["noise_dim"],
            hidden_dim = hp["hidden_dim"],
            num_layers = hp["num_layers"],
            dropout    = hp["dropout"],
            module     = hp["module"],
            cond_input = True,
            cond_dim   = hp["cond_dim"],
        )
        self.fc = nn.Linear(hp["hidden_dim"], hp["hidden_dim"])

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Z: (B, T, noise_dim), c: (B, 1)  →  H̃: (B, T, hidden_dim)"""
        c_emb = self.cond_emb(c)
        out   = self.rnn(z, c_emb)
        return torch.sigmoid(self.fc(out))


class Discriminator(nn.Module):
    """
    Temporal discriminator: classifies sequences as real or synthetic.

    H (B, T, hidden_dim)  →  Y (B, T, 1)

    The discriminator operates on the latent sequences H (encoded real)
    and H̃ (generated), not on raw signals.  This avoids the mode-collapse
    problem of raw-signal discriminators on short sequences.
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.cond_emb = ConditionalEmbedding(hp["cond_dim"])
        self.rnn = RNNBlock(
            input_size = hp["hidden_dim"],
            hidden_dim = hp["hidden_dim"],
            num_layers = max(hp["num_layers"] - 1, 1),
            dropout    = hp["dropout"],
            module     = hp["module"],
            cond_input = True,
            cond_dim   = hp["cond_dim"],
        )
        self.fc = nn.Linear(hp["hidden_dim"], 1)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """H: (B, T, hidden_dim)  →  Y: (B, T, 1)"""
        c_emb = self.cond_emb(c)
        out   = self.rnn(h, c_emb)
        return self.fc(out)



# 3. THREE-PHASE TRAINING
def train_phase1_embedder(embedder:  Embedder,
                           recovery:  Recovery,
                           loader:    DataLoader,
                           epochs:    int,
                           lr:        float,
                           device:    torch.device,
                           log_every: int,
                           weight_decay: float = 1e-4) -> list[dict]:
    """
    Phase 1: Pre-train the Embedder + Recovery as an autoencoder.

    Loss: MSE(X, R(E(X))) , standard reconstruction.

    This phase builds a useful latent space before any adversarial
    training begins.  Without it the generator has no stable target
    to match.
    """
    params = list(embedder.parameters()) + list(recovery.parameters())
    opt    = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/10)
    mse    = nn.MSELoss()
    history = []

    print(f"\n  Phase 1:  Embedder pre-training ({epochs} epochs) …")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        embedder.train(); recovery.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x_seq, c_seq in loader:
            x_seq = x_seq.to(device)   # (B, T, 1)
            c_seq = c_seq.to(device)   # (B, 1)

            h   = embedder(x_seq, c_seq)   # (B, T, H)
            x_r = recovery(h, c_seq)       # (B, T, 1)

            loss = mse(x_r, x_seq)
            opt.zero_grad(); loss.backward(); opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg = epoch_loss / n_batches
        history.append({"epoch": epoch, "phase": 1, "recon_loss": avg})

        if epoch % log_every == 0:
            print(f"    Epoch {epoch:5d}/{epochs} | Recon loss: {avg:.6f} | "
                  f"Time: {time.time()-t0:.0f}s")

    print(f"  Phase 1 done in {time.time()-t0:.1f}s")
    return history


def train_phase2_supervisor(embedder:   Embedder,
                             supervisor: Supervisor,
                             loader:     DataLoader,
                             epochs:     int,
                             lr:         float,
                             device:     torch.device,
                             log_every:  int,
                             weight_decay: float = 1e-4) -> list[dict]:
    """
    Phase 2: Pre-train the Supervisor on real embeddings.

    Loss: MSE(H_{t+1}, S(H_t))  — next-step prediction in latent space.

    This gives the supervisor a head-start so that in Phase 3 the
    generator receives meaningful gradient signal from the very first epoch.
    """
    # Freeze embedder — we don't want to disturb the good latent space
    for p in embedder.parameters():
        p.requires_grad = False

    opt   = torch.optim.Adam(supervisor.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/10)
    mse   = nn.MSELoss()
    history = []

    print(f"\n  Phase 2:  Supervisor pre-training ({epochs} epochs) …")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        supervisor.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x_seq, c_seq in loader:
            x_seq = x_seq.to(device)
            c_seq = c_seq.to(device)

            with torch.no_grad():
                h = embedder(x_seq, c_seq)   # (B, T, H)

            h_hat = supervisor(h, c_seq)     # (B, T, H)

            # Predict H[1:] from H[:-1]
            loss = mse(h_hat[:, :-1, :], h[:, 1:, :])

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg = epoch_loss / n_batches
        history.append({"epoch": epoch, "phase": 2, "sup_loss": avg})

        if epoch % log_every == 0:
            print(f"    Epoch {epoch:5d}/{epochs} | Sup loss: {avg:.6f} | "
                  f"Time: {time.time()-t0:.0f}s")

    # Unfreeze embedder for Phase 3
    for p in embedder.parameters():
        p.requires_grad = True

    print(f"  Phase 2 done in {time.time()-t0:.1f}s")
    return history


def train_phase3_joint(embedder:   Embedder,
                        recovery:   Recovery,
                        supervisor: Supervisor,
                        generator:  TemporalGenerator,
                        discriminator: Discriminator,
                        loader:     DataLoader,
                        hp:         dict,
                        device:     torch.device) -> list[dict]:
    """
    Phase 3: Joint adversarial training.

    Four losses are optimised alternately:

    1. Discriminator loss (D_loss):
       BCE on real latent H vs generated latent H̃

    2. Generator loss — unsupervised (G_u_loss):
       BCE adversarial loss (fool discriminator)

    3. Generator loss — supervised (G_s_loss):
       MSE(S(H̃)_t, H̃_{t+1})  — temporal coherence

    4. Embedder + Recovery joint loss (E_loss):
       η * MSE(X, R(S(E(X)))) + MSE(E(X)_moments, G̃_moments)
       The moment-matching term aligns the statistical moments of real
       and generated latent sequences, preventing the generator from
       learning a latent space that looks nothing like the real one.
    """
    epochs    = hp["phase3_epochs"]
    log_every = hp["log_every"]
    gamma     = hp["gamma"]
    eta       = hp["eta"]
    lr        = hp["lr"]
    wd        = hp["weight_decay"]

    opt_G = torch.optim.Adam(list(generator.parameters()) +
                              list(supervisor.parameters()),
                              lr=lr, weight_decay=wd)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=wd)
    opt_E = torch.optim.Adam(list(embedder.parameters()) +
                              list(recovery.parameters()),
                              lr=lr, weight_decay=wd)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history = []

    print(f"\n  Phase 3:  Joint adversarial training ({epochs} epochs) …")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        embedder.train(); recovery.train()
        supervisor.train(); generator.train(); discriminator.train()

        epoch_d = 0.0; epoch_g = 0.0; epoch_e = 0.0
        n_batches = 0

        for x_seq, c_seq in loader:
            x_seq = x_seq.to(device)   # (B, T, 1)
            c_seq = c_seq.to(device)   # (B, 1)
            B, T, _ = x_seq.shape

            # Generate fake latent sequence 
            z     = torch.randn(B, T, hp["noise_dim"], device=device)
            h_fake = generator(z, c_seq)        # (B, T, H)
            h_fake_sup = supervisor(h_fake, c_seq)  # (B, T, H) — after supervisor

            # Encode real signal 
            h_real = embedder(x_seq, c_seq)     # (B, T, H)
            h_real_sup = supervisor(h_real, c_seq)  # (B, T, H)


            # 3a: Discriminator update
            
            y_real = discriminator(h_real.detach(), c_seq)        # (B, T, 1)
            y_fake = discriminator(h_fake_sup.detach(), c_seq)    # (B, T, 1)

            d_loss_real = bce(y_real, torch.ones_like(y_real))
            d_loss_fake = bce(y_fake, torch.zeros_like(y_fake))
            d_loss      = d_loss_real + d_loss_fake

            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

           
            # 3b: Generator update (adversarial + supervised)
            # Regenerate to get fresh graph
            z      = torch.randn(B, T, hp["noise_dim"], device=device)
            h_fake = generator(z, c_seq)
            h_fake_sup = supervisor(h_fake, c_seq)

            # Adversarial loss — fool discriminator
            y_fake_g = discriminator(h_fake_sup, c_seq)
            g_loss_u = bce(y_fake_g, torch.ones_like(y_fake_g))

            # Supervised loss — temporal coherence
            g_loss_s = mse(h_fake_sup[:, :-1, :], h_fake[:, 1:, :])

            # Moment matching — mean and variance of latent sequences
            g_loss_v = (
                torch.mean(torch.abs(h_fake.mean(dim=0) - h_real.detach().mean(dim=0))) +
                torch.mean(torch.abs(h_fake.var(dim=0) - h_real.detach().var(dim=0)))
            )

            g_loss = g_loss_u + gamma * g_loss_s + g_loss_v

            opt_G.zero_grad(); g_loss.backward(); opt_G.step()


            # 3c: Embedder + Recovery update
            h_emb  = embedder(x_seq, c_seq)
            h_sup  = supervisor(h_emb, c_seq)
            x_rec  = recovery(h_sup, c_seq)

            # Reconstruction loss
            e_loss_rec = mse(x_rec, x_seq)

            # Supervised loss on real embeddings
            e_loss_sup = mse(h_sup[:, :-1, :], h_emb[:, 1:, :])

            e_loss = eta * torch.sqrt(e_loss_rec) + e_loss_sup

            opt_E.zero_grad(); e_loss.backward(); opt_E.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()
            epoch_e += e_loss.item()
            n_batches += 1

        avg_d = epoch_d / n_batches
        avg_g = epoch_g / n_batches
        avg_e = epoch_e / n_batches
        history.append({"epoch": epoch, "phase": 3,
                         "d_loss": avg_d, "g_loss": avg_g, "e_loss": avg_e})

        if epoch % log_every == 0:
            print(f"    Epoch {epoch:5d}/{epochs} | "
                  f"D: {avg_d:.4f}  G: {avg_g:.4f}  E: {avg_e:.4f} | "
                  f"Time: {time.time()-t0:.0f}s")

    print(f"  Phase 3 done in {time.time()-t0:.1f}s")
    return history


def train(hp: dict):
    set_seed(hp["seed"])
    device = get_device()

    # Data 
    X_norm, y_norm, E, X_min, X_max, y_raw = load_data(hp)

    # TimeGAN expects (B, T, 1) format
    X_seq  = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(2)  # (N, T, 1)
    y_t    = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(X_seq, y_t)
    loader  = DataLoader(dataset,
                         batch_size=hp["batch_size"],
                         shuffle=True,
                         drop_last=True)

    # Instantiate all networks 
    embedder      = Embedder(hp).to(device)
    recovery      = Recovery(hp).to(device)
    supervisor    = Supervisor(hp).to(device)
    generator     = TemporalGenerator(hp).to(device)
    discriminator = Discriminator(hp).to(device)

    n_params = {
        "embedder":      sum(p.numel() for p in embedder.parameters()),
        "recovery":      sum(p.numel() for p in recovery.parameters()),
        "supervisor":    sum(p.numel() for p in supervisor.parameters()),
        "generator":     sum(p.numel() for p in generator.parameters()),
        "discriminator": sum(p.numel() for p in discriminator.parameters()),
    }
    total = sum(n_params.values())
    print(f"\n  Network parameter counts:")
    for name, n in n_params.items():
        print(f"    {name:<15}: {n:,}")
    print(f"    {'TOTAL':<15}: {total:,}")

    # Three-phase training 
    h1 = train_phase1_embedder(embedder, recovery, loader,
                                hp["phase1_epochs"], hp["lr"],
                                device, hp["log_every"], hp["weight_decay"])

    h2 = train_phase2_supervisor(embedder, supervisor, loader,
                                  hp["phase2_epochs"], hp["lr"],
                                  device, hp["log_every"], hp["weight_decay"])

    h3 = train_phase3_joint(embedder, recovery, supervisor, generator,
                             discriminator, loader, hp, device)

    history = h1 + h2 + h3

    return (embedder, recovery, supervisor, generator, discriminator,
            X_min, X_max, history)


# 4. SIGNAL GENERATION
def generate_signals(generator:  TemporalGenerator,
                     recovery:   Recovery,
                     hp:         dict,
                     device:     torch.device,
                     X_min:      float,
                     X_max:      float,
                     n:          int | None = None,
                     conc_values: np.ndarray | None = None) -> pd.DataFrame:
    """
    Generate synthetic voltammogram signals.

    Pipeline:
        Z (noise) + c (cond) → G → H̃ (fake latent) → R → X̃ (signal, normalised)
        → denormalise → clip boundary → DataFrame

    Output format is identical to raw_signals_real.csv.
    """
    generator.eval(); recovery.eval()
    n = n or hp["n_generate"]

    if conc_values is None:
        if hp["gen_conc_mode"] == "log_uniform":
            log_min     = np.log10(hp["conc_min"])
            log_max     = np.log10(hp["conc_max"])
            conc_values = 10 ** np.random.uniform(log_min, log_max, size=n)
        else:
            df_real     = pd.read_csv(hp["real_csv"])
            real_concs  = df_real["concentration"].values
            conc_values = np.random.choice(real_concs, size=n, replace=True)

    c_norm = log10_conc_normalise(conc_values, hp["conc_min"], hp["conc_max"]).astype(np.float32)
    c_t    = torch.tensor(c_norm, dtype=torch.float32).unsqueeze(1).to(device)

    T = hp["signal_len"]
    signals_list = []
    batch_size   = 32

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            B    = end - start
            z    = torch.randn(B, T, hp["noise_dim"], device=device)
            c_b  = c_t[start:end]
            h    = generator(z, c_b)                  # (B, T, H)
            x_n  = recovery(h, c_b).squeeze(2)        # (B, T)
            x_n  = x_n.clamp(0.0, 1.0).cpu().numpy()
            signals_list.append(x_n)

    X_gen_norm = np.vstack(signals_list)               # (n, T)

    # Denormalise to µA scale
    X_gen = denormalise_signals(X_gen_norm, X_min, X_max)

    # Enforce boundary condition I[0] = I[228] = 0
    X_gen[:, 0]  = 0.0
    X_gen[:, -1] = 0.0

    I_cols = [f"I_{i}" for i in range(hp["signal_len"])]
    df_out = pd.DataFrame(X_gen.astype(np.float32), columns=I_cols)
    df_out.insert(0, "concentration", conc_values[:n])
    return df_out


# 5. SAVE / LOAD
def save_model(embedder, recovery, supervisor, generator, discriminator,
               X_min, X_max, hp):
    os.makedirs("/app/models", exist_ok=True)
    torch.save(embedder.state_dict(),      "/app/models/timegan_embedder.pt")
    torch.save(recovery.state_dict(),      "/app/models/timegan_recovery.pt")
    torch.save(supervisor.state_dict(),    "/app/models/timegan_supervisor.pt")
    torch.save(generator.state_dict(),     "/app/models/timegan_generator.pt")
    torch.save(discriminator.state_dict(), "/app/models/timegan_discriminator.pt")

    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    cfg["X_min"] = float(X_min)
    cfg["X_max"] = float(X_max)
    with open("/app/models/timegan_config.json", "w") as f: 
        json.dump(cfg, f, indent=2)

    print("  Saved: models/timegan_{embedder,recovery,supervisor,generator,discriminator}.pt")
    print("  Saved: models/timegan_config.json")


def load_model(config_path: str = "/app/models/timegan_config.json"):
    with open(config_path) as f:
        hp = json.load(f)
    X_min = hp.pop("X_min")
    X_max = hp.pop("X_max")

    generator  = TemporalGenerator(hp)
    recovery   = Recovery(hp)
    embedder   = Embedder(hp)
    supervisor = Supervisor(hp)
    discriminator = Discriminator(hp)

    generator.load_state_dict(    torch.load("/app/models/timegan_generator.pt",     map_location="cpu"))
    recovery.load_state_dict(     torch.load("/app/models/timegan_recovery.pt",      map_location="cpu"))
    embedder.load_state_dict(     torch.load("/app/models/timegan_embedder.pt",      map_location="cpu"))
    supervisor.load_state_dict(   torch.load("/app/models/timegan_supervisor.pt",    map_location="cpu"))
    discriminator.load_state_dict(torch.load("/app/models/timegan_discriminator.pt", map_location="cpu"))

    generator.eval(); recovery.eval()
    print(f"  Loaded TimeGAN models from /app/models/")
    return generator, recovery, embedder, supervisor, discriminator, X_min, X_max, hp


# 6. ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="TimeGAN Voltammogram Generator")
    parser.add_argument("--p1-epochs",  type=int, default=HP["phase1_epochs"])
    parser.add_argument("--p2-epochs",  type=int, default=HP["phase2_epochs"])
    parser.add_argument("--p3-epochs",  type=int, default=HP["phase3_epochs"])
    parser.add_argument("--batch-size", type=int, default=HP["batch_size"])
    parser.add_argument("--n-generate", type=int, default=HP["n_generate"])
    parser.add_argument("--train-on",   type=str, default=HP["train_on"],
                        choices=["real", "combined"])
    parser.add_argument("--eval-only",  action="store_true")
    args = parser.parse_args()

    hp = dict(HP)
    hp["phase1_epochs"] = args.p1_epochs
    hp["phase2_epochs"] = args.p2_epochs
    hp["phase3_epochs"] = args.p3_epochs
    hp["batch_size"]    = args.batch_size
    hp["n_generate"]    = args.n_generate
    hp["train_on"]      = args.train_on

    print("=" * 60)
    print("  TimeGAN — Pyocyanin Voltammogram Generator")
    print("=" * 60)

    device = get_device()

    if args.eval_only:
        print("\n  [Eval-only mode] Loading saved models …")
        generator, recovery, _, _, _, X_min, X_max, hp = load_model()
        generator = generator.to(device)
        recovery  = recovery.to(device)
    else:
        print(f"\n  Training on: {hp['train_on']} dataset")
        (embedder, recovery, supervisor, generator, discriminator,
         X_min, X_max, history) = train(hp)

        save_model(embedder, recovery, supervisor, generator, discriminator,
                   X_min, X_max, hp)

        os.makedirs("logs", exist_ok=True)
        pd.DataFrame(history).to_csv("logs/timegan_training_history.csv", index=False)
        print("  Saved: logs/timegan_training_history.csv")

        generator = generator.to(device)
        recovery  = recovery.to(device)

    print(f"\n  Generating {hp['n_generate']} synthetic signals …")
    df_gen = generate_signals(generator, recovery, hp, device, X_min, X_max)

    os.makedirs("samples", exist_ok=True)
    out_path = "samples/timegan_signals.csv"
    df_gen.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df_gen)} signals)")

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

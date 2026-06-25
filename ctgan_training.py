import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# HYPERPARAMETERS 
HP = {
    # Data 
    "real_csv":      "raw/raw_signals_real.csv",
    "combined_csv":  "raw/raw_signals_combined.csv",
    "train_on":      "combined",    # "real" | "combined"
    "signal_len":    229,

    # CTGAN hyperparameters
    # embedding_dim  : size of the random sample passed to the generator
    "embedding_dim":        128,
    # generator_dim  : layer sizes of the generator MLP
    "generator_dim":        (256, 256),
    # discriminator_dim : layer sizes of the discriminator MLP
    "discriminator_dim":    (256, 256),
    "generator_lr":         2e-4,
    "discriminator_lr":     2e-4,
    "generator_decay":      1e-6,
    "discriminator_decay":  1e-6,
    "batch_size":           500,
    # discriminator_steps : critic updates per generator step (like n_critic)
    "discriminator_steps":  1,
    "epochs":               300,
    # pac : number of samples used per discriminator step (PacGAN trick)
    "pac":                  10,
    "cuda":                 True,   # falls back to CPU if unavailable

    # Generation 
    "n_generate":        300,
    "gen_conc_mode":     "log_uniform",  # "log_uniform" | "real_classes"
    "conc_min":          0.1,
    "conc_max":          100.0,
    "seed":              42,
}


# DATA
def load_training_data(hp: dict) -> pd.DataFrame:
    csv = hp["combined_csv"] if hp["train_on"] == "combined" else hp["real_csv"]
    df  = pd.read_csv(csv)
    assert df.shape[1] == hp["signal_len"] + 1, (
        f"Expected {hp['signal_len'] + 1} columns, got {df.shape[1]}")
    print(f"  Loaded {len(df)} signals from {csv}")
    return df


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """
    Build SDV metadata.  All columns are numerical; concentration gets no
    special treatment - CTGAN's conditional layer handles the class structure.
    """
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    # Confirm everything is numerical (SDV may mis-detect some I_ columns)
    for col in df.columns:
        meta.update_column(col, sdtype="numerical")
    return meta


# TRAIN

def train(hp: dict) -> CTGANSynthesizer:
    np.random.seed(hp["seed"])
    df   = load_training_data(hp)
    meta = build_metadata(df)

    synth = CTGANSynthesizer(
        metadata           = meta,
        embedding_dim      = hp["embedding_dim"],
        generator_dim      = hp["generator_dim"],
        discriminator_dim  = hp["discriminator_dim"],
        generator_lr       = hp["generator_lr"],
        discriminator_lr   = hp["discriminator_lr"],
        generator_decay    = hp["generator_decay"],
        discriminator_decay= hp["discriminator_decay"],
        batch_size         = hp["batch_size"],
        discriminator_steps= hp["discriminator_steps"],
        epochs             = hp["epochs"],
        pac                = hp["pac"],
        cuda               = hp["cuda"],
        verbose            = True,
    )

    print(f"\n  Training CTGAN for {hp['epochs']} epochs …\n")
    synth.fit(df)
    print("\n  Training complete.")
    return synth


# GENERATE
def generate_signals(synth: CTGANSynthesizer, hp: dict) -> pd.DataFrame:
    """
    Generate synthetic signals and post-process to match the real CSV format.

    Post-processing steps:
      1. Clip I values to [-1, 40] µA (physical range)
      2. Enforce I[0] = I[228] = 0 (boundary condition from real signals)
      3. Clip concentration to [conc_min, conc_max]
    """
    n = hp["n_generate"]

    # SDV CTGAN samples from the learned joint distribution - concentration
    # is sampled from what it saw during training.  To get a specific
    # log-uniform spread we use conditional_sampling via a DataFrame of
    # known conditions (SDV 1.x supports this via sample_remaining_columns).
    if hp["gen_conc_mode"] == "log_uniform":
        log_min = np.log10(hp["conc_min"])
        log_max = np.log10(hp["conc_max"])
        target_concs = 10 ** np.random.uniform(log_min, log_max, n)
    else:
        df_real      = pd.read_csv(hp["real_csv"])
        target_concs = np.random.choice(df_real["concentration"].values, n, replace=True)

    # Build a condition DataFrame with only the concentration column set;
    # CTGAN fills in the remaining 229 I_ columns.
    known = pd.DataFrame({"concentration": target_concs})
    df_gen = synth.sample_remaining_columns(known_columns=known, max_tries_per_batch=500)

    # Post-processing 
    I_cols = [f"I_{i}" for i in range(hp["signal_len"])]

    # Keep only the columns we need, in the right order
    df_gen = df_gen[["concentration"] + I_cols].copy()

    # Physical range clip
    df_gen[I_cols] = df_gen[I_cols].clip(lower=-1.0, upper=40.0)

    # Boundary condition
    df_gen["I_0"]   = 0.0
    df_gen["I_228"] = 0.0

    # Concentration clip
    df_gen["concentration"] = df_gen["concentration"].clip(
        lower=hp["conc_min"], upper=hp["conc_max"])

    print(f"  Generated {len(df_gen)} signals.")
    return df_gen



# SAVE / LOAD
def save_model(synth: CTGANSynthesizer) -> None:
    os.makedirs("models", exist_ok=True)
    with open("models/ctgan_synthesizer.pkl", "wb") as f:
        pickle.dump(synth, f)
    print("  Saved: models/ctgan_synthesizer.pkl")


def load_model(path: str = "models/ctgan_synthesizer.pkl") -> CTGANSynthesizer:
    with open(path, "rb") as f:
        synth = pickle.load(f)
    print(f"  Loaded synthesizer from {path}")
    return synth


# ENTRY POINT

def main():
    parser = argparse.ArgumentParser(description="CTGAN Voltammogram Generator (SDV)")
    parser.add_argument("--epochs",     type=int, default=HP["epochs"])
    parser.add_argument("--batch-size", type=int, default=HP["batch_size"])
    parser.add_argument("--n-generate", type=int, default=HP["n_generate"])
    parser.add_argument("--train-on",   type=str, default=HP["train_on"],
                        choices=["real", "combined"])
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, load saved model and generate")
    args = parser.parse_args()

    hp = dict(HP)
    hp["epochs"]     = args.epochs
    hp["batch_size"] = args.batch_size
    hp["n_generate"] = args.n_generate
    hp["train_on"]   = args.train_on

    print("=" * 55)
    print("  CTGAN - Pyocyanin Voltammogram Generator (SDV)")
    print("=" * 55)

    if args.eval_only:
        synth = load_model()
    else:
        synth = train(hp)
        save_model(synth)

    print(f"\n  Generating {hp['n_generate']} signals …")
    df_gen = generate_signals(synth, hp)

    os.makedirs("samples", exist_ok=True)
    out_path = "samples/ctgan_signals.csv"
    df_gen.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Sanity check
    I_cols = [f"I_{i}" for i in range(hp["signal_len"])]
    I_mat  = df_gen[I_cols].values
    print(f"\n  Sanity check:")
    print(f"    I[0] range   : [{I_mat[:,0].min():.4f}, {I_mat[:,0].max():.4f}]  (expect 0)")
    print(f"    I[228] range : [{I_mat[:,-1].min():.4f}, {I_mat[:,-1].max():.4f}]  (expect 0)")
    print(f"    I max overall: {I_mat.max():.4f} µA")
    print(f"    conc range   : [{df_gen['concentration'].min():.3f}, "
          f"{df_gen['concentration'].max():.3f}] µM")
    print("\n  Done.")


if __name__ == "__main__":
    main()

"""
Usage
-----
    python timegan_training_sdv.py                # train and generate
    python timegan_training_sdv.py --epochs 500   # quick test
    python timegan_training_sdv.py --eval-only    # load and generate

Output
------
    models/timegan_sdv_synthesizer.pkl   saved PARSynthesizer
    samples/timegan_sdv_signals.csv      generated signals (same format as real)
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata

# HYPERPARAMETERS
HP = {
    # Data 
    "real_csv":      "raw/raw_signals_real.csv",
    "combined_csv":  "raw/raw_signals_combined.csv",
    "train_on":      "combined",    # "real" | "combined"
    "signal_len":    229,

    # PAR hyperparameters 
    # epochs : training epochs for the PAR RNN
    "epochs":        512,
    # sample_size : number of generated samples per real sequence used
    #               during training (data augmentation inside PAR)
    "sample_size":   1,
    "cuda":          True,

    # Generation 
    "n_generate":    300,
    "gen_conc_mode": "log_uniform",  # "log_uniform" | "real_classes"
    "conc_min":      0.1,
    "conc_max":      100.0,
    "seed":          42,
}


# DATA HELPERS

def to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape (N, 230) wide → (N×229, 4) long format required by PARSynthesizer.

    Columns: signal_id | timestep | concentration | current
    signal_id is an integer unique per signal.
    timestep  is the index into the potential grid (0 … 228).
    """
    I_cols = [c for c in df_wide.columns if c.startswith("I_")]
    rows   = []
    for signal_id, row in enumerate(df_wide.itertuples(index=False)):
        conc = row.concentration
        for t, col in enumerate(I_cols):
            rows.append({
                "signal_id":     signal_id,
                "timestep":      t,
                "concentration": conc,
                "current":       getattr(row, col),
            })
    return pd.DataFrame(rows)


def to_wide_format(df_long: pd.DataFrame, signal_len: int = 229) -> pd.DataFrame:
    """
    Reshape (N×229, 4) long → (N, 230) wide, matching raw_signals_real.csv.

    Uses pivot; handles the case where PAR assigns random integer signal_ids.
    """
    wide_list = []
    for sig_id, group in df_long.groupby("signal_id"):
        group = group.sort_values("timestep")
        if len(group) != signal_len:
            continue   # skip incomplete sequences
        conc    = group["concentration"].iloc[0]
        current = group["current"].values.astype(np.float32)
        row     = {"concentration": conc}
        row.update({f"I_{t}": float(current[t]) for t in range(signal_len)})
        wide_list.append(row)

    df_wide    = pd.DataFrame(wide_list)
    col_order  = ["concentration"] + [f"I_{i}" for i in range(signal_len)]
    return df_wide[col_order]


def build_metadata(df_long: pd.DataFrame) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df_long)
    meta.update_column("signal_id", sdtype="id")
    meta.set_sequence_key("signal_id")
    meta.set_sequence_index("timestep")
    return meta


# TRAIN
def train(hp: dict) -> PARSynthesizer:
    np.random.seed(hp["seed"])
    csv     = hp["combined_csv"] if hp["train_on"] == "combined" else hp["real_csv"]
    df_wide = pd.read_csv(csv)
    print(f"  Loaded {len(df_wide)} signals from {csv}")

    print("  Reshaping to long format …")
    df_long = to_long_format(df_wide)
    print(f"  Long format: {df_long.shape}  ({len(df_wide)} signals × 229 timesteps)")

    meta  = build_metadata(df_long)
    synth = PARSynthesizer(
        metadata         = meta,
        context_columns  = ["concentration"],
        epochs           = hp["epochs"],
        sample_size      = hp["sample_size"],
        cuda             = hp["cuda"],
        verbose          = True,
    )

    print(f"\n  Training PARSynthesizer for {hp['epochs']} epochs …\n")
    synth.fit(df_long)
    print("\n  Training complete.")
    return synth

# GENERATE
def generate_signals(synth: PARSynthesizer, hp: dict) -> pd.DataFrame:
    """
    Generate signals and convert back to wide format.

    PARSynthesizer.sample() draws concentrations from the training
    distribution.  We post-hoc reassign concentrations to the requested
    target values (log-uniform or real classes) so the output covers the
    full calibration range, matching the physics-augmented CSV layout.
    """
    n = hp["n_generate"]

    # Target concentrations
    if hp["gen_conc_mode"] == "log_uniform":
        log_min = np.log10(hp["conc_min"])
        log_max = np.log10(hp["conc_max"])
        target_concs = 10 ** np.random.uniform(log_min, log_max, n)
    else:
        df_real      = pd.read_csv(hp["real_csv"])
        target_concs = np.random.choice(df_real["concentration"].values, n, replace=True)

    print(f"  Sampling {n} sequences from PARSynthesizer …")
    df_long_gen = synth.sample(num_sequences=n)

    # Convert to wide
    df_wide_gen = to_wide_format(df_long_gen, hp["signal_len"])

    # If PAR generated fewer sequences than requested (rare), pad with repeats
    if len(df_wide_gen) < n:
        print(f"  Warning: PAR returned {len(df_wide_gen)} sequences, requested {n}. "
              f"Padding by repeating.")
        extra = n - len(df_wide_gen)
        df_wide_gen = pd.concat(
            [df_wide_gen, df_wide_gen.sample(extra, replace=True)], ignore_index=True)

    df_wide_gen = df_wide_gen.iloc[:n].copy()

    # Overwrite concentration with the target distribution
    df_wide_gen["concentration"] = target_concs[:len(df_wide_gen)]

    # Post-processing 
    I_cols = [f"I_{i}" for i in range(hp["signal_len"])]
    df_wide_gen[I_cols] = df_wide_gen[I_cols].clip(lower=-1.0, upper=40.0)
    df_wide_gen["I_0"]   = 0.0
    df_wide_gen["I_228"] = 0.0
    df_wide_gen["concentration"] = df_wide_gen["concentration"].clip(
        lower=hp["conc_min"], upper=hp["conc_max"])

    print(f"  Generated {len(df_wide_gen)} signals.")
    return df_wide_gen

# SAVE / LOAD
def save_model(synth: PARSynthesizer) -> None:
    os.makedirs("models", exist_ok=True)
    with open("models/timegan_sdv_synthesizer.pkl", "wb") as f:
        pickle.dump(synth, f)
    print("  Saved: models/timegan_sdv_synthesizer.pkl")


def load_model(path: str = "models/timegan_sdv_synthesizer.pkl") -> PARSynthesizer:
    with open(path, "rb") as f:
        synth = pickle.load(f)
    print(f"  Loaded synthesizer from {path}")
    return synth


# ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="PAR/TimeGAN Voltammogram Generator (SDV)")
    parser.add_argument("--epochs",     type=int, default=HP["epochs"])
    parser.add_argument("--n-generate", type=int, default=HP["n_generate"])
    parser.add_argument("--train-on",   type=str, default=HP["train_on"],
                        choices=["real", "combined"])
    parser.add_argument("--eval-only",  action="store_true")
    args = parser.parse_args()

    hp = dict(HP)
    hp["epochs"]     = args.epochs
    hp["n_generate"] = args.n_generate
    hp["train_on"]   = args.train_on

    print("=" * 58)
    print("  PAR / TimeGAN — Pyocyanin Voltammogram Generator (SDV)")
    print("=" * 58)

    if args.eval_only:
        synth = load_model()
    else:
        synth = train(hp)
        save_model(synth)

    print(f"\n  Generating {hp['n_generate']} signals …")
    df_gen = generate_signals(synth, hp)

    os.makedirs("samples", exist_ok=True)
    out_path = "samples/timegan_sdv_signals.csv"
    df_gen.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

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

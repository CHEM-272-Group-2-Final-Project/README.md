import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

box_half = 100.0

files = sorted(
    glob.glob("particles_step_*.csv"),
    key=lambda s: int(s.split("_")[-1].split(".")[0])
)

for fname in files:
    df = pd.read_csv(fname)

    # ensure numeric dtypes
    for col in ["x", "y", "mass", "step", "T", "a", "b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    sizes = df["mass"].to_numpy(dtype=float) * 10.0

    step = int(df["step"].iloc[0])
    T = float(df["T"].iloc[0])
    a = float(df["a"].iloc[0])
    b = float(df["b"].iloc[0])

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=sizes, alpha=0.4)
    plt.xlim(-box_half, box_half)
    plt.ylim(-box_half, box_half)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Iteration {step}  T={T:.3f}  a={a}  b={b}")
    plt.tight_layout()
    plt.show()

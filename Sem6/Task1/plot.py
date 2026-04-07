from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


TRAJ_RE = re.compile(r"trajectory_(.+)\.csv$")
STAB_RE = re.compile(r"stability_real_(.+)\.csv$")


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_trajectory_file(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    name = csv_path.stem

    # y1(t)
    plt.figure(figsize=(9, 5))
    plt.plot(df["t"], df["y1"])
    plt.xlabel("t")
    plt.ylabel("y1")
    plt.title(f"{name}: y1(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_y1.png", dpi=150)
    plt.close()

    # y2(t)
    plt.figure(figsize=(9, 5))
    plt.plot(df["t"], df["y2"])
    plt.xlabel("t")
    plt.ylabel("y2")
    plt.title(f"{name}: y2(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_y2.png", dpi=150)
    plt.close()

    # фазовый портрет
    plt.figure(figsize=(6, 6))
    plt.plot(df["y1"], df["y2"])
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title(f"{name}: phase portrait")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_phase.png", dpi=150)
    plt.close()


def plot_stability_file(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    name = csv_path.stem

    # |R(x)|
    plt.figure(figsize=(9, 5))
    plt.plot(df["x"], df["AbsR"])
    plt.xlabel("x")
    plt.ylabel("|R(x)|")
    plt.title(f"{name}: |R(x)|")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_AbsR.png", dpi=150)
    plt.close()

    # Re R(x)
    plt.figure(figsize=(9, 5))
    plt.plot(df["x"], df["ReR"])
    plt.xlabel("x")
    plt.ylabel("Re R(x)")
    plt.title(f"{name}: Re R(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_ReR.png", dpi=150)
    plt.close()

    # Im R(x)
    plt.figure(figsize=(9, 5))
    plt.plot(df["x"], df["ImR"])
    plt.xlabel("x")
    plt.ylabel("Im R(x)")
    plt.title(f"{name}: Im R(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_ImR.png", dpi=150)
    plt.close()


def main():
    base_dir = Path(".")
    out_dir = base_dir / "plots"
    traj_out = out_dir / "trajectories"
    stab_out = out_dir / "stability"

    make_dir(traj_out)
    make_dir(stab_out)

    trajectory_files = sorted(base_dir.glob("trajectory_*.csv"))
    stability_files = sorted(base_dir.glob("stability_real_*.csv"))

    for csv_path in trajectory_files:
        plot_trajectory_file(csv_path, traj_out)
        print(f"готово: {csv_path.name}")

    for csv_path in stability_files:
        plot_stability_file(csv_path, stab_out)
        print(f"готово: {csv_path.name}")


if __name__ == "__main__":
    main()
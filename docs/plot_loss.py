"""Plot validation loss for every model variant onto one chart.

Reads docs/runs.csv (the real measured numbers from each run) and writes docs/loss.png.

    python docs/plot_loss.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # no display needed, write straight to a file
import matplotlib.pyplot as plt

DOCS = Path(__file__).resolve().parent
CSV = DOCS / "runs.csv"
OUT = DOCS / "loss.png"

# one colour per variant, dark to light as the model gets better
COLOURS = {
    "bigram":            "#c44536",
    "1 head":            "#d97706",
    "4 heads":           "#ca8a04",
    "+ feed-forward":    "#4d7c0f",
    "+ 3 blocks":        "#0f766e",
    "scaled up (10.8M)": "#1d4ed8",
}


def read_runs(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    steps, losses = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            steps[row["run"]].append(int(row["step"]))
            losses[row["run"]].append(float(row["val_loss"]))
    return {name: (steps[name], losses[name]) for name in steps}


def main() -> None:
    runs = read_runs(CSV)

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, (x, y) in runs.items():
        colour = COLOURS.get(name)
        width = 2.4 if "scaled" in name else 1.6
        ax.plot(x, y, label=f"{name}  ({y[-1]:.2f})", color=colour, linewidth=width)

    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Every step of the build, measured", loc="left", fontsize=13, weight="bold")

    # ln(65) is the loss of an even guess across the 65-character vocabulary:
    # the line any working model has to get under.
    ax.axhline(4.17, color="#9ca3af", linestyle="--", linewidth=1)
    ax.text(150, 4.25, "ln(65) = 4.17 — knowing nothing",
            color="#6b7280", fontsize=9, ha="left")

    ax.set_ylim(1.3, 5.0)
    ax.set_xlim(0, 4900)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

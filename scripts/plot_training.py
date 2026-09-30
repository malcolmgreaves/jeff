"""Plot a scripts/reasoner_lora.py run: validation loss and training loss against the training step, with the learning
rate. Each learning-rate phase is a shaded band labelled with its rate, and each halving is marked with the step whose
adapter the run went back to. The kept adapter's validation loss is labelled (the lowest so far while the run is going).

Usage: python scripts/plot_training.py LOG OUT.png [--learning-rate 1e-4] [--title TITLE] [--subtitle SUBTITLE]"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Chart chrome and categorical slots 1-2 of the dataviz reference palette (light surface), validated as a pair.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VALIDATION, TRAINING = "#2a78d6", "#eb6834"
BANDS = ("#f3f2ee", SURFACE)  # alternate learning-rate phases
TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
VALID = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
HALVED = re.compile(r"Iter (\d+): learning rate halved to ([\d.eE+-]+); back to the adapter from iteration (\d+)")
KEPT = re.compile(r"Saved final adapters to .* \(iteration (\d+), validation loss ([\d.]+)\)")


def parse(text: str) -> dict:
    lines = re.split(r"[\r\n]+", text)
    matches = lambda pattern: [m.groups() for m in map(pattern.search, lines) if m]  # noqa: E731
    return {"train": [(int(s), float(v)) for s, v in matches(TRAIN)], "valid": [(int(s), float(v)) for s, v in matches(VALID)],
            "halved": [(int(s), float(r), int(b)) for s, r, b in matches(HALVED)],
            "kept": next(((int(s), float(v)) for s, v in matches(KEPT)), None)}


def decimal(value: float, _: object) -> str:
    return f"{value:g}"


def rate_text(rate: float) -> str:
    """1e-4, 5e-5, 2.5e-5."""
    mantissa, exponent = f"{rate:e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="The run's starting learning rate")
    parser.add_argument("--title", default="Loss during training")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()
    run = parse(args.log.read_text())
    last = max(step for step, _ in run["train"] + run["valid"])
    kept = run["kept"] or min(run["valid"], key=lambda point: point[1])

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10, "text.color": INK, "axes.labelcolor": SECONDARY,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    figure, axes = plt.subplots(figsize=(10, 5.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    phases = [(0, args.learning_rate)] + [(step, rate) for step, rate, _ in run["halved"]]
    for index, (start, rate) in enumerate(phases):
        end = phases[index + 1][0] if index + 1 < len(phases) else last
        axes.axvspan(start, end, color=BANDS[index % 2], zorder=0, linewidth=0)
        # Label each phase above the plot: centred on a wide band; a narrow band at the right edge is labelled flush right.
        wide = (end - start) / last > 0.25
        x, align = ((start + end) / 2, "center") if wide else ((last, "right") if start / last > 0.75 else (start, "left"))
        axes.text(x, 1.015, f"learning rate {rate_text(rate)}", transform=axes.get_xaxis_transform(),
                  ha=align, va="bottom", color=SECONDARY, fontsize=9)
    for step, rate, back in run["halved"]:
        axes.axvline(step, color=AXIS, linewidth=1, zorder=1)
        axes.text(step, 0.96, f"halved to {rate_text(rate)} at step {step:,};\nback to the step-{back:,} adapter ",
                  transform=axes.get_xaxis_transform(), ha="right", va="top", color=SECONDARY, fontsize=8)

    steps, losses = zip(*run["train"])
    axes.plot(steps, losses, color=TRAINING, linewidth=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=2,
              label="Training loss (mean of each 25 steps)")
    steps, losses = zip(*run["valid"])
    axes.plot(steps, losses, color=VALIDATION, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
              markeredgewidth=2, solid_joinstyle="round", zorder=3, label="Validation loss")
    axes.plot(*kept, marker="o", markersize=11, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.5, zorder=4)
    axes.annotate(f"{'kept' if run['kept'] else 'best so far'}: step {kept[0]:,}, {kept[1]:.4f}", kept, xytext=(0, -22), textcoords="offset points",
                  ha="center", va="top", color=INK, fontsize=9)

    axes.set_yscale("log")
    lowest = min(loss for _, loss in run["train"] + run["valid"])
    axes.set_ylim(bottom=lowest * 0.7)
    axes.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axes.yaxis.set_major_formatter(FuncFormatter(decimal))
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_xlim(0, last)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes.grid(axis="y", which="major", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(length=0)
    axes.set_xlabel("Training step")
    axes.set_ylabel("Loss (log scale)")
    axes.legend(loc="lower left", frameon=False, labelcolor=INK)
    figure.text(0.07, 0.965, args.title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if args.subtitle:
        figure.text(0.07, 0.915, args.subtitle, ha="left", va="top", fontsize=9.5, color=SECONDARY)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

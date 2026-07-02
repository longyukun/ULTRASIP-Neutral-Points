#!/usr/bin/env python3
"""Compare upward and downward Moog tilt calibration sweeps."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_pass(root: Path, direction: str) -> list[dict[str, float]]:
    rows = []
    for nominal_tilt in range(0, 51, 2):
        files = sorted((root / direction / f"tilt_{nominal_tilt:02d}").glob("*.json"))
        result = json.loads(files[-1].read_text(encoding="utf-8"))
        rows.append(
            {
                "nominal": float(nominal_tilt),
                "actual": result["scan"]["actual_tilt_mean_deg"],
                **result["fit"],
            }
        )
    return rows


def linear_fit(rows: list[dict[str, float]]) -> tuple[float, float]:
    xs = [row["actual"] for row in rows]
    ys = [row["offset_deg"] for row in rows]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )
    return y_mean - slope * x_mean, slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root directory containing up/ and down/ results.")
    args = parser.parse_args()
    root = args.root.resolve()
    passes = {direction: load_pass(root, direction) for direction in ("up", "down")}
    colors = {"up": "tab:blue", "down": "tab:red"}
    markers = {"up": "o", "down": "s"}
    fits = {direction: linear_fit(rows) for direction, rows in passes.items()}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for direction, rows in passes.items():
        x = [row["actual"] for row in rows]
        offset = [row["offset_deg"] for row in rows]
        amplitude = [row["amplitude_deg"] for row in rows]
        phase = [row["phase_deg"] for row in rows]
        intercept, slope = fits[direction]
        residual = [y - (intercept + slope * value) for value, y in zip(x, offset)]
        label = f"{direction}: offset={slope:.5f} tilt {intercept:+.3f} deg"

        axes[0, 0].plot(x, offset, marker=markers[direction], color=colors[direction], label=label)
        axes[0, 1].plot(x, residual, marker=markers[direction], color=colors[direction], label=direction)
        axes[1, 0].plot(x, amplitude, marker=markers[direction], color=colors[direction], label=direction)
        axes[1, 1].plot(x, phase, marker=markers[direction], color=colors[direction], label=direction)

    axes[0, 0].set_ylabel("Fitted offset (deg)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Offset linear-fit residual (deg)")
    axes[1, 0].set_ylabel("Sinusoidal amplitude (deg)")
    axes[1, 1].set_ylabel("Phase (deg)")
    for ax in axes.flat:
        ax.set_xlabel("Actual Moog tilt (deg)")
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    fig.suptitle("Moog Tilt Calibration: Upward vs Downward Pass")
    fig.tight_layout()
    fig.savefig(root / "tilt_hysteresis_summary.png", dpi=180)
    plt.close(fig)

    average_slope = sum(slope for _, slope in fits.values()) / len(fits)
    nominal = [row["nominal"] for row in passes["up"]]
    offset_difference = []
    phase_difference = []
    amplitude_difference = []
    for up, down in zip(passes["up"], passes["down"]):
        actual_tilt_difference = down["actual"] - up["actual"]
        offset_difference.append(
            down["offset_deg"] - up["offset_deg"] - average_slope * actual_tilt_difference
        )
        phase_difference.append(down["phase_deg"] - up["phase_deg"])
        amplitude_difference.append(down["amplitude_deg"] - up["amplitude_deg"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    series = (
        (offset_difference, "Offset difference (deg)"),
        (phase_difference, "Phase difference (deg)"),
        (amplitude_difference, "Amplitude difference (deg)"),
    )
    for ax, (values, label) in zip(axes, series):
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.plot(nominal, values, marker="o", color="tab:purple")
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        ax.set_ylabel(label)
        ax.text(0.01, 0.92, f"RMS = {rms:.3f} deg", transform=ax.transAxes, va="top")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Nominal Moog tilt (deg)")
    fig.suptitle("Downward Minus Upward Pass at Matching Tilt")
    fig.tight_layout()
    fig.savefig(root / "tilt_hysteresis_differences.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot zero-offset-corrected PRO3600 offset against commanded Moog tilt."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_ROOT = Path(__file__).parent / "calibration_outputs"
ZERO_OFFSET_DEG = 12.53


def main() -> None:
    commanded_tilts = list(range(0, 31, 2))
    actual_tilts = []
    corrected_offsets = []
    rms_values = []

    for tilt in commanded_tilts:
        files = sorted(
            (OUTPUT_ROOT / f"tilt_{tilt:02d}").glob("*.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not files:
            raise FileNotFoundError(f"No calibration JSON found for tilt={tilt}")
        result = json.loads(files[-1].read_text(encoding="utf-8"))
        fit = result["fit"]
        actual_tilts.append(result["scan"].get("actual_tilt_mean_deg", float(tilt)))
        corrected_offsets.append(fit["offset_deg"] + ZERO_OFFSET_DEG)
        rms_values.append(fit["rms_residual_deg"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ideal_min = min(actual_tilts)
    ideal_max = max(actual_tilts)
    ax.plot(
        [ideal_min, ideal_max],
        [ideal_min, ideal_max],
        "--",
        color="0.35",
        linewidth=1.5,
        label="Ideal: y = tilt",
    )
    points = ax.scatter(
        actual_tilts,
        corrected_offsets,
        c=rms_values,
        cmap="viridis",
        s=55,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        label="Measured offset + 12.53 deg",
    )
    ax.plot(actual_tilts, corrected_offsets, color="tab:blue", linewidth=1, alpha=0.65)
    ax.set_xlabel("Actual Moog tilt (deg)")
    ax.set_ylabel("Fitted offset + 12.53 deg")
    ax.set_title("Corrected PRO3600 Offset vs Moog Tilt")
    ax.grid(True, alpha=0.3)
    ax.legend()
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Fit RMS residual (deg)")
    fig.tight_layout()

    output = OUTPUT_ROOT / "offset_plus_12_53_vs_tilt.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

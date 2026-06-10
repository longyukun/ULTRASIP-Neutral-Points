import argparse
import csv
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from compute_pointing_calibration_v2 import (
    acquisition_names,
    is_real_h5,
    parse_timestamp,
    sinusoid,
)
from robust_np_all_acquisitions import (
    UV_DEG_PER_PIXEL,
    calibration_dtilt_from_attrs,
    detect_sun_center_raw,
    parse_commanded_calibration_dtilt,
    solar_altitude_deg,
    sun_is_complete,
)


DEFAULT_DIR = Path("/Volumes/LaCie/Level2 data/2026_06_05")


def per_frame_residual(handle, name, aq0, lat, lon, alt0, px_per_deg):
    """Return (pan_deg, residual_raw_deg) for one calibration frame, or None.

    residual_raw_deg is the boresight zenith error implied by the sun's pixel
    position vs. where the commanded geometry expects it to be (positive means
    the instrument is pointed too high relative to the sun, same sign/units as
    'Tilt Error v2/v3 Pred [deg]').
    """
    grp = handle.get(name)
    if grp is None:
        return None
    uv = grp.get("UV Image Data")
    raw_ds = uv.get("UV Raw Images") if uv is not None else None
    if raw_ds is None:
        return None
    raw = raw_ds[:]
    frame = raw[0] if raw.ndim == 3 else raw if raw.ndim == 2 else None
    if frame is None:
        return None
    h, w = frame.shape
    cy0 = h / 2.0
    result = detect_sun_center_raw(frame)
    if result is None:
        return None
    cx, cy = result
    if not sun_is_complete(cx, cy, w, h):
        return None
    dtilt, _ = calibration_dtilt_from_attrs(name, aq0.attrs, grp.attrs)
    if not np.isfinite(dtilt):
        return None
    t_i = parse_timestamp(grp.attrs)
    solar_motion = solar_altitude_deg(t_i, lat, lon) - alt0
    cy_expected = cy0 + dtilt * px_per_deg
    dy = cy - cy_expected
    corr_at_frame = dy * UV_DEG_PER_PIXEL
    residual_raw = corr_at_frame - solar_motion
    pan_deg = float(grp.attrs.get("Geometry Pan Used [deg]", aq0.attrs.get("Geometry Pan Used [deg]")))
    commanded_dtilt, _ = parse_commanded_calibration_dtilt(name, grp.attrs)
    return pan_deg, residual_raw, commanded_dtilt


def plot_residuals_by_dtilt(rows, residual_key, title, out_path):
    dtilts = np.array([r["commanded_dtilt_deg"] for r in rows], dtype=float)
    values = np.array([r[residual_key] for r in rows], dtype=float)
    valid = np.isfinite(dtilts) & np.isfinite(values)
    dtilts = dtilts[valid]
    values = values[valid]
    if dtilts.size == 0:
        return

    groups = sorted(set(np.round(dtilts, 2)))
    group_means = []
    group_stds = []
    for g in groups:
        sel = np.isclose(dtilts, g, atol=1e-6)
        group_means.append(np.mean(values[sel]))
        group_stds.append(np.std(values[sel]))
    group_means = np.array(group_means)
    group_stds = np.array(group_stds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for g, mean, std in zip(groups, group_means, group_stds):
        sel = np.isclose(dtilts, g, atol=1e-6)
        color = "tab:blue" if g < 0 else "tab:orange"
        jitter = (np.random.rand(np.sum(sel)) - 0.5) * 0.06
        ax1.scatter(np.full(np.sum(sel), g) + jitter, values[sel], color=color, alpha=0.6, s=14)
        ax1.errorbar([g], [mean], yerr=[std], fmt="o", color="black", capsize=4, zorder=3)
        ax1.text(g, mean + std + 0.03, f"{mean:+.2f}\nSD {std:.2f}",
                 ha="center", va="bottom", fontsize=8)

    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xlabel("Calibration delta tilt [deg]")
    ax1.set_ylabel("Residual [deg]")
    ax1.set_title(f"{title}\n(per calibration frame)")

    ax2.errorbar(groups, group_means, yerr=group_stds, fmt="o-", color="black",
                  capsize=4, label="mean ± 1 SD", zorder=3)

    neg = np.array(groups) < 0
    pos = np.array(groups) > 0
    if np.sum(neg) >= 2:
        slope, intercept = np.polyfit(np.array(groups)[neg], group_means[neg], 1)
        xs = np.array(groups)[neg]
        ax2.plot(xs, slope * xs + intercept, color="tab:blue",
                 label=f"down trend: slope {slope:+.2f}")
    if np.sum(pos) >= 2:
        slope, intercept = np.polyfit(np.array(groups)[pos], group_means[pos], 1)
        xs = np.array(groups)[pos]
        ax2.plot(xs, slope * xs + intercept, color="tab:red",
                 label=f"up trend: slope {slope:+.2f}")

    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.axvline(0.0, color="gray", linewidth=0.6)
    ax2.set_xlabel("Calibration delta tilt [deg]")
    ax2.set_ylabel("Group mean residual [deg]")
    ax2.set_title("Mean residual trend")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_residuals_comparison(rows, out_path):
    dtilts = np.array([r["commanded_dtilt_deg"] for r in rows], dtype=float)
    valid = np.isfinite(dtilts)
    dtilts = dtilts[valid]
    rows = [r for r, v in zip(rows, valid) if v]
    groups = sorted(set(np.round(dtilts, 2)))

    versions = [
        ("residual_raw_deg", "raw", "tab:gray"),
        ("residual_v2_deg", "v2", "tab:blue"),
        ("residual_v3_deg", "v3", "tab:orange"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    n_versions = len(versions)
    width = 0.6
    offsets = np.linspace(-width / 2, width / 2, n_versions)

    for (key, label, color), offset in zip(versions, offsets):
        values = np.array([r[key] for r in rows], dtype=float)
        group_means = []
        group_stds = []
        for g in groups:
            sel = np.isclose(dtilts, g, atol=1e-6)
            group_means.append(np.mean(values[sel]))
            group_stds.append(np.std(values[sel]))
        group_means = np.array(group_means)
        group_stds = np.array(group_stds)
        xs = np.array(groups) + offset
        ax1.errorbar(xs, group_means, yerr=group_stds, fmt="o", color=color,
                      capsize=4, label=label)

        ax2.plot(groups, group_means, "o-", color=color, label=label)

    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xlabel("Calibration delta tilt [deg]")
    ax1.set_ylabel("Group mean ± 1 SD residual [deg]")
    ax1.set_title("Per-group mean residual: raw vs v2 vs v3")
    ax1.legend()

    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.axvline(0.0, color="gray", linewidth=0.6)
    ax2.set_xlabel("Calibration delta tilt [deg]")
    ax2.set_ylabel("Group mean residual [deg]")
    ax2.set_title("Mean residual trend: raw vs v2 vs v3")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each calibration frame, compare the sun's image position to "
            "where the geometry says it should be, and report the residual "
            "zenith error with no correction (raw), with v2 (folder-wide "
            "sinusoid), and with v3 (this file's own calibration only)."
        )
    )
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    directory = args.directory.expanduser()

    files = sorted(p for p in directory.iterdir() if is_real_h5(p))
    if not files:
        raise SystemExit(f"No H5 files in {directory}")

    rows = []
    for path in files:
        with h5py.File(path, "r") as handle:
            meta = handle["Measurement_Metadata"]
            aq0 = handle["Aquistion_0"]
            lat = float(meta.attrs["Latitude"])
            lon = float(meta.attrs["Longitude"])
            t0 = parse_timestamp(aq0.attrs)
            alt0 = solar_altitude_deg(t0, lat, lon)
            px_per_deg = 1.0 / UV_DEG_PER_PIXEL

            calib_group = handle.get("Pointing_Calibration_v2")
            if calib_group is None:
                print(f"SKIP {path.name}: no Pointing_Calibration_v2 group", flush=True)
                continue
            A = float(calib_group.attrs["A_deg"])
            phi = float(calib_group.attrs["phi_rad"])
            C = float(calib_group.attrs["C_deg"])
            tilt_err_v3 = float(calib_group.attrs["this_file_calibration_tilt_err_deg"])

            calib_names = sorted(n for n in handle.keys() if n.startswith("calibration_acqui_"))
            source_names = calib_names if calib_names else ["Aquistion_1"]

            for name in source_names:
                result = per_frame_residual(handle, name, aq0, lat, lon, alt0, px_per_deg)
                if result is None:
                    continue
                pan_deg, residual_raw, commanded_dtilt = result
                tilt_err_pred_v2 = float(sinusoid(pan_deg, A, phi, C))
                residual_v2 = residual_raw - tilt_err_pred_v2
                residual_v3 = residual_raw - tilt_err_v3
                rows.append({
                    "h5_file": path.name,
                    "frame": name,
                    "pan_deg": pan_deg,
                    "commanded_dtilt_deg": commanded_dtilt,
                    "residual_raw_deg": residual_raw,
                    "residual_v2_deg": residual_v2,
                    "residual_v3_deg": residual_v3,
                })

    if not rows:
        raise SystemExit("No usable calibration frames found")

    out_csv = directory / "pointing_residuals_raw_v2_v3.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}", flush=True)

    for label, key in (
        ("raw (no correction)", "residual_raw_deg"),
        ("v2 (folder-wide sinusoid)", "residual_v2_deg"),
        ("v3 (this file's calibration only)", "residual_v3_deg"),
    ):
        values = np.array([r[key] for r in rows])
        mean = np.mean(values)
        std = np.std(values)
        rms = np.sqrt(np.mean(values ** 2))
        mae = np.mean(np.abs(values))
        print(
            f"{label:38s} n={len(values):3d}  mean={mean:+.4f}  std={std:.4f}  "
            f"RMS={rms:.4f}  MAE={mae:.4f}  (deg)",
            flush=True,
        )

    print("\nPer-file mean |residual| (deg):", flush=True)
    print(f"{'h5_file':35s} {'n':>3s} {'raw':>8s} {'v2':>8s} {'v3':>8s}", flush=True)
    by_file = {}
    for r in rows:
        by_file.setdefault(r["h5_file"], []).append(r)
    for fname, frows in by_file.items():
        n = len(frows)
        raw_mae = np.mean([abs(r["residual_raw_deg"]) for r in frows])
        v2_mae = np.mean([abs(r["residual_v2_deg"]) for r in frows])
        v3_mae = np.mean([abs(r["residual_v3_deg"]) for r in frows])
        print(f"{fname:35s} {n:3d} {raw_mae:8.4f} {v2_mae:8.4f} {v3_mae:8.4f}", flush=True)

    for label, key, fname in (
        ("Raw (no correction)", "residual_raw_deg", "pointing_residuals_raw.png"),
        ("v2 (folder-wide sinusoid)", "residual_v2_deg", "pointing_residuals_v2.png"),
        ("v3 (this file's calibration only)", "residual_v3_deg", "pointing_residuals_v3.png"),
    ):
        plot_residuals_by_dtilt(rows, key, f"Sun-center pointing residual: {label}", directory / fname)

    plot_residuals_comparison(rows, directory / "pointing_residuals_comparison.png")


if __name__ == "__main__":
    main()

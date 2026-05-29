import glob
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Depolarization/Instrument Data/UV Polarimeter/Montana_Data/2025_07_10"
TARGET_FILE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_DIR, "NormRoof_20250710_10_23_33.h5")
INTENSITY_MODE = sys.argv[3] if len(sys.argv) > 3 else "p0_p90"
OUT_DIR = "/Users/dustin/Library/CloudStorage/OneDrive-UniversityofArizona/ULTRASIP-Neutral-Points-main/Data_Analysis_Visualization/sun_center_circle_fit"

HFOV = 0.0020
VFOV = HFOV
IMG_X = 2848
IMG_Y = 2848
FRAME_SIZE = IMG_X * IMG_Y
ROW_BLOCK = 128
DISPLAY_STEP = 4


def close_h5_quietly(h5):
    try:
        h5.close()
    except RuntimeError as exc:
        print(f"Warning: ignored HDF5 close error: {exc}")


def read_frame_rows(raw, frame_index, row_start, row_stop):
    start = frame_index * FRAME_SIZE + row_start * IMG_X
    stop = frame_index * FRAME_SIZE + row_stop * IMG_X
    return raw[start:stop].reshape(row_stop - row_start, IMG_X).astype(np.float32)


def iter_intensity_blocks(aq):
    raw = aq["UV Image Data/UV Raw Images"]
    for row_start in range(0, IMG_Y, ROW_BLOCK):
        row_stop = min(row_start + ROW_BLOCK, IMG_Y)
        p0 = read_frame_rows(raw, 0, row_start, row_stop)
        if INTENSITY_MODE == "p0":
            intensity = p0
        else:
            p90 = read_frame_rows(raw, 2, row_start, row_stop)
            intensity = p0 + p90
        yield row_start, row_stop, intensity


def downsample_intensity(aq):
    rows = len(range(0, IMG_Y, DISPLAY_STEP))
    cols = len(range(0, IMG_X, DISPLAY_STEP))
    display = np.empty((rows, cols), dtype=np.float32)
    cursor = 0
    for row_start, row_stop, intensity in iter_intensity_blocks(aq):
        display_rows = np.arange(row_start, row_stop, DISPLAY_STEP)
        if display_rows.size == 0:
            continue
        local_rows = display_rows - row_start
        next_cursor = cursor + len(local_rows)
        display[cursor:next_cursor, :] = intensity[
            local_rows[:, None], np.arange(0, IMG_X, DISPLAY_STEP)
        ]
        cursor = next_cursor
    return display


def collect_mask_points(aq, threshold_fraction=0.75, max_points=20000):
    max_intensity = -np.inf
    for _, _, intensity in iter_intensity_blocks(aq):
        max_intensity = max(max_intensity, float(np.nanmax(intensity)))

    threshold = threshold_fraction * max_intensity
    row_parts = []
    col_parts = []
    cols = np.arange(IMG_X)

    for row_start, row_stop, intensity in iter_intensity_blocks(aq):
        mask = intensity > threshold
        if not np.any(mask):
            continue
        rr, cc = np.nonzero(mask)
        row_parts.append(rr + row_start)
        col_parts.append(cols[cc])

    if not row_parts:
        return np.array([]), np.array([]), max_intensity, threshold

    rows = np.concatenate(row_parts).astype(float)
    cols = np.concatenate(col_parts).astype(float)
    if rows.size > max_points:
        idx = np.linspace(0, rows.size - 1, max_points).astype(int)
        rows = rows[idx]
        cols = cols[idx]
    return rows, cols, max_intensity, threshold


def collect_edge_points(aq, threshold_fraction=0.75, max_points=20000):
    max_intensity = -np.inf
    for _, _, intensity in iter_intensity_blocks(aq):
        max_intensity = max(max_intensity, float(np.nanmax(intensity)))

    threshold = threshold_fraction * max_intensity
    mask = np.zeros((IMG_Y, IMG_X), dtype=bool)

    for row_start, row_stop, intensity in iter_intensity_blocks(aq):
        mask[row_start:row_stop, :] = intensity > threshold

    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    edge = mask & ~interior
    edge[:3, :] = False
    edge[-3:, :] = False
    edge[:, :3] = False
    edge[:, -3:] = False

    rows, cols = np.nonzero(edge)
    if rows.size > max_points:
        idx = np.linspace(0, rows.size - 1, max_points).astype(int)
        rows = rows[idx]
        cols = cols[idx]
    return rows.astype(float), cols.astype(float), max_intensity, threshold


def fit_circle_from_points(rows, cols):
    if rows.size < 20:
        return None
    x = cols
    y = rows
    a = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    radius = float(np.sqrt(r2))
    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    return {
        "x": float(cx),
        "y": float(cy),
        "radius": radius,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "n_points": int(rows.size),
    }


def fit_fixed_radius_from_points(rows, cols, radius, initial_fit=None):
    if rows.size < 20:
        return None
    if initial_fit is None:
        cx = float(np.mean(cols))
        cy = float(np.mean(rows))
    else:
        cx = float(initial_fit["x"])
        cy = float(initial_fit["y"])

    x = cols
    y = rows
    for _ in range(30):
        dx = cx - x
        dy = cy - y
        dist = np.sqrt(dx * dx + dy * dy)
        valid = dist > 1e-6
        if np.count_nonzero(valid) < 20:
            return None
        residual = dist[valid] - radius
        jac = np.column_stack([dx[valid] / dist[valid], dy[valid] / dist[valid]])
        try:
            step = np.linalg.lstsq(jac, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        cx += float(step[0])
        cy += float(step[1])
        if float(np.hypot(step[0], step[1])) < 1e-4:
            break

    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    return {
        "x": float(cx),
        "y": float(cy),
        "radius": float(radius),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "n_points": int(rows.size),
        "fit_method": "fixed_radius_edge",
    }


def fit_completed_disk_centroid_from_arc(aq, radius, initial_fit=None):
    for threshold_fraction in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        rows, cols, max_intensity, threshold = collect_edge_points(aq, threshold_fraction)
        fit = fit_fixed_radius_from_points(rows, cols, radius, initial_fit)
        if fit is None:
            continue
        # The visible edge arc defines a completed disk. For a completed circular
        # disk, the centroid is the fitted disk center.
        fit["threshold_fraction"] = threshold_fraction
        fit["max_intensity"] = max_intensity
        fit["threshold"] = threshold
        fit["is_center_in_frame"] = 0 <= fit["x"] < IMG_X and 0 <= fit["y"] < IMG_Y
        fit["centroid_x"] = fit["x"]
        fit["centroid_y"] = fit["y"]
        fit["fit_method"] = "completed_disk_centroid_from_arc"
        fit["quality_ok"] = fit["n_points"] >= 20 and fit["radius"] >= 20
        fit["strict_quality_ok"] = (
            fit["n_points"] >= 100
            and fit["radius"] >= 20
            and fit["rmse"] <= max(25.0, 0.35 * fit["radius"])
        )
        fit["sun_disk_candidate"] = (
            fit["n_points"] >= 20
            and 20.0 <= fit["radius"] <= 250.0
            and fit["rmse"] <= max(15.0, 0.20 * fit["radius"])
        )
        return fit
    return None


def fit_sun_circle_fixed_radius(aq, radius, initial_fit=None):
    return fit_completed_disk_centroid_from_arc(aq, radius, initial_fit)


def fit_sun_circle(aq):
    # Try high thresholds first; if too few points are visible, relax the threshold.
    for threshold_fraction in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        rows, cols, max_intensity, threshold = collect_mask_points(aq, threshold_fraction)
        fit = fit_circle_from_points(rows, cols)
        if fit is None:
            continue
        fit["threshold_fraction"] = threshold_fraction
        fit["max_intensity"] = max_intensity
        fit["threshold"] = threshold
        fit["fit_method"] = "free_filled_mask"
        fit["is_center_in_frame"] = 0 <= fit["x"] < IMG_X and 0 <= fit["y"] < IMG_Y
        fit["quality_ok"] = fit["n_points"] >= 20 and fit["radius"] >= 20
        fit["strict_quality_ok"] = (
            fit["n_points"] >= 100
            and fit["radius"] >= 20
            and fit["rmse"] <= max(25.0, 0.35 * fit["radius"])
        )
        fit["sun_disk_candidate"] = (
            fit["n_points"] >= 100
            and 20.0 <= fit["radius"] <= 250.0
            and fit["rmse"] <= max(50.0, 0.60 * fit["radius"])
        )
        return fit
    return None


def level1_zenith_at_y(aq, detected_y, reference_y, sza0_altitude):
    tilt = float(aq.attrs["Tilt"])
    sun_alt = float(aq.attrs["Sun Position Altitude"])
    delta_zen = sun_alt - sza0_altitude
    row_dev = (detected_y - reference_y) * VFOV
    altitude = (tilt - delta_zen) - row_dev
    return 90.0 - altitude


def plot_fit(path, aqnum, fit, reference_fit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    h5 = h5py.File(path, "r")
    try:
        aq = h5[f"Aquistion_{aqnum}"]
        display = downsample_intensity(aq)
    finally:
        close_h5_quietly(h5)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(display, cmap="gray", origin="upper", interpolation="none")
    ax.set_title(f"{os.path.basename(path)} Aquisition_{aqnum}")
    ax.set_xlabel(f"column / {DISPLAY_STEP}")
    ax.set_ylabel(f"row / {DISPLAY_STEP}")

    if fit is not None:
        theta = np.linspace(0, 2 * np.pi, 400)
        cx = fit["x"] / DISPLAY_STEP
        cy = fit["y"] / DISPLAY_STEP
        radius = fit["radius"] / DISPLAY_STEP
        ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), "r-", lw=2)
        ax.plot(cx, cy, "r+", ms=14, mew=2)
        ax.text(
            0.02,
            0.98,
            f"center=({fit['x']:.1f},{fit['y']:.1f})\n"
            f"r={fit['radius']:.1f}, rmse={fit['rmse']:.1f}\n"
            f"n={fit['n_points']}, thr={fit['threshold_fraction']:.2f}\n"
            f"quality_ok={fit['quality_ok']}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8},
        )

    if reference_fit is not None:
        ax.plot(reference_fit["x"] / DISPLAY_STEP, reference_fit["y"] / DISPLAY_STEP, "c+", ms=12, mew=2)

    out_path = os.path.join(
        OUT_DIR,
        f"{os.path.basename(path).replace('.h5', '')}_aq{aqnum:02d}_circle_fit.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def compare_file(path, make_plots=False):
    h5 = h5py.File(path, "r")
    try:
        if "Aquistion_0" not in h5 or "Aquistion_1" not in h5:
            return None
        aq0 = h5["Aquistion_0"]
        aq1 = h5["Aquistion_1"]
        sza0_altitude = float(aq0.attrs["Sun Position Altitude"])

        fit0 = fit_sun_circle(aq0)
        if fit0 is None:
            return {
                "file": os.path.basename(path),
                "aq0_fit": fit0,
                "usable": False,
            }

        fit1_free = fit_sun_circle(aq1)
        if fit1_free is None:
            return {"file": os.path.basename(path), "aq1_fit": fit1_free, "usable": False}

        fit1_fixed = fit_sun_circle_fixed_radius(aq1, fit0["radius"], initial_fit=fit1_free)
        fit1 = fit1_fixed if fit1_fixed is not None else fit1_free

        z0 = level1_zenith_at_y(aq0, fit0["y"], fit0["y"], sza0_altitude)
        z1 = level1_zenith_at_y(aq1, fit1["y"], fit0["y"], sza0_altitude)
    finally:
        close_h5_quietly(h5)

    plots = []
    if make_plots:
        plots.append(plot_fit(path, 0, fit0))
        plots.append(plot_fit(path, 1, fit1, reference_fit=fit0))

    return {
        "file": os.path.basename(path),
        "usable": True,
        "aq0_fit": fit0,
        "aq1_fit": fit1,
        "aq1_free_fit": fit1_free,
        "aq0_center_zenith": z0,
        "aq1_center_zenith": z1,
        "diff": z1 - z0,
        "plots": plots,
    }


def main():
    if os.path.exists(TARGET_FILE):
        one = compare_file(TARGET_FILE, make_plots=True)
        print("Single file:")
        print(one)
    else:
        print(f"Single target not found, skipping: {TARGET_FILE}")

    print(f"\nBatch sample, intensity_mode={INTENSITY_MODE}:")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "NormRoof_*.h5")))
    sample_idx = np.linspace(0, len(files) - 1, min(8, len(files))).round().astype(int)
    for i in sample_idx:
        result = compare_file(files[i], make_plots=False)
        if result is None:
            continue
        fit1 = result.get("aq1_fit")
        if not result["usable"]:
            if fit1 is None:
                print(f"{result['file']}: aq1 circle fit failed")
            else:
                print(
                    f"{result['file']}: unusable, center=({fit1['x']:.1f},{fit1['y']:.1f}), "
                    f"r={fit1['radius']:.1f}, rmse={fit1['rmse']:.1f}, n={fit1['n_points']}"
                )
            continue
        fit0 = result["aq0_fit"]
        fit1 = result["aq1_fit"]
        print(
            f"{result['file']}: diff={result['diff']:.3f} deg, "
            f"aq0=({fit0['x']:.1f},{fit0['y']:.1f}), "
            f"aq1=({fit1['x']:.1f},{fit1['y']:.1f}), "
            f"aq1 r={fit1['radius']:.1f}, rmse={fit1['rmse']:.1f}, n={fit1['n_points']}, "
            f"strict={fit1['strict_quality_ok']}"
        )


if __name__ == "__main__":
    main()

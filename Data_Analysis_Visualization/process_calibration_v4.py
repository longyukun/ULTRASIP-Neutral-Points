#!/usr/bin/env python3
"""Process v4 pitch/roll calibration raster-scan H5 files (*_calibration.h5).

These files are produced by the Measurement GUI "Pitch/Roll Calibration Scan":
a rectangle around the sun is raster-scanned row by row and a single
polarizer-0 frame is captured at each grid point. Because only one polarizer
angle is recorded, no Stokes Q/U is computed and no NUC or W-matrix is applied.

For each Calibration_* acquisition this script:
  1. Finds the sun by fitting a circle to the edge of the saturated solar disk
     (free edge fit; falls back to a fixed-radius arc fit using the median
     radius of the good fits when the disk is clipped or the fit is poor).
  2. Assigns per-pixel zenith/azimuth exactly like process_single_level0_1_2
     Level 1 (pixel_geometry with drift correction disabled), writing the
     view_az / view_zen datasets and sun_az / sun_zen scalars.

It then fits the affine model

    o = M @ u + t

where, for each grid point,
    u = (moog_pan + pan_offset_used - sun_pan,
         moog_tilt + tilt_offset_used - sun_alt)   [mount deg, boresight-to-sun]
    o = (cx - IMG_X/2, cy - IMG_Y/2) * DEG_PER_PIXEL [image deg, +x right, +y down]

and derives:
    u0 = -M^-1 t          aim residual: the mount offset at which the sun is
                           exactly centered in the image
    delta_pitch_deg  = -u0_y   correction to ADD to the GUI tilt offset
    delta_pan_deg    = -u0_x   correction to ADD to the GUI pan offset
    delta_roll_deg   = rotation of the raster pattern in the image (camera
                       roll about the boresight), positive = pattern rotated
                       counter-clockwise on the sky (+x right, +y up), after
                       normalizing the pan image direction.

Results and per-point statistics are written to the Pointing_Calibration_v4
group in the H5 file, plus a CSV summary and a diagnostic PNG next to it.

Usage:
    python process_calibration_v4.py FILE_OR_DIR [more...] [--dry-run] [--no-plots]
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_metadata import pixel_geometry


IMG_X = 2848
IMG_Y = 2848
HFOV = 0.0020
VFOV = HFOV
DEG_PER_PIXEL = 7.20 / 3600.0
SCRIPT_VERSION = "1.0"
MIN_FIT_POINTS = 4


def azimuth_to_moog_pan(azimuth_deg):
    """Compass azimuth (north=0, east=90) to Moog pan (south=0, west positive)."""
    return (float(azimuth_deg) % 360.0) - 180.0


def overwrite_dataset(group, name, data, **kwargs):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def finite_attr(attrs, key):
    if key not in attrs:
        return None
    try:
        value = float(attrs[key])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def select_tilt(attrs):
    for key in ["Moog Post Capture Actual Tilt [deg]", "Moog Actual Tilt [deg]", "Tilt"]:
        value = finite_attr(attrs, key)
        if value is not None:
            return value, key
    raise KeyError("No usable tilt attribute found")


def calibration_group_names(handle):
    names = [
        name for name in handle.keys()
        if name.startswith("Calibration_") and "UV Image Data" in handle[name]
    ]
    return sorted(names, key=lambda name: int(handle[name].attrs.get("Grid Index", 0)))


def load_frame(aq):
    raw = aq["UV Image Data/UV Raw Images"][:]
    if raw.ndim == 3:
        return raw[0].astype(np.float32)
    if raw.ndim == 2:
        return raw.astype(np.float32)
    raise ValueError(f"Unsupported UV Raw Images shape: {raw.shape}")


def edge_points_from_mask(mask, margin=3):
    """Boundary pixels of the mask, excluding a margin at the frame border."""
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    edge = mask & ~interior
    edge[:margin, :] = False
    edge[-margin:, :] = False
    edge[:, :margin] = False
    edge[:, -margin:] = False
    rows, cols = np.nonzero(edge)
    return rows.astype(float), cols.astype(float)


def fit_circle_algebraic(rows, cols):
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


def fit_circle_fixed_radius(rows, cols, radius, x0, y0):
    if rows.size < 20:
        return None
    cx = float(x0)
    cy = float(y0)
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
    }


def quality_flags(fit):
    fit["is_center_in_frame"] = bool(0 <= fit["x"] < IMG_X and 0 <= fit["y"] < IMG_Y)
    fit["sun_disk_candidate"] = bool(
        fit["n_points"] >= 50
        and 20.0 <= fit["radius"] <= 250.0
        and fit["rmse"] <= max(15.0, 0.20 * fit["radius"])
    )
    return fit


def fit_sun_disk(frame):
    """Free circle fit to the edge of the bright (saturated) solar disk."""
    vmax = float(np.nanmax(frame))
    if vmax <= 0:
        return None
    for threshold_fraction in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        mask = frame >= threshold_fraction * vmax
        if int(mask.sum()) < 50:
            continue
        rows, cols = edge_points_from_mask(mask)
        fit = fit_circle_algebraic(rows, cols)
        if fit is None:
            continue
        fit["threshold_fraction"] = threshold_fraction
        fit["max_intensity"] = vmax
        fit["fit_method"] = "free_edge_circle"
        return quality_flags(fit)
    return None


def fit_sun_disk_fixed_radius(frame, radius, initial_fit=None):
    """Fixed-radius arc fit for clipped disks, completing the circle."""
    vmax = float(np.nanmax(frame))
    if vmax <= 0:
        return None
    for threshold_fraction in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        mask = frame >= threshold_fraction * vmax
        if int(mask.sum()) < 50:
            continue
        rows, cols = edge_points_from_mask(mask)
        if rows.size < 20:
            continue
        if initial_fit is not None:
            x0, y0 = initial_fit["x"], initial_fit["y"]
        else:
            x0, y0 = float(np.mean(cols)), float(np.mean(rows))
        fit = fit_circle_fixed_radius(rows, cols, radius, x0, y0)
        if fit is None:
            continue
        fit["threshold_fraction"] = threshold_fraction
        fit["max_intensity"] = vmax
        fit["fit_method"] = "fixed_radius_edge_arc"
        return quality_flags(fit)
    return None


def collect_point(aq):
    attrs = aq.attrs
    pan_offset = float(attrs["Pan Offset"])
    tilt_offset = float(attrs["Tilt Offset"])
    moog_pan = float(attrs.get("Moog Actual Pan [deg]", attrs["Pan"]))
    moog_tilt = float(attrs.get("Moog Actual Tilt [deg]", attrs["Tilt"]))
    sun_az = float(attrs.get("Sun Pre Capture Azimuth [deg]", attrs["Sun Position Azimuth"]))
    sun_alt = float(attrs.get("Sun Pre Capture Altitude [deg]", attrs["Sun Position Altitude"]))
    sun_pan = azimuth_to_moog_pan(sun_az)
    return {
        "grid_index": int(attrs.get("Grid Index", -1)),
        "grid_row": int(attrs.get("Grid Row", -1)),
        "grid_col": int(attrs.get("Grid Col", -1)),
        "dpan_cmd": float(attrs.get("Grid Delta Pan [deg]", np.nan)),
        "dtilt_cmd": float(attrs.get("Grid Delta Tilt [deg]", np.nan)),
        "pan_offset": pan_offset,
        "tilt_offset": tilt_offset,
        "u_x": moog_pan + pan_offset - sun_pan,
        "u_y": moog_tilt + tilt_offset - sun_alt,
        "sun_alt": sun_alt,
    }


def fit_affine(u, o):
    """Least-squares o = M @ u + t. u, o are (n, 2) arrays in degrees."""
    design = np.column_stack([u, np.ones(len(u))])
    solution, _, rank, _ = np.linalg.lstsq(design, o, rcond=None)
    if rank < 3:
        return None
    m_matrix = solution[:2].T
    t_vec = solution[2]
    residuals = o - (u @ solution[:2] + t_vec)
    return m_matrix, t_vec, residuals


def roll_from_matrix(m_matrix):
    """Camera roll from the affine matrix, normalizing the pan image direction."""
    det = float(np.linalg.det(m_matrix))
    pan_sign = -1.0 if det < 0 else 1.0
    normalized = np.diag([pan_sign, 1.0]) @ m_matrix
    u_svd, scales, vt_svd = np.linalg.svd(normalized)
    rotation = u_svd @ vt_svd
    # Pixel coords are +y down; negate so positive roll = counter-clockwise on
    # the sky with +x right, +y up.
    roll_deg = -float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    scale_pan = float(np.linalg.norm(m_matrix[:, 0]))
    scale_tilt = float(np.linalg.norm(m_matrix[:, 1]))
    return roll_deg, pan_sign, scale_pan, scale_tilt


def assign_pixel_geometry(handle, names, x_center, y_center):
    """Level 1 style per-pixel zenith/azimuth (drift correction disabled)."""
    for name in names:
        aq = handle[name]
        uv_data = aq["UV Image Data"]
        attrs = aq.attrs

        sun_az_attr = float(attrs["Sun Position Azimuth"])
        sun_alt_attr = float(attrs["Sun Position Altitude"])
        pan = float(attrs.get("Moog Actual Pan [deg]", attrs["Pan"]))
        tilt, tilt_source = select_tilt(attrs)
        attrs["Geometry Pan Source"] = (
            "Moog Actual Pan [deg]" if "Moog Actual Pan [deg]" in attrs else "Pan")
        attrs["Geometry Tilt Source"] = tilt_source
        attrs["Geometry Sun Altitude Drift Correction"] = "disabled"
        attrs["Geometry Pan Used [deg]"] = pan
        attrs["Geometry Tilt Used [deg]"] = float(tilt)

        view_az, view_zen, sun_az, sun_zen = pixel_geometry(
            sun_alt_attr,
            HFOV,
            VFOV,
            IMG_X,
            IMG_Y,
            x_center,
            y_center,
            pan,
            tilt,
            sun_az_attr,
            sun_alt_attr,
        )
        overwrite_dataset(uv_data, "view_az", np.degrees(view_az).astype(np.float32), compression="gzip")
        overwrite_dataset(uv_data, "view_zen", (90 - np.degrees(view_zen)).astype(np.float32), compression="gzip")
        overwrite_dataset(uv_data, "sun_az", np.array(np.degrees(sun_az), dtype=np.float32))
        overwrite_dataset(uv_data, "sun_zen", np.array(90 - np.degrees(sun_zen), dtype=np.float32))
        print(f"  geometry: {name}", flush=True)


def write_fit_attrs(aq, fit, expected=None):
    attrs = aq.attrs
    attrs["Sun Circle Fit OK"] = int(fit is not None)
    if fit is None:
        return
    attrs["Sun Circle Fit X [px]"] = fit["x"]
    attrs["Sun Circle Fit Y [px]"] = fit["y"]
    attrs["Sun Circle Fit Radius [px]"] = fit["radius"]
    attrs["Sun Circle Fit RMSE [px]"] = fit["rmse"]
    attrs["Sun Circle Fit N Points"] = fit["n_points"]
    attrs["Sun Circle Fit Threshold Fraction"] = fit["threshold_fraction"]
    attrs["Sun Circle Fit Method"] = fit["fit_method"]
    attrs["Sun Circle Fit Candidate"] = int(fit["sun_disk_candidate"])
    if expected is not None:
        attrs["Sun Circle Fit Expected X [px]"] = expected[0]
        attrs["Sun Circle Fit Expected Y [px]"] = expected[1]
        attrs["Sun Circle Fit Residual X [px]"] = fit["x"] - expected[0]
        attrs["Sun Circle Fit Residual Y [px]"] = fit["y"] - expected[1]


def write_v4_group(handle, summary, table):
    if "Pointing_Calibration_v4" in handle:
        del handle["Pointing_Calibration_v4"]
    grp = handle.create_group("Pointing_Calibration_v4")
    grp.attrs["model"] = (
        "o = M @ u + t; u = (moog_pan + pan_offset - sun_pan, moog_tilt + tilt_offset - sun_alt) "
        "[mount deg]; o = (cx - IMG_X/2, cy - IMG_Y/2) * deg_per_pixel [image deg, +x right, +y down]; "
        "u0 = -inv(M) @ t; delta_pan/pitch = -u0; suggested_offset = used_offset - u0; "
        "delta_roll = rotation of M after normalizing pan image direction, positive = "
        "counter-clockwise on sky (+x right, +y up)"
    )
    for key, value in summary.items():
        grp.attrs[key] = value
    for key, values in table.items():
        if all(isinstance(v, str) for v in values):
            grp.create_dataset(key, data=np.array(values, dtype=h5py.string_dtype()))
        else:
            grp.create_dataset(key, data=np.asarray(values, dtype=float))


def save_plot(path, points, fits, m_matrix, t_vec, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for point, fit in zip(points, fits):
        if fit is None:
            continue
        expected = None
        if m_matrix is not None:
            o_pred = m_matrix @ np.array([point["u_x"], point["u_y"]]) + t_vec
            expected = (o_pred / DEG_PER_PIXEL) + np.array([IMG_X / 2, IMG_Y / 2])
            ax.plot(expected[0], expected[1], "c+", ms=10, mew=1.5)
        ax.plot(fit["x"], fit["y"], "r.", ms=8)
        ax.annotate(
            f"r{point['grid_row']}c{point['grid_col']}",
            (fit["x"], fit["y"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlim(0, IMG_X)
    ax.set_ylim(IMG_Y, 0)
    ax.set_aspect("equal")
    ax.set_title("Fitted sun centers (red) vs affine model (cyan)")
    ax.set_xlabel("column [px]")
    ax.set_ylabel("row [px]")

    ax = axes[1]
    if m_matrix is not None:
        residual_list = []
        for point, fit in zip(points, fits):
            if fit is None:
                continue
            o_obs = (np.array([fit["x"], fit["y"]]) - np.array([IMG_X / 2, IMG_Y / 2])) * DEG_PER_PIXEL
            o_pred = m_matrix @ np.array([point["u_x"], point["u_y"]]) + t_vec
            residual_list.append((point, o_obs - o_pred))
        max_res = max((float(np.hypot(*res)) for _, res in residual_list), default=1.0)
        dpan_values = [p["dpan_cmd"] for p, _ in residual_list]
        step = float(np.median(np.diff(sorted(set(dpan_values))))) if len(set(dpan_values)) > 1 else 0.5
        arrow_scale = 0.6 * step / max(max_res, 1e-9)
        for point, res in residual_list:
            ax.arrow(
                point["dpan_cmd"],
                point["dtilt_cmd"],
                res[0] * arrow_scale,
                -res[1] * arrow_scale,
                head_width=0.04 * step,
                color="tab:red",
            )
            ax.plot(point["dpan_cmd"], point["dtilt_cmd"], "k.", ms=4)
        ax.set_title(f"Residuals x{arrow_scale:.0f} on command grid (max {max_res:.3f} deg)")
        ax.set_xlabel("commanded delta pan [deg]")
        ax.set_ylabel("commanded delta tilt [deg]")
        ax.set_aspect("equal")
        text = (
            f"delta_pitch={summary['delta_pitch_deg']:+.4f} deg\n"
            f"delta_pan={summary['delta_pan_deg']:+.4f} deg\n"
            f"delta_roll={summary['delta_roll_deg']:+.4f} deg\n"
            f"rms=({summary['residual_rms_x_deg']:.4f}, {summary['residual_rms_y_deg']:.4f}) deg\n"
            f"n={summary['fit_n_points_used']}"
        )
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8},
        )
    else:
        ax.text(0.5, 0.5, "Affine fit skipped", ha="center", va="center")

    fig.suptitle(Path(path).name)
    fig.tight_layout()
    out_path = str(path).replace(".h5", "_v4_fit.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def process_file(path, dry_run=False, make_plots=True):
    print(f"\n=== {path}", flush=True)
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as handle:
        names = calibration_group_names(handle)
        if not names:
            print("  no Calibration_* groups; skipping", flush=True)
            return None

        points = []
        fits = []
        for name in names:
            aq = handle[name]
            point = collect_point(aq)
            point["name"] = name
            frame = load_frame(aq)
            fit = fit_sun_disk(frame)
            points.append(point)
            fits.append(fit)
            del frame

        good_radii = [f["radius"] for f in fits if f is not None and f["sun_disk_candidate"]]
        median_radius = float(np.median(good_radii)) if good_radii else None
        if median_radius is not None:
            for idx, (name, fit) in enumerate(zip(names, fits)):
                if fit is not None and fit["sun_disk_candidate"]:
                    continue
                frame = load_frame(handle[name])
                refit = fit_sun_disk_fixed_radius(frame, median_radius, initial_fit=fit)
                del frame
                if refit is not None and (fit is None or refit["rmse"] < fit["rmse"]):
                    fits[idx] = refit

        for name, point, fit in zip(names, points, fits):
            status = "ok" if fit is not None and fit["sun_disk_candidate"] else "poor" if fit else "FAILED"
            detail = ""
            if fit is not None:
                detail = (
                    f" center=({fit['x']:.1f},{fit['y']:.1f}) r={fit['radius']:.1f} "
                    f"rmse={fit['rmse']:.2f} n={fit['n_points']} [{fit['fit_method']}]"
                )
            print(
                f"  {name}: dpan={point['dpan_cmd']:+.2f} dtilt={point['dtilt_cmd']:+.2f} "
                f"fit={status}{detail}",
                flush=True,
            )

        used = [
            (point, fit) for point, fit in zip(points, fits)
            if fit is not None and fit["sun_disk_candidate"]
        ]
        m_matrix = None
        t_vec = None
        summary = None
        if len(used) >= MIN_FIT_POINTS:
            u = np.array([[p["u_x"], p["u_y"]] for p, _ in used])
            o = np.array(
                [
                    [(f["x"] - IMG_X / 2) * DEG_PER_PIXEL, (f["y"] - IMG_Y / 2) * DEG_PER_PIXEL]
                    for _, f in used
                ]
            )
            result = fit_affine(u, o)
            if result is None:
                print("  affine fit skipped: degenerate grid (rank < 3)", flush=True)
            else:
                m_matrix, t_vec, residuals = result
                u0 = -np.linalg.solve(m_matrix, t_vec)
                roll_deg, pan_sign, scale_pan, scale_tilt = roll_from_matrix(m_matrix)
                pan_offset_used = used[0][0]["pan_offset"]
                tilt_offset_used = used[0][0]["tilt_offset"]
                mean_alt = float(np.mean([p["sun_alt"] for p, _ in used]))
                summary = {
                    "script_version": SCRIPT_VERSION,
                    "processed_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "deg_per_pixel": DEG_PER_PIXEL,
                    "fit_n_points_total": len(points),
                    "fit_n_points_used": len(used),
                    "affine_matrix": m_matrix,
                    "affine_translation_deg": t_vec,
                    "aim_residual_u0_pan_deg": float(u0[0]),
                    "aim_residual_u0_tilt_deg": float(u0[1]),
                    "delta_pan_deg": float(-u0[0]),
                    "delta_pitch_deg": float(-u0[1]),
                    "delta_roll_deg": float(roll_deg),
                    "pan_offset_used_deg": float(pan_offset_used),
                    "tilt_offset_used_deg": float(tilt_offset_used),
                    "suggested_pan_offset_deg": float(pan_offset_used - u0[0]),
                    "suggested_tilt_offset_deg": float(tilt_offset_used - u0[1]),
                    "pan_image_direction": "+pan moves sun left" if pan_sign < 0 else "+pan moves sun right",
                    "scale_pan": scale_pan,
                    "scale_tilt": scale_tilt,
                    "expected_scale_pan_cos_alt": float(np.cos(np.radians(mean_alt))),
                    "mean_sun_altitude_deg": mean_alt,
                    "residual_rms_x_deg": float(np.sqrt(np.mean(residuals[:, 0] ** 2))),
                    "residual_rms_y_deg": float(np.sqrt(np.mean(residuals[:, 1] ** 2))),
                    "residual_rms_x_px": float(np.sqrt(np.mean(residuals[:, 0] ** 2)) / DEG_PER_PIXEL),
                    "residual_rms_y_px": float(np.sqrt(np.mean(residuals[:, 1] ** 2)) / DEG_PER_PIXEL),
                    "residual_max_abs_x_deg": float(np.max(np.abs(residuals[:, 0]))),
                    "residual_max_abs_y_deg": float(np.max(np.abs(residuals[:, 1]))),
                    "residual_mean_x_deg": float(np.mean(residuals[:, 0])),
                    "residual_mean_y_deg": float(np.mean(residuals[:, 1])),
                    "residual_std_x_deg": float(np.std(residuals[:, 0])),
                    "residual_std_y_deg": float(np.std(residuals[:, 1])),
                }
                print(
                    f"  FIT n={len(used)}/{len(points)}: "
                    f"delta_pitch={summary['delta_pitch_deg']:+.4f} deg, "
                    f"delta_pan={summary['delta_pan_deg']:+.4f} deg, "
                    f"delta_roll={summary['delta_roll_deg']:+.4f} deg",
                    flush=True,
                )
                print(
                    f"  suggested offsets: pan={summary['suggested_pan_offset_deg']:.3f} "
                    f"(used {pan_offset_used:.3f}), tilt={summary['suggested_tilt_offset_deg']:.3f} "
                    f"(used {tilt_offset_used:.3f}); "
                    f"rms=({summary['residual_rms_x_deg']:.4f}, {summary['residual_rms_y_deg']:.4f}) deg",
                    flush=True,
                )
        else:
            print(
                f"  affine fit skipped: only {len(used)} good circle fits "
                f"(need >= {MIN_FIT_POINTS})",
                flush=True,
            )

        # Reference pixel for per-pixel geometry: fitted center of the good grid
        # point closest to (0, 0), mirroring Level 1's use of Aquistion_0.
        reference = None
        if used:
            reference = min(used, key=lambda pf: abs(pf[0]["dpan_cmd"]) + abs(pf[0]["dtilt_cmd"]))
        x_center = reference[1]["x"] if reference else IMG_X / 2
        y_center = reference[1]["y"] if reference else IMG_Y / 2

        table = {
            "name": [p["name"] for p in points],
            "grid_index": [p["grid_index"] for p in points],
            "grid_row": [p["grid_row"] for p in points],
            "grid_col": [p["grid_col"] for p in points],
            "dpan_cmd_deg": [p["dpan_cmd"] for p in points],
            "dtilt_cmd_deg": [p["dtilt_cmd"] for p in points],
            "u_pan_deg": [p["u_x"] for p in points],
            "u_tilt_deg": [p["u_y"] for p in points],
            "fit_ok": [1.0 if f is not None and f["sun_disk_candidate"] else 0.0 for f in fits],
            "center_x_px": [f["x"] if f else np.nan for f in fits],
            "center_y_px": [f["y"] if f else np.nan for f in fits],
            "radius_px": [f["radius"] if f else np.nan for f in fits],
            "circle_rmse_px": [f["rmse"] if f else np.nan for f in fits],
        }

        if not dry_run:
            for name, point, fit in zip(names, points, fits):
                expected = None
                if m_matrix is not None:
                    o_pred = m_matrix @ np.array([point["u_x"], point["u_y"]]) + t_vec
                    expected = (
                        float(o_pred[0] / DEG_PER_PIXEL + IMG_X / 2),
                        float(o_pred[1] / DEG_PER_PIXEL + IMG_Y / 2),
                    )
                write_fit_attrs(handle[name], fit, expected)

            assign_pixel_geometry(handle, names, x_center, y_center)
            handle["Measurement_Metadata"].attrs["Center Pixel (x,y)"] = np.array([x_center, y_center])
            handle["Measurement_Metadata"].attrs["Processed Level"] = "Calibration v4"
            handle["Measurement_Metadata"].attrs["Calibration v4 Note"] = (
                "Single polarizer angle scan: no Q/U Stokes, no NUC, no W matrix. "
                "Per-pixel view_az/view_zen assigned Level 1 style; sun centers from "
                "circle fits to the solar disk edge."
            )
            if summary is not None:
                write_v4_group(handle, summary, table)

    csv_path = str(path).replace(".h5", "_v4_points.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(table.keys())
        writer.writerow(keys)
        for idx in range(len(points)):
            writer.writerow([table[key][idx] for key in keys])
        if summary is not None:
            writer.writerow([])
            for key in (
                "delta_pitch_deg", "delta_pan_deg", "delta_roll_deg",
                "suggested_pan_offset_deg", "suggested_tilt_offset_deg",
                "residual_rms_x_deg", "residual_rms_y_deg",
                "fit_n_points_used", "fit_n_points_total",
            ):
                writer.writerow([key, summary[key]])
    print(f"  wrote {csv_path}", flush=True)

    if make_plots:
        plot_path = save_plot(path, points, fits, m_matrix, t_vec, summary or {})
        print(f"  wrote {plot_path}", flush=True)

    return summary


def gather_files(paths):
    files = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files.extend(sorted(p.glob("*_calibration.h5")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARNING: not found: {p}", flush=True)
    return [f for f in files if not f.name.startswith("._")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="*_calibration.h5 files or directories containing them")
    parser.add_argument("--dry-run", action="store_true", help="Compute and report only; do not modify H5 files")
    parser.add_argument("--no-plots", action="store_true", help="Skip the diagnostic PNG")
    args = parser.parse_args()

    files = gather_files(args.paths)
    if not files:
        raise SystemExit("No calibration H5 files found")

    start = time.time()
    results = []
    for path in files:
        summary = process_file(path, dry_run=args.dry_run, make_plots=not args.no_plots)
        results.append((path, summary))

    print(f"\nDone in {time.time() - start:.1f} s")
    fitted = [(p, s) for p, s in results if s is not None]
    if len(fitted) > 1:
        pitches = [s["delta_pitch_deg"] for _, s in fitted]
        rolls = [s["delta_roll_deg"] for _, s in fitted]
        print(
            f"Across {len(fitted)} files: delta_pitch mean={np.mean(pitches):+.4f} "
            f"std={np.std(pitches):.4f} deg; delta_roll mean={np.mean(rolls):+.4f} "
            f"std={np.std(rolls):.4f} deg"
        )


if __name__ == "__main__":
    main()

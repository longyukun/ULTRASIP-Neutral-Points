import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
import sys
import threading

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("/private/tmp", "ultrasip_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
)


IMG_EDGE_MARGIN = 80
PLOT_LOCK = threading.Lock()
DEFAULT_CIRCULAR_CROP_RADIUS_FRACTION = 0.82
DEFAULT_CIRCULAR_CROP_RADIUS_SHRINK_PX = 100.0
UV_ARCSEC_PER_PIXEL = 7.20
UV_DEG_PER_PIXEL = UV_ARCSEC_PER_PIXEL / 3600.0
SUN_COMPLETENESS_MARGIN_PX = 220


def circular_crop_radius_fraction():
    try:
        return float(os.environ.get(
            "NP_CIRCULAR_CROP_RADIUS_FRACTION",
            DEFAULT_CIRCULAR_CROP_RADIUS_FRACTION,
        ))
    except Exception:
        return DEFAULT_CIRCULAR_CROP_RADIUS_FRACTION


def circular_crop_radius_shrink_px():
    try:
        return float(os.environ.get(
            "NP_CIRCULAR_CROP_RADIUS_SHRINK_PX",
            DEFAULT_CIRCULAR_CROP_RADIUS_SHRINK_PX,
        ))
    except Exception:
        return DEFAULT_CIRCULAR_CROP_RADIUS_SHRINK_PX


def circular_crop_mask(shape):
    radius_fraction = circular_crop_radius_fraction()
    rows, cols = np.indices(shape)
    center_row = (shape[0] - 1) / 2.0
    center_col = (shape[1] - 1) / 2.0
    radius = max(1.0, radius_fraction * min(shape) / 2.0 - circular_crop_radius_shrink_px())
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2


def circular_crop_radius_px(shape):
    radius = circular_crop_radius_fraction() * min(shape) / 2.0 - circular_crop_radius_shrink_px()
    return max(1.0, float(radius))


def circular_crop_slices(shape):
    radius = circular_crop_radius_px(shape)
    half_width = max(1, int(np.floor(radius)))
    center_row = (shape[0] - 1) / 2.0
    center_col = (shape[1] - 1) / 2.0
    row_start = max(0, int(np.floor(center_row - half_width)))
    row_stop = min(shape[0], int(np.ceil(center_row + half_width + 1)))
    col_start = max(0, int(np.floor(center_col - half_width)))
    col_stop = min(shape[1], int(np.ceil(center_col + half_width + 1)))
    return slice(row_start, row_stop), slice(col_start, col_stop), radius


def circular_crop_mask_for_radius(shape, radius):
    rows, cols = np.indices(shape)
    center_row = (shape[0] - 1) / 2.0
    center_col = (shape[1] - 1) / 2.0
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2


def apply_center_crop(*arrays):
    row_slice, col_slice, radius = circular_crop_slices(arrays[0].shape)
    return [np.asarray(arr)[row_slice, col_slice] for arr in arrays], radius


def acquisition_names(handle):
    names = [name for name in handle.keys() if name.startswith("Aquistion_")]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def fallback_csv_path(h5_path):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robust_np_outputs")
    return os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(h5_path))[0] + "_robust_np_methods.csv")


def write_rows_csv(preferred_path, h5_path, rows):
    try:
        with open(preferred_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return preferred_path
    except PermissionError:
        fallback_path = fallback_csv_path(h5_path)
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return fallback_path


def require_level2_products(uv, aq_name):
    required_stokes = ["I_corrected", "Q_corrected", "U_corrected"]
    missing = [name for name in required_stokes if name not in uv]
    has_angles = (
        ("view_zen_corrected" in uv and "view_az_corrected" in uv)
        or ("view_zen" in uv and "view_az" in uv)
    )
    if not has_angles:
        missing.extend(["view_zen/view_az"])
    if missing:
        raise RuntimeError(
            f"{aq_name} is not Level2-ready. Missing {', '.join(missing)}. "
            "Run process_single_level0_1_2.py on this H5 before robust NP analysis.")


def validate_product_shapes(aq_name, products):
    reference_shape = products["I"].shape
    for name in ["Q", "U", "view_zen", "view_az"]:
        if products[name].shape != reference_shape:
            raise ValueError(
                f"{aq_name}: {name} shape {products[name].shape} does not match "
                f"I shape {reference_shape}. Expected all image products as (row, col).")


def corrected_zenith_to_canonical(uv, corrected_name, reference_name=None):
    corrected = uv[corrected_name][:]
    converted = 90.0 - corrected
    if reference_name and reference_name in uv:
        reference = uv[reference_name][:]
        direct_err = np.nanmedian(np.abs(corrected - reference))
        converted_err = np.nanmedian(np.abs(converted - reference))
        if np.isfinite(direct_err) and np.isfinite(converted_err):
            return converted if converted_err < direct_err else corrected
    return converted


def scalar_corrected_zenith_to_canonical(uv, corrected_name, reference_name=None):
    corrected = float(uv[corrected_name][()])
    converted = 90.0 - corrected
    if reference_name and reference_name in uv:
        reference = float(uv[reference_name][()])
        return converted if abs(converted - reference) < abs(corrected - reference) else corrected
    return converted


def load_view_geometry(uv):
    if "view_zen" in uv and "view_az" in uv:
        view_zen = uv["view_zen"][:]
        view_az = uv["view_az"][:]
    else:
        view_zen = uv["view_zen_corrected"][:]
        view_az = uv["view_az_corrected"][:]
    return view_zen, view_az


def load_sun_zenith(uv, aq):
    if "sun_zen_corrected" in uv:
        return scalar_corrected_zenith_to_canonical(uv, "sun_zen_corrected", "sun_zen")
    if "sun_zen" in uv:
        return float(uv["sun_zen"][()])
    return 90 - float(aq.attrs["Sun Position Altitude"])


def solar_altitude_deg(dt, lat_deg, lon_deg):
    local_dt = dt.astimezone()
    day = local_dt.timetuple().tm_yday
    hour = local_dt.hour + local_dt.minute / 60.0 + local_dt.second / 3600.0
    utc_offset_minutes = local_dt.utcoffset().total_seconds() / 60.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    eq_time = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    time_offset = eq_time + 4.0 * lon_deg - utc_offset_minutes
    true_solar_time = (hour * 60.0 + time_offset) % 1440.0
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    cos_zenith = (math.sin(lat) * math.sin(decl)
                  + math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
    cos_zenith = min(max(cos_zenith, -1.0), 1.0)
    return 90.0 - math.degrees(math.acos(cos_zenith))


def detect_sun_center_raw(frame):
    image = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.8])
    if hi <= lo:
        return None
    norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    threshold = max(float(np.percentile(norm, 99.3)), 0.72)
    mask = norm >= threshold
    if not np.any(mask):
        return None
    weights = np.where(mask, norm, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return None
    yy, xx = np.indices(image.shape)
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def sun_is_complete(cx, cy, width, height):
    m = SUN_COMPLETENESS_MARGIN_PX
    return m <= cx <= width - m and m <= cy <= height - m


def finite_attr(attrs, key):
    try:
        value = float(attrs.get(key, np.nan))
    except Exception:
        value = np.nan
    return value if np.isfinite(value) else np.nan


def parse_commanded_calibration_dtilt(name, attrs):
    try:
        dtilt = float(attrs.get("Delta Tilt From Sun [deg]", np.nan))
    except Exception:
        dtilt = np.nan
    if np.isfinite(dtilt):
        return dtilt, "Delta Tilt From Sun [deg]"

    parts = name.split("_")
    direction = parts[2] if len(parts) > 2 else "center"
    amount_str = parts[3].replace("p", ".") if len(parts) > 3 else "0"
    try:
        sign = 1.0 if direction == "up" else -1.0 if direction == "down" else 0.0
        return sign * float(amount_str), "calibration group name"
    except Exception:
        return np.nan, "unavailable"


def calibration_dtilt_from_attrs(name, aq0_attrs, attrs):
    for key, source in (
        ("Moog Post Capture Actual Tilt [deg]", "post-capture actual tilt"),
        ("Moog Actual Tilt [deg]", "actual tilt"),
    ):
        base = finite_attr(aq0_attrs, key)
        current = finite_attr(attrs, key)
        if np.isfinite(base) and np.isfinite(current):
            return current - base, source
    return parse_commanded_calibration_dtilt(name, attrs)


def compute_calibration_tilt_correction(handle):
    meta = handle.get("Measurement_Metadata")
    aq0 = handle.get("Aquistion_0")
    if meta is None or aq0 is None:
        return {
            "correction_deg": np.nan,
            "applied_deg": 0.0,
            "n_used": 0,
            "n_total": 0,
            "details": [],
        }

    try:
        lat = float(meta.attrs.get("Latitude", 0.0))
        lon = float(meta.attrs.get("Longitude", 0.0))
        t0_str = str(aq0.attrs.get("Timestamp Local New")
                     or aq0.attrs.get("Timestamp UTC New", ""))
        t0 = datetime.fromisoformat(t0_str.replace("Z", "+00:00"))
        alt0 = solar_altitude_deg(t0, lat, lon)
    except Exception:
        return {
            "correction_deg": np.nan,
            "applied_deg": 0.0,
            "n_used": 0,
            "n_total": 0,
            "details": [],
        }

    px_per_deg = 1.0 / UV_DEG_PER_PIXEL
    calib_names = sorted(n for n in handle.keys() if n.startswith("calibration_acqui_"))
    corrections = []
    details = []

    for name in calib_names:
        grp = handle[name]
        uv = grp.get("UV Image Data")
        raw_ds = uv.get("UV Raw Images") if uv is not None else None
        if raw_ds is None:
            details.append({"name": name, "skip": "no UV Raw Images"})
            continue

        raw = raw_ds[:]
        frame = raw[0] if raw.ndim == 3 else raw if raw.ndim == 2 else None
        if frame is None:
            details.append({"name": name, "skip": "unexpected raw shape"})
            continue

        h, w = frame.shape
        cy0 = h / 2.0
        result = detect_sun_center_raw(frame)
        if result is None:
            details.append({"name": name, "skip": "sun not detected"})
            continue
        cx, cy = result
        if not sun_is_complete(cx, cy, w, h):
            details.append({
                "name": name,
                "skip": "sun incomplete",
                "cx_px": round(cx, 1),
                "cy_px": round(cy, 1),
            })
            continue

        dtilt, dtilt_source = calibration_dtilt_from_attrs(name, aq0.attrs, grp.attrs)
        if not np.isfinite(dtilt):
            details.append({"name": name, "skip": "cannot determine dtilt"})
            continue

        try:
            ti_str = str(grp.attrs.get("Timestamp Local New")
                         or grp.attrs.get("Timestamp UTC New", ""))
            t_i = datetime.fromisoformat(ti_str.replace("Z", "+00:00"))
            solar_motion = solar_altitude_deg(t_i, lat, lon) - alt0
        except Exception:
            solar_motion = 0.0

        cy_expected = cy0 + dtilt * px_per_deg
        dy = cy - cy_expected
        corr_at_frame = dy * UV_DEG_PER_PIXEL
        corr_at_aq0 = corr_at_frame - solar_motion
        corrections.append(corr_at_aq0)
        details.append({
            "name": name,
            "dtilt_deg": round(float(dtilt), 4),
            "dtilt_source": dtilt_source,
            "cx_px": round(cx, 1),
            "cy_px": round(cy, 1),
            "cy_expected_px": round(cy_expected, 1),
            "dy_px": round(dy, 1),
            "corr_at_frame_deg": round(corr_at_frame, 6),
            "solar_motion_deg": round(solar_motion, 6),
            "corr_at_aq0_deg": round(corr_at_aq0, 6),
        })

    return {
        "correction_deg": float(np.nanmean(corrections)) if corrections else np.nan,
        "applied_deg": 0.0,
        "n_used": len(corrections),
        "n_total": len(calib_names),
        "details": details,
    }


def sample_grid(grid, row, col):
    return float(map_coordinates(grid, [[row], [col]], order=1, mode="nearest")[0])


def edge_score(row, col, shape):
    distance = min(row, col, shape[0] - 1 - row, shape[1] - 1 - col)
    return float(np.clip(distance / 400, 0, 1)), float(distance)


def weighted_center(mask, weights):
    yy, xx = np.indices(mask.shape)
    w = np.where(mask, weights, 0)
    total = np.nansum(w)
    if not np.isfinite(total) or total <= 0:
        return None
    return float(np.nansum(yy * w) / total), float(np.nansum(xx * w) / total)


def robust_dolp_min(
        I,
        Q,
        U,
        view_zen,
        view_az,
        sun_zen,
        crop_mask=None,
        min_sun_zen_separation=1.0,
        sigma=2,
        low_percentile=1):
    I_s = gaussian_filter(I, sigma=sigma)
    Q_s = gaussian_filter(Q, sigma=sigma)
    U_s = gaussian_filter(U, sigma=sigma)

    dolp = np.sqrt(Q_s ** 2 + U_s ** 2) / np.maximum(np.abs(I_s), 1e-6)
    valid = np.isfinite(dolp) & (I_s > np.nanpercentile(I_s, 10))
    if crop_mask is not None:
        valid &= crop_mask
    valid[:IMG_EDGE_MARGIN, :] = False
    valid[-IMG_EDGE_MARGIN:, :] = False
    valid[:, :IMG_EDGE_MARGIN] = False
    valid[:, -IMG_EDGE_MARGIN:] = False

    if not np.any(valid):
        return None, dolp, valid, None

    threshold = np.nanpercentile(dolp[valid], low_percentile)
    low_mask = valid & (dolp <= threshold)
    weights = np.where(low_mask, 1 / np.maximum(dolp, 1e-6), 0)
    center = weighted_center(low_mask, weights)
    if center is None:
        return None, dolp, valid, low_mask

    row, col = center
    zen = sample_grid(view_zen, row, col)
    az = sample_grid(view_az, row, col)
    sun_zen_sep = abs(zen - sun_zen)
    edge_conf, edge_dist = edge_score(row, col, dolp.shape)

    low_count = int(np.count_nonzero(low_mask))
    compact_radius = float(np.sqrt(low_count / np.pi))
    p1 = float(threshold)
    p5 = float(np.nanpercentile(dolp[valid], 5))
    contrast = (p5 - p1) / max(p5, 1e-6)
    contrast_conf = float(np.clip(contrast, 0, 1))
    compact_conf = float(np.clip(120 / max(compact_radius, 1), 0, 1))
    sun_sep_conf = float(np.clip(sun_zen_sep / min_sun_zen_separation, 0, 1))
    confidence = float(
        (0.45 * edge_conf + 0.35 * contrast_conf + 0.20 * compact_conf)
        * sun_sep_conf)

    result = {
        "row": row,
        "col": col,
        "zen": zen,
        "az": az,
        "min_dolp": float(sample_grid(dolp, row, col)),
        "threshold": p1,
        "low_pixel_count": low_count,
        "compact_radius_px": compact_radius,
        "edge_distance_px": edge_dist,
        "sun_zen_separation_deg": float(sun_zen_sep),
        "sun_zen_separation_ok": bool(sun_zen_sep >= min_sun_zen_separation),
        "confidence": confidence,
    }
    return result, dolp, valid, low_mask


def zero_cross_mask(arr, valid):
    s = np.sign(arr)
    cross = np.zeros(arr.shape, dtype=bool)
    cross[:-1, :] |= (s[:-1, :] * s[1:, :] < 0)
    cross[1:, :] |= (s[:-1, :] * s[1:, :] < 0)
    cross[:, :-1] |= (s[:, :-1] * s[:, 1:] < 0)
    cross[:, 1:] |= (s[:, :-1] * s[:, 1:] < 0)
    return cross & valid


def robust_qu_zero_intersection(
        I,
        Q,
        U,
        view_zen,
        view_az,
        sun_zen,
        crop_mask=None,
        min_sun_zen_separation=1.0,
        sigma=2):
    I_s = gaussian_filter(I, sigma=sigma)
    q_s = gaussian_filter(Q / np.maximum(I, 1e-6), sigma=sigma)
    u_s = gaussian_filter(U / np.maximum(I, 1e-6), sigma=sigma)
    residual = np.sqrt(q_s ** 2 + u_s ** 2)

    valid = np.isfinite(q_s) & np.isfinite(u_s) & (I_s > np.nanpercentile(I_s, 10))
    if crop_mask is not None:
        valid &= crop_mask
    valid[:IMG_EDGE_MARGIN, :] = False
    valid[-IMG_EDGE_MARGIN:, :] = False
    valid[:, :IMG_EDGE_MARGIN] = False
    valid[:, -IMG_EDGE_MARGIN:] = False

    q_zero = zero_cross_mask(q_s, valid)
    u_zero = zero_cross_mask(u_s, valid)
    q_zero_count = int(np.count_nonzero(q_zero))
    u_zero_count = int(np.count_nonzero(u_zero))

    q_line_in_fov = q_zero_count >= 500
    u_line_in_fov = u_zero_count >= 500
    if not (q_line_in_fov and u_line_in_fov):
        return None, residual, np.zeros_like(valid), q_zero, u_zero

    q_zero_d = binary_dilation(q_zero, iterations=3)
    u_zero_d = binary_dilation(u_zero, iterations=3)
    overlap = valid & q_zero_d & u_zero_d

    # Require a real in-FOV intersection/neighborhood of the two zero lines.
    # If the dilated zero lines do not overlap, reject instead of extrapolating.
    if np.count_nonzero(overlap) < 5:
        return None, residual, overlap, q_zero, u_zero
    near = overlap

    residual_masked = np.where(near, residual, np.nan)
    if not np.any(np.isfinite(residual_masked)):
        return None, residual, near, q_zero, u_zero

    threshold = np.nanpercentile(residual_masked[np.isfinite(residual_masked)], 10)
    candidate = near & (residual <= threshold)
    weights = np.where(candidate, 1 / np.maximum(residual, 1e-7), 0)
    center = weighted_center(candidate, weights)
    if center is None:
        return None, residual, near, q_zero, u_zero

    row, col = center
    zen = sample_grid(view_zen, row, col)
    az = sample_grid(view_az, row, col)
    sun_zen_sep = abs(zen - sun_zen)
    edge_conf, edge_dist = edge_score(row, col, residual.shape)
    q_at = sample_grid(q_s, row, col)
    u_at = sample_grid(u_s, row, col)
    res_at = float(np.sqrt(q_at ** 2 + u_at ** 2))
    residual_conf = float(np.clip(1 - res_at / 0.02, 0, 1))
    cross_count = int(np.count_nonzero(near))
    cross_conf = float(np.clip(cross_count / 2000, 0, 1))
    sun_sep_conf = float(np.clip(sun_zen_sep / min_sun_zen_separation, 0, 1))
    confidence = float(
        (0.40 * edge_conf + 0.40 * residual_conf + 0.20 * cross_conf)
        * sun_sep_conf)

    result = {
        "row": row,
        "col": col,
        "zen": zen,
        "az": az,
        "q_at": q_at,
        "u_at": u_at,
        "residual": res_at,
        "candidate_pixel_count": int(np.count_nonzero(candidate)),
        "near_zero_pixel_count": cross_count,
        "q_zero_pixel_count": q_zero_count,
        "u_zero_pixel_count": u_zero_count,
        "q_line_in_fov": q_line_in_fov,
        "u_line_in_fov": u_line_in_fov,
        "edge_distance_px": edge_dist,
        "sun_zen_separation_deg": float(sun_zen_sep),
        "sun_zen_separation_ok": bool(sun_zen_sep >= min_sun_zen_separation),
        "confidence": confidence,
    }
    return result, residual, near, q_zero, u_zero


def agreement_confidence(dolp_result, zero_result):
    if dolp_result is None or zero_result is None:
        return 0.0, np.nan, np.nan
    drow = dolp_result["row"] - zero_result["row"]
    dcol = dolp_result["col"] - zero_result["col"]
    pixel_distance = float(np.sqrt(drow ** 2 + dcol ** 2))
    angle_distance = float(np.sqrt(
        (dolp_result["zen"] - zero_result["zen"]) ** 2
        + (dolp_result["az"] - zero_result["az"]) ** 2))
    confidence = float(np.exp(-pixel_distance / 250))
    return confidence, pixel_distance, angle_distance


def save_diagnostic(fig_dir, aq_name, dolp, low_mask, residual, near_zero, dolp_result, zero_result):
    with PLOT_LOCK:
        os.makedirs(fig_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(dolp * 100, cmap="hot", vmin=0, vmax=8)
        axes[0].contour(low_mask, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[0].set_title("Smoothed DoLP [%]")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(residual, cmap="magma", vmin=0, vmax=0.03)
        axes[1].contour(near_zero, levels=[0.5], colors="lime", linewidths=0.8)
        axes[1].set_title("Q/U Zero-Line Residual")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        if dolp_result is not None:
            for ax in axes:
                ax.scatter(dolp_result["col"], dolp_result["row"], marker="+", s=140, c="cyan", linewidths=2)
        if zero_result is not None:
            for ax in axes:
                ax.scatter(zero_result["col"], zero_result["row"], marker="x", s=120, c="lime", linewidths=2)

        fig.suptitle(aq_name)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{aq_name}_robust_np.png"), dpi=180)
        plt.close(fig)


def process_acquisition(path, aq_name, fig_dir, min_sun_zen_separation, calib_tilt):
    with h5py.File(path, "r") as handle:
        aq = handle[aq_name]
        uv = aq["UV Image Data"]
        require_level2_products(uv, aq_name)
        view_zen, view_az = load_view_geometry(uv)
        sun_zen = load_sun_zenith(uv, aq)
        I = uv["I_corrected"][:]
        Q = uv["Q_corrected"][:]
        U = uv["U_corrected"][:]
        validate_product_shapes(
            aq_name,
            {"I": I, "Q": Q, "U": U, "view_zen": view_zen, "view_az": view_az},
        )
        (I, Q, U, view_zen, view_az), crop_radius_px = apply_center_crop(
            I, Q, U, view_zen, view_az)
        crop_mask = circular_crop_mask_for_radius(I.shape, crop_radius_px)

        dolp_result, dolp, valid, low_mask = robust_dolp_min(
            I,
            Q,
            U,
            view_zen,
            view_az,
            sun_zen,
            crop_mask=crop_mask,
            min_sun_zen_separation=min_sun_zen_separation)
        zero_result, residual, near_zero, q_zero, u_zero = robust_qu_zero_intersection(
            I,
            Q,
            U,
            view_zen,
            view_az,
            sun_zen,
            crop_mask=crop_mask,
            min_sun_zen_separation=min_sun_zen_separation)
        agree_conf, pixel_dist, angle_dist = agreement_confidence(dolp_result, zero_result)

        dolp_conf = dolp_result["confidence"] if dolp_result else 0.0
        zero_conf = zero_result["confidence"] if zero_result else 0.0
        total_conf = float(0.40 * dolp_conf + 0.40 * zero_conf + 0.20 * agree_conf)

        row = {
            "acquisition": aq_name,
            "timestamp_utc": aq.attrs.get("Timestamp UTC", ""),
            "sun_altitude_attr_deg": float(aq.attrs["Sun Position Altitude"]),
            "sun_azimuth_attr_deg": float(aq.attrs["Sun Position Azimuth"]),
            "sun_zen_for_filter_deg": sun_zen,
            "min_sun_zen_separation_deg": min_sun_zen_separation,
            "circular_crop_radius_fraction": circular_crop_radius_fraction(),
            "circular_crop_radius_shrink_px": circular_crop_radius_shrink_px(),
            "calib_tilt_correction_deg": calib_tilt["correction_deg"],
            "calib_tilt_applied_deg": calib_tilt["applied_deg"],
            "calib_tilt_n_used": calib_tilt["n_used"],
            "calib_tilt_n_total": calib_tilt["n_total"],
            "calib_tilt_details_json": json.dumps(
                calib_tilt["details"],
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "dolp_row": dolp_result["row"] if dolp_result else np.nan,
            "dolp_col": dolp_result["col"] if dolp_result else np.nan,
            "dolp_zen_deg": dolp_result["zen"] if dolp_result else np.nan,
            "dolp_az_deg": dolp_result["az"] if dolp_result else np.nan,
            "dolp_min": dolp_result["min_dolp"] if dolp_result else np.nan,
            "dolp_sun_zen_separation_deg": dolp_result["sun_zen_separation_deg"] if dolp_result else np.nan,
            "dolp_sun_zen_separation_ok": dolp_result["sun_zen_separation_ok"] if dolp_result else False,
            "dolp_confidence": dolp_conf,
            "zero_row": zero_result["row"] if zero_result else np.nan,
            "zero_col": zero_result["col"] if zero_result else np.nan,
            "zero_zen_deg": zero_result["zen"] if zero_result else np.nan,
            "zero_az_deg": zero_result["az"] if zero_result else np.nan,
            "zero_residual": zero_result["residual"] if zero_result else np.nan,
            "zero_sun_zen_separation_deg": zero_result["sun_zen_separation_deg"] if zero_result else np.nan,
            "zero_sun_zen_separation_ok": zero_result["sun_zen_separation_ok"] if zero_result else False,
            "zero_confidence": zero_conf,
            "q_zero_pixel_count": zero_result["q_zero_pixel_count"] if zero_result else 0,
            "u_zero_pixel_count": zero_result["u_zero_pixel_count"] if zero_result else 0,
            "q_line_in_fov": zero_result["q_line_in_fov"] if zero_result else False,
            "u_line_in_fov": zero_result["u_line_in_fov"] if zero_result else False,
            "method_pixel_distance": pixel_dist,
            "method_angle_distance_deg": angle_dist,
            "agreement_confidence": agree_conf,
            "total_confidence": total_conf,
        }

        if os.environ.get("NP_SAVE_DIAGNOSTICS", "1") != "0":
            save_diagnostic(fig_dir, aq_name, dolp, low_mask, residual, near_zero, dolp_result, zero_result)
        return row


def worker_count(acquisition_count):
    requested = int(os.environ.get("NP_WORKERS", "0"))
    if requested > 0:
        return max(1, min(requested, acquisition_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count, acquisition_count))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python robust_np_all_acquisitions.py /path/to/file.h5")

    path = sys.argv[1]
    min_sun_zen_separation = float(os.environ.get("NP_MIN_SUN_ZEN_SEPARATION_DEG", "1.0"))
    out_csv = os.path.join(
        os.path.dirname(path),
        os.path.splitext(os.path.basename(path))[0] + "_robust_np_methods.csv")
    fig_dir = os.path.join(os.path.dirname(path), "robust_np_figures")

    with h5py.File(path, "r") as handle:
        names = acquisition_names(handle)
        if not names:
            raise RuntimeError(f"No Aquistion_* groups found in {path}")
        for aq_name in names:
            require_level2_products(handle[aq_name]["UV Image Data"], aq_name)
        calib_tilt = compute_calibration_tilt_correction(handle)

    rows = []
    workers = worker_count(len(names))
    print(f"Processing {len(names)} acquisitions with {workers} worker(s)", flush=True)
    if workers == 1:
        for aq_name in names:
            print(f"Processing {aq_name}", flush=True)
            rows.append(process_acquisition(path, aq_name, fig_dir, min_sun_zen_separation, calib_tilt))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_acquisition,
                    path,
                    aq_name,
                    fig_dir,
                    min_sun_zen_separation,
                    calib_tilt): aq_name
                for aq_name in names
            }
            for future in as_completed(futures):
                aq_name = futures[future]
                rows.append(future.result())
                print(f"Finished {aq_name}", flush=True)

    rows.sort(key=lambda row: row["total_confidence"], reverse=True)
    for idx, row in enumerate(rows):
        if idx == 0:
            row["selection_label"] = "BEST"
        elif row["total_confidence"] >= 0.65:
            row["selection_label"] = "POSSIBLE"
        else:
            row["selection_label"] = ""

    out_csv = write_rows_csv(out_csv, path, rows)

    print(f"Wrote {out_csv}")
    print(f"Figures: {fig_dir}")
    print(
        "Calibration tilt correction: "
        f"{calib_tilt['correction_deg']:.6g} deg "
        f"({calib_tilt['n_used']}/{calib_tilt['n_total']} calib frames), "
        f"applied={calib_tilt['applied_deg']:.6g} deg")
    print("Top acquisitions:")
    for row in rows[:8]:
        label = f"{row['selection_label']} " if row["selection_label"] else ""
        print(
            f"{label}{row['acquisition']} total={row['total_confidence']:.3f} "
            f"DoLP=({row['dolp_zen_deg']:.4f},{row['dolp_az_deg']:.4f}) "
            f"Zero=({row['zero_zen_deg']:.4f},{row['zero_az_deg']:.4f}) "
            f"angle_dist={row['method_angle_distance_deg']:.4f}")


if __name__ == "__main__":
    main()

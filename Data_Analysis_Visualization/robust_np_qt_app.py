import csv
import math
import os
import subprocess
import sys
from datetime import datetime

import h5py
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter


DEFAULT_CIRCULAR_CROP_RADIUS_FRACTION = 0.82
DEFAULT_CIRCULAR_CROP_RADIUS_SHRINK_PX = 100.0


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


def circular_crop_mask_for_radius(shape, radius):
    rows, cols = np.indices(shape)
    center_row = (shape[0] - 1) / 2.0
    center_col = (shape[1] - 1) / 2.0
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2


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


def apply_circular_crop(data, mask):
    cropped = np.asarray(data, dtype=np.float32).copy()
    cropped[~mask] = np.nan
    return cropped


def apply_center_circular_crop(*arrays):
    row_slice, col_slice, radius = circular_crop_slices(arrays[0].shape)
    cropped_arrays = [np.asarray(arr)[row_slice, col_slice] for arr in arrays]
    mask = circular_crop_mask_for_radius(cropped_arrays[0].shape, radius)
    return [apply_circular_crop(arr, mask) for arr in cropped_arrays], mask


def acquisition_names(handle):
    names = [
        name for name in handle.keys()
        if (name.startswith("Aquistion_") or name.startswith("calibration_acqui_"))
        and "UV Image Data" in handle[name]
    ]

    def sort_key(name):
        if name.startswith("Aquistion_"):
            return (0, int(name.split("_")[-1]))
        if name.startswith("calibration_acqui_"):
            parts = name.split("_")
            direction = parts[2] if len(parts) > 2 else ""
            amount = parts[3].replace("p", ".") if len(parts) > 3 else "0"
            try:
                amount_value = float(amount)
            except ValueError:
                amount_value = 0.0
            return (1, 0 if direction == "down" else 1, amount_value)
        return (2, name)

    return sorted(names, key=sort_key)


def default_csv_for_h5(h5_path):
    preferred = os.path.join(
        os.path.dirname(h5_path),
        os.path.splitext(os.path.basename(h5_path))[0] + "_robust_np_methods.csv")
    if os.path.exists(preferred):
        return preferred
    fallback = os.path.join(
        script_dir(),
        "robust_np_outputs",
        os.path.splitext(os.path.basename(h5_path))[0] + "_robust_np_methods.csv")
    return fallback if os.path.exists(fallback) else preferred


def read_np_table(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    return {row["acquisition"]: row for row in rows}


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def find_calibration_files():
    candidates = [
        script_dir(),
        os.path.join(os.path.dirname(script_dir()), "ULTRASIP-Neutral-Points", "Data_Analysis_Visualization"),
        os.path.join(os.getcwd(), "ULTRASIP-Neutral-Points", "Data_Analysis_Visualization"),
    ]
    for directory in candidates:
        nuc_path = os.path.join(directory, "NUC_0813.npz")
        wmatrix_path = os.path.join(directory, "ULTRASIP_AvgWmatrix_15.npy")
        if os.path.exists(nuc_path) and os.path.exists(wmatrix_path):
            return nuc_path, wmatrix_path
    return None, None


def has_level2_products(handle):
    names = acquisition_names(handle)
    if not names:
        return False
    for aq_name in names:
        uv = handle[aq_name]["UV Image Data"]
        has_stokes = all(name in uv for name in ["I_corrected", "Q_corrected", "U_corrected"])
        has_angles = (
            ("view_zen_corrected" in uv and "view_az_corrected" in uv)
            or ("view_zen" in uv and "view_az" in uv)
        )
        if has_stokes and has_angles:
            return True
    return False


def h5_has_level2_products(path):
    with h5py.File(path, "r") as handle:
        return has_level2_products(handle)


def as_float(row, key):
    try:
        return float(row.get(key, np.nan))
    except Exception:
        return np.nan


def zero_crossing_1d(values, coords, preferred_index=None):
    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(coords)
    candidates = []

    for idx in np.where(finite)[0]:
        if values[idx] == 0:
            candidates.append((float(coords[idx]), float(idx), 0.0, True))

    valid_idx = np.where(finite)[0]
    for left, right in zip(valid_idx[:-1], valid_idx[1:]):
        v0 = values[left]
        v1 = values[right]
        if v0 == 0 or v1 == 0 or v0 * v1 > 0:
            continue
        frac = -v0 / (v1 - v0)
        coord = coords[left] + frac * (coords[right] - coords[left])
        index = left + frac * (right - left)
        candidates.append((float(coord), float(index), 0.0, True))

    if candidates:
        if preferred_index is None:
            coord, index, residual, crossed = candidates[0]
        else:
            coord, index, residual, crossed = min(
                candidates,
                key=lambda item: abs(item[1] - preferred_index))
        return {
            "coord": coord,
            "index": index,
            "residual": residual,
            "crossed": crossed,
        }

    if not np.any(finite):
        return None

    finite_indices = np.where(finite)[0]
    best = finite_indices[np.nanargmin(np.abs(values[finite]))]
    return {
        "coord": float(coords[best]),
        "index": float(best),
        "residual": float(abs(values[best])),
        "crossed": False,
    }


def fit_average_window(coord_values, stokes_values, start, stop):
    xw = np.asarray(stokes_values[start:stop], dtype=np.float64)
    yw = np.asarray(coord_values[start:stop], dtype=np.float64)
    finite = np.isfinite(xw) & np.isfinite(yw)
    if np.count_nonzero(finite) < 20:
        return None

    xw = xw[finite]
    yw = yw[finite]
    design = np.column_stack([np.ones_like(xw), xw])
    try:
        beta, *_ = np.linalg.lstsq(design, yw, rcond=None)
    except np.linalg.LinAlgError:
        return None

    fitted = design @ beta
    residual = yw - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((yw - np.mean(yw)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(len(yw) - 2, 1)
    sigma2 = ss_res / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
        intercept_stderr = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        intercept_stderr = np.inf

    intercept = float(beta[0])
    coord_min = float(np.nanmin(yw))
    coord_max = float(np.nanmax(yw))
    outside = max(coord_min - intercept, 0.0, intercept - coord_max)
    zero_gap = 0.0
    if np.nanmin(xw) > 0:
        zero_gap = float(np.nanmin(xw))
    elif np.nanmax(xw) < 0:
        zero_gap = float(abs(np.nanmax(xw)))

    stderr_arcsec = intercept_stderr * 3600.0
    score = r2 - stderr_arcsec / 1000.0 - outside * 2.0 - zero_gap * 10.0
    return {
        "start": start,
        "stop": stop,
        "intercept": intercept,
        "stderr_arcsec": stderr_arcsec,
        "r2": float(r2),
        "slope": float(beta[1]),
        "score": float(score),
        "outside_deg": float(outside),
        "zero_gap": zero_gap,
        "n": int(len(yw)),
    }


def best_average_crop(coord_values, stokes_values):
    best = None
    edge_margin = 100
    widths = [200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200]
    for width in widths:
        if len(stokes_values) < width + 2 * edge_margin:
            continue
        step = 5 if width <= 400 else 10
        for start in range(edge_margin, len(stokes_values) - width - edge_margin + 1, step):
            candidate = fit_average_window(coord_values, stokes_values, start, start + width)
            if candidate is None:
                continue
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


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


def rotate_qu(q, u, theta_deg):
    """Rotate Q/I, U/I in Stokes space by theta using R(theta) = [[cos,-sin],[sin,cos]]."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    q_rot = c * q + s * u
    u_rot = -s * q + c * u
    return q_rot, u_rot


def load_products(handle, aq_name, sigma):
    aq = handle[aq_name]
    uv = aq["UV Image Data"]

    I = uv["I_corrected"][:]
    Q = uv["Q_corrected"][:]
    U = uv["U_corrected"][:]
    view_zen, view_az = load_view_geometry(uv)
    validate_product_shapes(
        aq_name,
        {"I": I, "Q": Q, "U": U, "view_zen": view_zen, "view_az": view_az},
    )

    I_s = gaussian_filter(I, sigma=sigma)
    Q_s = gaussian_filter(Q, sigma=sigma)
    U_s = gaussian_filter(U, sigma=sigma)
    q = Q_s / np.maximum(I_s, 1e-6)
    u = U_s / np.maximum(I_s, 1e-6)
    dolp = np.sqrt(Q_s ** 2 + U_s ** 2) / np.maximum(np.abs(I_s), 1e-6)
    (I_s, q, u, dolp, view_zen, view_az), mask = apply_center_circular_crop(
        I_s, q, u, dolp, view_zen, view_az)

    return {
        "aq": aq,
        "I": I_s,
        "q": q,
        "u": u,
        "dolp": dolp,
        "log_dolp": np.log10(np.maximum(dolp, 1e-6)),
        "view_zen": view_zen,
        "view_az": view_az,
        "crop_mask": mask,
        "crop_radius_fraction": circular_crop_radius_fraction(),
        "crop_radius_shrink_px": circular_crop_radius_shrink_px(),
    }


def label_angular_image_axes(ax, view_zen, view_az):
    center_row = view_zen.shape[0] // 2
    center_col = view_zen.shape[1] // 2

    def az_formatter(x_value, _position):
        col = int(np.clip(round(x_value), 0, view_az.shape[1] - 1))
        return f"{nearest_finite_1d(view_az[:, col], center_row):.1f}"

    def zen_formatter(y_value, _position):
        row = int(np.clip(round(y_value), 0, view_zen.shape[0] - 1))
        return f"{nearest_finite_1d(view_zen[row, :], center_col):.1f}"

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Zenith [deg]")
    ax.xaxis.set_major_formatter(FuncFormatter(az_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(zen_formatter))
    ax.set_ylim(-0.5, view_zen.shape[0] - 0.5)
    ax.tick_params(labelsize=7)


def nearest_finite_1d(values, index):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    index = int(np.clip(round(index), 0, len(values) - 1))
    if finite[index]:
        return float(values[index])
    finite_indices = np.flatnonzero(finite)
    nearest = finite_indices[np.argmin(np.abs(finite_indices - index))]
    return float(values[nearest])


def label_raw_image_axes(ax, view_zen, view_az):
    center_row = view_zen.shape[0] // 2
    center_col = view_zen.shape[1] // 2

    def az_formatter(x_value, _position):
        col = int(np.clip(round(x_value), 0, view_az.shape[1] - 1))
        return f"{nearest_finite_1d(view_az[:, col], center_row):.1f}"

    def zen_formatter(y_value, _position):
        row = int(np.clip(round(y_value), 0, view_zen.shape[0] - 1))
        return f"{nearest_finite_1d(view_zen[row, :], center_col):.1f}"

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Zenith [deg]")
    ax.xaxis.set_major_formatter(FuncFormatter(az_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(zen_formatter))
    ax.tick_params(labelsize=7)


def label_flipped_image_axes(ax, view_zen, view_az):
    center_row = view_zen.shape[0] // 2
    center_col = view_zen.shape[1] // 2

    def az_formatter(x_value, _position):
        col = int(np.clip(round(x_value), 0, view_az.shape[1] - 1))
        return f"{nearest_finite_1d(view_az[:, col], center_row):.1f}"

    def zen_formatter(y_value, _position):
        display_row = int(np.clip(round(y_value), 0, view_zen.shape[0] - 1))
        source_row = view_zen.shape[0] - 1 - display_row
        return f"{nearest_finite_1d(view_zen[source_row, :], center_col):.1f}"

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Zenith [deg]")
    ax.xaxis.set_major_formatter(FuncFormatter(az_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(zen_formatter))
    ax.tick_params(labelsize=7)


def angular_extent(view_zen, view_az):
    return [
        float(np.nanmin(view_az)),
        float(np.nanmax(view_az)),
        float(np.nanmin(view_zen)),
        float(np.nanmax(view_zen)),
    ]


def pixel_to_angle(row, col, view_zen, view_az):
    row_idx = int(np.clip(round(row), 0, view_zen.shape[0] - 1))
    col_idx = int(np.clip(round(col), 0, view_zen.shape[1] - 1))
    return float(view_az[row_idx, col_idx]), float(view_zen[row_idx, col_idx])


def display_np_points(row, view_zen, view_az):
    points = {}
    dolp_row = as_float(row, "dolp_row")
    dolp_col = as_float(row, "dolp_col")
    if np.isfinite(dolp_row) and np.isfinite(dolp_col):
        az, zen = pixel_to_angle(dolp_row, dolp_col, view_zen, view_az)
        if not np.isfinite(az):
            az = as_float(row, "dolp_az_deg")
        if not np.isfinite(zen):
            zen = as_float(row, "dolp_zen_deg")
        points["dolp"] = {
            "row": dolp_row,
            "col": dolp_col,
            "zen": zen,
            "az": az,
        }

    zero_row = as_float(row, "zero_row")
    zero_col = as_float(row, "zero_col")
    if np.isfinite(zero_row) and np.isfinite(zero_col):
        az, zen = pixel_to_angle(zero_row, zero_col, view_zen, view_az)
        if not np.isfinite(az):
            az = as_float(row, "zero_az_deg")
        if not np.isfinite(zen):
            zen = as_float(row, "zero_zen_deg")
        points["zero"] = {
            "row": zero_row,
            "col": zero_col,
            "zen": zen,
            "az": az,
        }
    return points


def center_priority_weights(shape, step, mode="center"):
    rows = np.arange(0, shape[0], step, dtype=np.float32)
    cols = np.arange(0, shape[1], step, dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    if mode == "upper":
        row_weight = np.where(rr <= (shape[0] - 1) / 2, 1.0, (shape[0] - 1 - rr) / max((shape[0] - 1) / 2, 1))
        row_weight = np.clip(row_weight, 0.0, 1.0)
    elif mode == "lower":
        row_weight = np.where(rr >= (shape[0] - 1) / 2, 1.0, rr / max((shape[0] - 1) / 2, 1))
        row_weight = np.clip(row_weight, 0.0, 1.0)
    else:
        row_norm = np.abs((rr - (shape[0] - 1) / 2) / max((shape[0] - 1) / 2, 1))
        row_weight = np.where(row_norm <= 0.5, 1.0, np.clip((1.0 - row_norm) / 0.5, 0.0, 1.0))

    col_norm = np.abs((cc - (shape[1] - 1) / 2) / max((shape[1] - 1) / 2, 1))
    col_weight = np.where(col_norm <= 0.5, 1.0, np.clip((1.0 - col_norm) / 0.5, 0.0, 1.0))
    return np.minimum(row_weight, col_weight)


def angular_step_from_grid(view_zen, view_az, step):
    az_sample = view_az[::step, ::step]
    zen_sample = view_zen[::step, ::step]
    az_delta = np.abs(np.diff(az_sample, axis=1))
    zen_delta = np.abs(np.diff(zen_sample, axis=0))
    deltas = np.concatenate([
        az_delta[np.isfinite(az_delta)].ravel(),
        zen_delta[np.isfinite(zen_delta)].ravel(),
    ])
    deltas = deltas[(deltas > 1e-4) & (deltas < 5)]
    if deltas.size == 0:
        return 0.05
    return float(np.clip(np.nanmedian(deltas), 0.01, 0.25))


def add_to_panorama_blend(value_sum, weight_sum, row_idx, col_idx, values, weights):
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
        & (row_idx >= 0)
        & (row_idx < weight_sum.shape[0])
        & (col_idx >= 0)
        & (col_idx < weight_sum.shape[1])
    )
    if not np.any(valid):
        return
    rr = row_idx[valid]
    cc = col_idx[valid]
    vv = values[valid]
    ww = weights[valid]
    np.add.at(value_sum, (rr, cc), vv * ww)
    np.add.at(weight_sum, (rr, cc), ww)


def blended_panorama(value_sum, weight_sum):
    with np.errstate(invalid="ignore", divide="ignore"):
        blended = value_sum / weight_sum
    blended[weight_sum <= 0] = np.nan
    return blended.astype(np.float32)


def nan_gaussian_filter(data, sigma):
    if sigma <= 0:
        return data.copy()
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    weight = gaussian_filter(valid.astype(np.float32), sigma=sigma)
    smoothed = gaussian_filter(filled, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / weight
    smoothed[weight < 1e-3] = np.nan
    return smoothed.astype(np.float32)


def roughness_metric(data):
    gy, gx = np.gradient(data.astype(np.float64))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.nanmedian(mag))


_UV_ARCSEC_PER_PIXEL = 7.20
_UV_DEG_PER_PIXEL = _UV_ARCSEC_PER_PIXEL / 3600.0
_SUN_COMPLETENESS_MARGIN_PX = 220  # ~solar radius (133 px) + halo margin


def _solar_altitude_deg(dt, lat_deg, lon_deg):
    """Return sun altitude in degrees for a timezone-aware datetime."""
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


def _detect_sun_center_raw(frame):
    """Return (cx, cy) of sun centroid in raw UV image, or None if not found."""
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


def _sun_is_complete(cx, cy, width, height):
    """True if sun disk is fully within the frame (center is far from edges)."""
    m = _SUN_COMPLETENESS_MARGIN_PX
    return m <= cx <= width - m and m <= cy <= height - m


def _finite_attr(attrs, key):
    try:
        value = float(attrs.get(key, np.nan))
    except Exception:
        value = np.nan
    return value if np.isfinite(value) else np.nan


def _parse_commanded_calibration_dtilt(name, attrs):
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


def _calibration_dtilt_from_attrs(name, aq0_attrs, attrs):
    for key, source in (
        ("Moog Post Capture Actual Tilt [deg]", "post-capture actual tilt"),
        ("Moog Actual Tilt [deg]", "actual tilt"),
    ):
        base = _finite_attr(aq0_attrs, key)
        current = _finite_attr(attrs, key)
        if np.isfinite(base) and np.isfinite(current):
            return current - base, source
    return _parse_commanded_calibration_dtilt(name, attrs)


def compute_calibration_tilt_correction(handle):
    """
    Estimate tilt pointing correction from calibration acquisitions' raw UV images.

    Each calibration frame was captured with a known tilt offset (dtilt) relative to
    the initial sun altitude. By detecting the actual sun centre in the raw image and
    comparing it to the expected position, we derive the tilt error at each frame.
    Solar altitude change between Acquisition_0 and each calibration frame is removed
    so all corrections are referred back to the Acquisition_0 epoch, then averaged.

    Sign convention (positive correction):
      Sun appeared BELOW the expected position → instrument pointed too high
      → view_zen values are too large → subtract correction from view_zen

    Returns a dict with:
      correction_deg  – mean correction in degrees (nan if no usable frames)
      n_used          – number of frames that passed the completeness check
      details         – list of per-frame dicts for diagnostics
    """
    meta = handle.get("Measurement_Metadata")
    if meta is None:
        return None
    try:
        lat = float(meta.attrs.get("Latitude", 0.0))
        lon = float(meta.attrs.get("Longitude", 0.0))
    except Exception:
        return None

    # Reference time and solar altitude at Acquisition_0
    aq0 = handle.get("Aquistion_0")
    if aq0 is None:
        return None
    try:
        t0_str = str(aq0.attrs.get("Timestamp Local New")
                     or aq0.attrs.get("Timestamp UTC New", ""))
        t0 = datetime.fromisoformat(t0_str.replace("Z", "+00:00"))
        alt0 = _solar_altitude_deg(t0, lat, lon)
    except Exception:
        return None

    px_per_deg = 1.0 / _UV_DEG_PER_PIXEL
    calib_names = sorted(n for n in handle.keys() if n.startswith("calibration_acqui_"))
    corrections = []
    details = []

    for name in calib_names:
        grp = handle[name]
        uv = grp.get("UV Image Data")
        if uv is None:
            details.append({"name": name, "skip": "no UV Image Data"})
            continue
        raw_ds = uv.get("UV Raw Images")
        if raw_ds is None:
            details.append({"name": name, "skip": "no UV Raw Images"})
            continue
        raw = raw_ds[:]
        frame = raw[0] if raw.ndim == 3 else raw if raw.ndim == 2 else None
        if frame is None:
            details.append({"name": name, "skip": "unexpected raw shape"})
            continue

        h, w = frame.shape
        cy0, cx0 = h / 2.0, w / 2.0

        result = _detect_sun_center_raw(frame)
        if result is None:
            details.append({"name": name, "skip": "sun not detected"})
            continue
        cx, cy = result

        if not _sun_is_complete(cx, cy, w, h):
            details.append({"name": name, "skip": "sun incomplete",
                             "cx": round(cx, 1), "cy": round(cy, 1)})
            continue

        dtilt, dtilt_source = _calibration_dtilt_from_attrs(name, aq0.attrs, grp.attrs)
        if not np.isfinite(dtilt):
            details.append({"name": name, "skip": "cannot determine dtilt"})
            continue

        # Solar motion correction: altitude change from Acquisition_0 to this frame
        try:
            ti_str = str(grp.attrs.get("Timestamp Local New")
                         or grp.attrs.get("Timestamp UTC New", ""))
            t_i = datetime.fromisoformat(ti_str.replace("Z", "+00:00"))
            solar_motion = _solar_altitude_deg(t_i, lat, lon) - alt0
        except Exception:
            solar_motion = 0.0

        # Expected sun row if pointing is perfect at dtilt:
        #   dtilt>0 means tilted UP from sun → sun moves DOWN (larger row) → cy_expected = cy0 + dtilt*px_per_deg
        cy_expected = cy0 + dtilt * px_per_deg
        dy = cy - cy_expected  # positive: sun below expected → instrument pointed too high

        # Tilt error at this frame epoch (positive = instrument over-tilted = pointed too high in altitude)
        corr_at_frame = dy * _UV_DEG_PER_PIXEL

        # Refer correction back to Acquisition_0 epoch (subtract solar altitude rise from t0→ti)
        corr_at_aq0 = corr_at_frame - solar_motion

        corrections.append(corr_at_aq0)
        details.append({
            "name": name,
            "dtilt_cmd_deg": round(dtilt, 3),
            "dtilt_source": dtilt_source,
            "cx_px": round(cx, 1),
            "cy_px": round(cy, 1),
            "cy_expected_px": round(cy_expected, 1),
            "dy_px": round(dy, 1),
            "corr_at_frame_deg": round(corr_at_frame, 4),
            "solar_motion_deg": round(solar_motion, 4),
            "corr_at_aq0_deg": round(corr_at_aq0, 4),
        })

    correction_deg = float(np.nanmean(corrections)) if corrections else np.nan
    return {
        "correction_deg": correction_deg,
        "n_used": len(corrections),
        "n_total": len(calib_names),
        "details": details,
    }


class RobustNPQtApp(QMainWindow):
    def __init__(self, initial_h5=None, initial_index=0):
        super().__init__()
        self.setWindowTitle("ULTRASIP Robust Neutral Point Viewer")
        self.resize(1900, 1200)

        self.h5_path = None
        self.handle = None
        self.names = []
        self.table = {}
        self.best_dolp_acquisition = None
        self.best_zero_acquisition = None
        self.best_avg_acquisition = None
        self.avg_candidates = {}
        self.cache = {}
        self.panorama_cache = None
        self.nuc_path = None
        self.wmatrix_path = None
        self.initial_index = initial_index
        self.tilt_correction_result = None

        self.open_button = QPushButton("Open H5")
        self.run_button = QPushButton("Run robust analysis")
        self.copy_jpg_button = QPushButton("Copy JPG 200dpi")
        self.calibration_button = QPushButton("Choose calibration")
        self.calibration_label = QLabel("Auto calibration")
        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.combo = QComboBox()
        self.sza_reference_combo = QComboBox()
        self.frame_display_combo = QComboBox()
        self.frame_display_combo.addItem("Raw row order", userData="raw")
        self.frame_display_combo.addItem("Zenith top", userData="zenith")
        self.stitch_priority_combo = QComboBox()
        self.stitch_priority_combo.addItem("Center 1/2", userData="center")
        self.stitch_priority_combo.addItem("Upper 1/2", userData="upper")
        self.stitch_priority_combo.addItem("Lower 1/2", userData="lower")
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.5, 8.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setValue(2.0)
        self.sigma_spin.setSuffix(" px")
        auto_nuc, auto_wmatrix = find_calibration_files()
        if auto_nuc and auto_wmatrix:
            self.calibration_label.setText(
                f"Auto: {os.path.basename(auto_nuc)} + {os.path.basename(auto_wmatrix)}")

        self.apply_tilt_corr_check = QCheckBox("Apply tilt corr")
        self.apply_tilt_corr_check.setToolTip(
            "Subtract the calibration-derived tilt correction from view_zen for all acquisitions.\n"
            "Positive correction = instrument pointed too high → zenith values decrease.")
        self.tilt_corr_label = QLabel("Tilt corr: ---")
        self.tilt_corr_label.setMinimumWidth(130)

        self.apply_roll_rotation_check = QCheckBox("Apply v2 (roll + zenith/az)")
        self.apply_roll_rotation_check.setToolTip(
            "Apply the per-frame v2 pointing calibration (Pointing_Calibration_v2):\n"
            " - Rotate Q/I and U/I in Stokes space by theta = 'Camera Roll v2 [deg]'\n"
            "   using R(theta) = [[cos, -sin], [sin, cos]]\n"
            " - Shift view_zen by 'Tilt Error v2 Pred [deg]' and view_az by\n"
            "   'Pan_v2 [deg]' - 'Geometry Pan Used [deg]'")
        self.roll_rotation_label = QLabel("Roll: ---")
        self.roll_rotation_label.setMinimumWidth(110)

        self.apply_v3_check = QCheckBox("Apply v3 (file calib only)")
        self.apply_v3_check.setToolTip(
            "Apply the per-file v3 pointing calibration (this file's own calibration\n"
            "frames only, no cross-file sinusoid):\n"
            " - Shift view_zen by 'Tilt Error v3 Pred [deg]'\n"
            " - No Q/U rotation and no azimuth shift (camera_roll_v3 = 0,\n"
            "   pan_v3 = pan_raw)")
        self.v3_label = QLabel("v3: ---")
        self.v3_label.setMinimumWidth(110)

        controls_row1 = QHBoxLayout()
        controls_row1.addWidget(self.open_button)
        controls_row1.addWidget(self.run_button)
        controls_row1.addWidget(self.copy_jpg_button)
        controls_row1.addWidget(self.calibration_button)
        controls_row1.addWidget(self.calibration_label)
        controls_row1.addSpacing(12)
        controls_row1.addWidget(QLabel("Acquisition"))
        controls_row1.addWidget(self.combo, stretch=1)
        controls_row1.addWidget(self.prev_button)
        controls_row1.addWidget(self.next_button)
        controls_row1.addSpacing(12)
        controls_row1.addWidget(QLabel("NP zenith ref"))
        controls_row1.addWidget(self.sza_reference_combo)

        controls_row2 = QHBoxLayout()
        controls_row2.addWidget(QLabel("1x4 display"))
        controls_row2.addWidget(self.frame_display_combo)
        controls_row2.addSpacing(12)
        controls_row2.addWidget(QLabel("Stitch priority"))
        controls_row2.addWidget(self.stitch_priority_combo)
        controls_row2.addSpacing(12)
        controls_row2.addWidget(QLabel("Smoothing sigma"))
        controls_row2.addWidget(self.sigma_spin)
        controls_row2.addSpacing(12)
        controls_row2.addWidget(self.tilt_corr_label)
        controls_row2.addWidget(self.apply_tilt_corr_check)
        controls_row2.addSpacing(12)
        controls_row2.addWidget(self.roll_rotation_label)
        controls_row2.addWidget(self.apply_roll_rotation_check)
        controls_row2.addSpacing(12)
        controls_row2.addWidget(self.v3_label)
        controls_row2.addWidget(self.apply_v3_check)
        controls_row2.addStretch(1)

        self.figure = Figure(figsize=(19, 11), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout()
        layout.addLayout(controls_row1)
        layout.addLayout(controls_row2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        self.open_button.clicked.connect(self.open_h5)
        self.run_button.clicked.connect(self.run_analysis)
        self.copy_jpg_button.clicked.connect(self.copy_jpg_200dpi)
        self.calibration_button.clicked.connect(self.choose_calibration_files)
        self.prev_button.clicked.connect(self.prev_acquisition)
        self.next_button.clicked.connect(self.next_acquisition)
        self.combo.currentIndexChanged.connect(self.redraw)
        self.sza_reference_combo.currentIndexChanged.connect(self.redraw)
        self.frame_display_combo.currentIndexChanged.connect(self.redraw)
        self.stitch_priority_combo.currentIndexChanged.connect(self.clear_panorama_and_redraw)
        self.sigma_spin.valueChanged.connect(self.clear_cache_and_redraw)
        self.apply_tilt_corr_check.stateChanged.connect(self.clear_cache_and_redraw)
        self.apply_roll_rotation_check.stateChanged.connect(self.clear_cache_and_redraw)
        self.apply_roll_rotation_check.toggled.connect(self._on_v2_toggled)
        self.apply_v3_check.stateChanged.connect(self.clear_cache_and_redraw)
        self.apply_v3_check.toggled.connect(self._on_v3_toggled)

        if initial_h5:
            self.load_h5(initial_h5)

    def closeEvent(self, event):
        if self.handle is not None:
            self.handle.close()
        event.accept()

    def open_h5(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open ULTRASIP H5 file",
            os.path.expanduser("~/Documents"),
            "HDF5 files (*.h5 *.hdf5);;All files (*.*)",
        )
        if path:
            self.load_h5(path)

    def choose_calibration_files(self):
        nuc_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select NUC calibration file",
            os.path.expanduser("~/Documents"),
            "NumPy archives (*.npz);;All files (*.*)",
        )
        if not nuc_path:
            return

        wmatrix_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select W matrix calibration file",
            os.path.dirname(nuc_path),
            "NumPy arrays (*.npy);;All files (*.*)",
        )
        if not wmatrix_path:
            return

        self.nuc_path = nuc_path
        self.wmatrix_path = wmatrix_path
        self.calibration_label.setText(
            f"{os.path.basename(nuc_path)} + {os.path.basename(wmatrix_path)}")

    def copy_jpg_200dpi(self):
        if self.handle is None:
            QMessageBox.warning(self, "Nothing to copy", "Open an H5 file first.")
            return

        out_path = os.path.join("/private/tmp", "robust_np_viewer_200dpi.jpg")
        try:
            self.canvas.draw()
            self.figure.savefig(out_path, dpi=200, format="jpg", bbox_inches="tight")
            image = QImage(out_path)
            if image.isNull():
                raise RuntimeError("Qt could not read the generated JPG.")
            QApplication.clipboard().setImage(image)
        except Exception as exc:
            QMessageBox.critical(self, "Copy JPG failed", str(exc))
            return

        self.copy_jpg_button.setText("Copied JPG 200dpi")
        self.statusBar().showMessage(f"Copied 200 dpi JPG to clipboard: {out_path}", 5000)

    def load_h5(self, path):
        if self.handle is not None:
            self.handle.close()
        self.h5_path = path
        self.handle = h5py.File(path, "r")
        self.names = acquisition_names(self.handle)
        if not self.names:
            QMessageBox.warning(self, "No acquisitions", "No Aquistion_* groups found.")
            return

        self.table = read_np_table(default_csv_for_h5(path))
        self.best_dolp_acquisition = self.best_method_acquisition("dolp")
        self.best_zero_acquisition = self.best_method_acquisition("zero")
        self.cache = {}
        self.panorama_cache = None
        self.tilt_correction_result = compute_calibration_tilt_correction(self.handle)
        self._update_tilt_corr_label()
        level2_ready = has_level2_products(self.handle)
        self.avg_candidates = self.average_qu_candidates() if level2_ready else {}
        self.best_avg_acquisition = (
            max(self.avg_candidates.values(), key=lambda row: row["combined_score"])["acquisition"]
            if self.avg_candidates else None
        )
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in self.names:
            label = self.acquisition_display_label(name)
            text = f"{label} {name}".strip()
            self.combo.addItem(text, userData=name)
        self.combo.setCurrentIndex(int(np.clip(self.initial_index, 0, len(self.names) - 1)))
        self.initial_index = 0
        self.combo.blockSignals(False)
        self.populate_sza_reference_combo()
        if level2_ready:
            self.redraw()
        else:
            self.show_unprocessed_file_notice()

    def run_analysis(self):
        if not self.h5_path:
            self.open_h5()
            if not self.h5_path:
                return
        if self.handle is not None:
            self.handle.close()
            self.handle = None

        env = os.environ.copy()
        mpl_dir = os.path.join("/private/tmp", "ultrasip_matplotlib")
        os.makedirs(mpl_dir, exist_ok=True)
        env.setdefault("MPLCONFIGDIR", mpl_dir)
        env.setdefault("NP_MIN_SUN_ZEN_SEPARATION_DEG", "1.0")
        env.setdefault("NP_SAVE_DIAGNOSTICS", "0")

        commands = []
        try:
            if not h5_has_level2_products(self.h5_path):
                process_script = os.path.join(script_dir(), "process_single_level0_1_2.py")
                if self.nuc_path and self.wmatrix_path:
                    nuc_path, wmatrix_path = self.nuc_path, self.wmatrix_path
                    if not os.path.exists(nuc_path) or not os.path.exists(wmatrix_path):
                        QMessageBox.critical(
                            self,
                            "Calibration missing",
                            "Selected calibration file does not exist.")
                        self.load_h5(self.h5_path)
                        return
                else:
                    nuc_path, wmatrix_path = find_calibration_files()
                process_command = [sys.executable, process_script, self.h5_path]
                if nuc_path and wmatrix_path:
                    process_command.extend([nuc_path, wmatrix_path])
                commands.append(process_command)

            robust_script = os.path.join(script_dir(), "robust_np_all_acquisitions.py")
            commands.append([sys.executable, robust_script, self.h5_path])

            for command in commands:
                result = subprocess.run(
                    command,
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            output = "\n".join(
                part for part in [
                    "Command failed:",
                    " ".join(exc.cmd),
                    "",
                    exc.stdout or "",
                    exc.stderr or "",
                ]
                if part)
            QMessageBox.critical(self, "Analysis failed", output[-5000:])
            self.load_h5(self.h5_path)
            return
        try:
            self.load_h5(self.h5_path)
        except Exception as exc:
            QMessageBox.critical(self, "Reload failed", str(exc))
            return

    def show_unprocessed_file_notice(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "This H5 has raw UV images only.\nClick Run robust analysis to generate Level 0/1/2 products first.",
            ha="center",
            va="center",
            fontsize=16)
        ax.text(
            0.5,
            0.43,
            os.path.basename(self.h5_path),
            ha="center",
            va="center",
            fontsize=11,
            family="monospace")
        self.canvas.draw_idle()

    def selection_label(self, aq_name):
        row = self.table.get(aq_name, {})
        label = row.get("selection_label", "")
        if label:
            return label
        ranked = sorted(
            self.table.values(),
            key=lambda row: as_float(row, "total_confidence"),
            reverse=True,
        )
        for idx, row in enumerate(ranked):
            if row.get("acquisition") != aq_name:
                continue
            if idx == 0:
                return "BEST"
            if as_float(row, "total_confidence") >= 0.65:
                return "POSSIBLE"
        return ""

    def best_method_acquisition(self, method):
        row_key = f"{method}_row"
        col_key = f"{method}_col"
        confidence_key = f"{method}_confidence"
        rows = [
            row for row in self.table.values()
            if np.isfinite(as_float(row, row_key))
            and np.isfinite(as_float(row, col_key))
            and np.isfinite(as_float(row, confidence_key))
        ]
        if not rows:
            return None
        return max(rows, key=lambda row: as_float(row, confidence_key)).get("acquisition")

    def average_qu_candidates(self):
        candidates = {}
        for aq_name in self.names:
            if not aq_name.startswith("Aquistion_"):
                continue
            try:
                products = self.get_products(aq_name)
            except Exception:
                continue
            avg_q = np.nanmean(products["q"], axis=1)
            avg_q_zen = np.nanmean(products["view_zen"], axis=1)
            avg_u = np.nanmean(products["u"], axis=0)
            avg_u_az = np.nanmean(products["view_az"], axis=0)
            q_best = best_average_crop(avg_q_zen, avg_q)
            u_best = best_average_crop(avg_u_az, avg_u)
            if q_best is None or u_best is None:
                continue
            candidates[aq_name] = {
                "acquisition": aq_name,
                "avg_zen_deg": q_best["intercept"],
                "avg_az_deg": u_best["intercept"],
                "q_score": q_best["score"],
                "u_score": u_best["score"],
                "combined_score": q_best["score"] + u_best["score"],
                "q_r2": q_best["r2"],
                "u_r2": u_best["r2"],
                "q_error_arcsec": q_best["stderr_arcsec"],
                "u_error_arcsec": u_best["stderr_arcsec"],
                "q_crop": f"{q_best['start']}:{q_best['stop']}",
                "u_crop": f"{u_best['start']}:{u_best['stop']}",
            }
        return candidates

    def method_best_labels(self, aq_name):
        labels = []
        if aq_name == self.best_dolp_acquisition:
            labels.append("BEST DoLP")
        if aq_name == self.best_zero_acquisition:
            labels.append("BEST Q/U")
        if aq_name == self.best_avg_acquisition:
            labels.append("BEST Avg Q/U")
        return labels

    def acquisition_display_label(self, aq_name):
        labels = []
        selection = self.selection_label(aq_name)
        if selection:
            labels.append(selection)
        labels.extend(self.method_best_labels(aq_name))
        return " | ".join(labels)

    def refresh_acquisition_combo_labels(self):
        if not self.names or self.combo.count() != len(self.names):
            return
        current = self.current_name()
        self.combo.blockSignals(True)
        for index, name in enumerate(self.names):
            label = self.acquisition_display_label(name)
            self.combo.setItemText(index, f"{label} {name}".strip())
        if current:
            for index in range(self.combo.count()):
                if self.combo.itemData(index) == current:
                    self.combo.setCurrentIndex(index)
                    break
        self.combo.blockSignals(False)

    def label_color(self, aq_name):
        label = self.selection_label(aq_name)
        if label == "BEST":
            return "#d62728"
        if label == "POSSIBLE":
            return "#ffb000"
        return "#ffffff"

    def current_name(self):
        return self.combo.currentData()

    def acquisition_actual_solar_zenith(self, aq_name):
        if self.handle is not None and aq_name in self.handle:
            aq = self.handle[aq_name]
            sun_alt = aq.attrs.get("Sun Position Altitude", np.nan)
            try:
                sun_alt = float(sun_alt)
                if np.isfinite(sun_alt):
                    return 90.0 - sun_alt
            except Exception:
                pass

        row = self.table.get(aq_name, {})
        sun_zen = as_float(row, "sun_zen_for_filter_deg")
        return sun_zen if np.isfinite(sun_zen) else np.nan

    def acquisition_geometry_sun_zenith(self, aq_name):
        if self.handle is None or aq_name not in self.handle:
            return self.acquisition_actual_solar_zenith(aq_name)

        aq = self.handle[aq_name]
        uv = aq.get("UV Image Data")
        if uv is None or "view_zen" not in uv:
            return self.acquisition_actual_solar_zenith(aq_name)

        center = None
        if "Measurement_Metadata" in self.handle:
            center = self.handle["Measurement_Metadata"].attrs.get("Center Pixel (x,y)")
        if center is None or len(center) < 2:
            return self.acquisition_actual_solar_zenith(aq_name)

        try:
            col = int(np.clip(round(float(center[0])), 0, uv["view_zen"].shape[1] - 1))
            row = int(np.clip(round(float(center[1])), 0, uv["view_zen"].shape[0] - 1))
            sun_zen = float(uv["view_zen"][row, col])
            if np.isfinite(sun_zen):
                return sun_zen
        except Exception:
            pass
        return self.acquisition_actual_solar_zenith(aq_name)

    def acquisition_sun_zenith(self, aq_name):
        return self.acquisition_actual_solar_zenith(aq_name)

    def acquisition_sun_zeniths(self):
        values = []
        for name in self.names:
            sun_zen = self.acquisition_sun_zenith(name)
            if np.isfinite(sun_zen):
                values.append((name, sun_zen))
        return values

    def populate_sza_reference_combo(self):
        current_data = self.sza_reference_combo.currentData()
        current_key = current_data[0] if isinstance(current_data, tuple) else "actual"
        current_name = current_data[1] if isinstance(current_data, tuple) and len(current_data) > 1 else None

        self.sza_reference_combo.blockSignals(True)
        self.sza_reference_combo.clear()
        self.sza_reference_combo.addItem("Actual acquisition", userData=("actual", None))

        sun_zeniths = self.acquisition_sun_zeniths()
        if sun_zeniths:
            avg_sza = float(np.nanmean([value for _, value in sun_zeniths]))
            self.sza_reference_combo.addItem(f"Average SZA {avg_sza:.2f}", userData=("average", None))
            for name, sun_zen in sun_zeniths:
                self.sza_reference_combo.addItem(f"{name} SZA {sun_zen:.2f}", userData=("acquisition", name))

        restore_index = 0
        for index in range(self.sza_reference_combo.count()):
            data = self.sza_reference_combo.itemData(index)
            if not isinstance(data, tuple):
                continue
            if data[0] == current_key and (current_key != "acquisition" or data[1] == current_name):
                restore_index = index
                break
        self.sza_reference_combo.setCurrentIndex(restore_index)
        self.sza_reference_combo.blockSignals(False)

    def selected_reference_sza(self):
        data = self.sza_reference_combo.currentData()
        current = self.current_name()
        current_sza = self.acquisition_sun_zenith(current) if current else np.nan
        if not isinstance(data, tuple):
            return {
                "label": "Actual acquisition",
                "sza": current_sza,
                "delta": 0.0,
                "source": "actual",
            }

        mode, name = data
        if mode == "average":
            sun_zeniths = self.acquisition_sun_zeniths()
            ref_sza = float(np.nanmean([value for _, value in sun_zeniths])) if sun_zeniths else np.nan
            label = "Average SZA"
        elif mode == "acquisition":
            ref_sza = self.acquisition_sun_zenith(name)
            label = f"{name} SZA"
        else:
            ref_sza = current_sza
            label = "Actual acquisition"

        delta = ref_sza - current_sza if np.isfinite(ref_sza) and np.isfinite(current_sza) else np.nan
        if mode == "actual":
            delta = 0.0 if np.isfinite(current_sza) else np.nan
        return {
            "label": label,
            "sza": ref_sza,
            "delta": delta,
            "source": mode,
        }

    def reference_adjusted_zenith(self, zenith, reference):
        if not np.isfinite(zenith) or not np.isfinite(reference.get("delta", np.nan)):
            return np.nan
        return zenith + reference["delta"]

    def prev_acquisition(self):
        self.combo.setCurrentIndex(max(0, self.combo.currentIndex() - 1))

    def next_acquisition(self):
        self.combo.setCurrentIndex(min(self.combo.count() - 1, self.combo.currentIndex() + 1))

    def clear_cache_and_redraw(self):
        self.cache = {}
        self.panorama_cache = None
        if self.handle is not None and has_level2_products(self.handle):
            self.avg_candidates = self.average_qu_candidates()
            self.best_avg_acquisition = (
                max(self.avg_candidates.values(), key=lambda row: row["combined_score"])["acquisition"]
                if self.avg_candidates else None
            )
            self.refresh_acquisition_combo_labels()
        self.redraw()

    def clear_panorama_and_redraw(self):
        self.panorama_cache = None
        self.redraw()

    def stitch_priority_mode(self):
        if hasattr(self, "stitch_priority_combo"):
            return self.stitch_priority_combo.currentData() or "center"
        return "center"

    def stitch_priority_label(self):
        if hasattr(self, "stitch_priority_combo"):
            return self.stitch_priority_combo.currentText()
        return "Center 1/2"

    def _update_tilt_corr_label(self):
        res = self.tilt_correction_result
        if res is None:
            self.tilt_corr_label.setText("Tilt corr: n/a")
            return
        corr = res["correction_deg"]
        n = res["n_used"]
        total = res.get("n_total", "?")
        if np.isfinite(corr):
            self.tilt_corr_label.setText(f"Tilt corr: {corr:+.4f}° ({n}/{total})")
        else:
            self.tilt_corr_label.setText(f"Tilt corr: no data (0/{total})")

    def _active_tilt_correction_deg(self):
        """Return correction to subtract from view_zen, or 0 if not active."""
        if not self.apply_tilt_corr_check.isChecked():
            return 0.0
        res = self.tilt_correction_result
        if res is None:
            return 0.0
        corr = res.get("correction_deg", np.nan)
        return float(corr) if np.isfinite(corr) else 0.0

    def _roll_deg(self, aq_name, version):
        try:
            roll = float(self.handle[aq_name].attrs.get(f"Camera Roll {version} [deg]", np.nan))
        except Exception:
            roll = np.nan
        return roll

    def _geometry_shift_deg(self, aq_name, version):
        """Return (zen_shift_deg, az_shift_deg) to convert raw view_zen/view_az to vN."""
        try:
            attrs = self.handle[aq_name].attrs
            zen_shift = float(attrs.get(f"Tilt Error {version} Pred [deg]", np.nan))
            pan_vn = float(attrs.get(f"Pan_{version} [deg]", np.nan))
            pan_raw = float(attrs.get("Geometry Pan Used [deg]", np.nan))
        except Exception:
            return 0.0, 0.0
        zen_shift = zen_shift if np.isfinite(zen_shift) else 0.0
        az_shift = (pan_vn - pan_raw) if np.isfinite(pan_vn) and np.isfinite(pan_raw) else 0.0
        return zen_shift, az_shift

    def _active_version(self):
        """Return 'v2', 'v3', or None depending on which checkbox is active."""
        if self.apply_roll_rotation_check.isChecked():
            return "v2"
        if self.apply_v3_check.isChecked():
            return "v3"
        return None

    def _active_roll_deg(self, aq_name):
        """Return the Q/U rotation angle theta, or 0 if not active/unavailable."""
        version = self._active_version()
        if version is None:
            return 0.0
        roll = self._roll_deg(aq_name, version)
        return float(roll) if np.isfinite(roll) else 0.0

    def _active_geometry_shift(self, aq_name):
        version = self._active_version()
        if version is None:
            return 0.0, 0.0
        return self._geometry_shift_deg(aq_name, version)

    def _update_roll_rotation_label(self, aq_name):
        roll = self._roll_deg(aq_name, "v2")
        if np.isfinite(roll):
            self.roll_rotation_label.setText(f"Roll: {roll:+.4f}°")
        else:
            self.roll_rotation_label.setText("Roll: n/a")
        v3_zen = self._zen_shift_deg(aq_name, "v3")
        if np.isfinite(v3_zen):
            self.v3_label.setText(f"v3: {v3_zen:+.4f}°")
        else:
            self.v3_label.setText("v3: n/a")

    def _zen_shift_deg(self, aq_name, version):
        try:
            shift = float(self.handle[aq_name].attrs.get(f"Tilt Error {version} Pred [deg]", np.nan))
        except Exception:
            shift = np.nan
        return shift

    def _on_v2_toggled(self, checked):
        if checked and self.apply_v3_check.isChecked():
            self.apply_v3_check.blockSignals(True)
            self.apply_v3_check.setChecked(False)
            self.apply_v3_check.blockSignals(False)
            self.clear_cache_and_redraw()

    def _on_v3_toggled(self, checked):
        if checked and self.apply_roll_rotation_check.isChecked():
            self.apply_roll_rotation_check.blockSignals(True)
            self.apply_roll_rotation_check.setChecked(False)
            self.apply_roll_rotation_check.blockSignals(False)
            self.clear_cache_and_redraw()

    def get_products(self, aq_name):
        corr = self._active_tilt_correction_deg()
        roll = self._active_roll_deg(aq_name)
        zen_shift, az_shift = self._active_geometry_shift(aq_name)
        key = (
            aq_name, self.sigma_spin.value(), round(corr, 6),
            round(roll, 6), round(zen_shift, 6), round(az_shift, 6),
        )
        if key not in self.cache:
            products = load_products(self.handle, aq_name, self.sigma_spin.value())
            if corr != 0.0:
                products = dict(products)
                products["view_zen"] = products["view_zen"] - corr
            if zen_shift != 0.0 or az_shift != 0.0:
                products = dict(products)
                products["view_zen"] = products["view_zen"] - zen_shift
                products["view_az"] = products["view_az"] + az_shift
            if roll != 0.0:
                products = dict(products)
                products["q"], products["u"] = rotate_qu(products["q"], products["u"], roll)
            self.cache[key] = products
        return self.cache[key]

    def build_panorama(self):
        step = int(os.environ.get("NP_PANORAMA_PIXEL_STEP", "8"))
        step = max(1, step)
        az_min = np.inf
        az_max = -np.inf
        zen_min = np.inf
        zen_max = -np.inf
        angular_steps = []

        for aq_name in self.names:
            products = self.get_products(aq_name)
            view_zen = products["view_zen"]
            view_az = products["view_az"]
            az_sample = view_az[::step, ::step]
            zen_sample = view_zen[::step, ::step]
            finite = np.isfinite(az_sample) & np.isfinite(zen_sample)
            if not np.any(finite):
                continue
            az_min = min(az_min, float(np.nanmin(az_sample[finite])))
            az_max = max(az_max, float(np.nanmax(az_sample[finite])))
            zen_min = min(zen_min, float(np.nanmin(zen_sample[finite])))
            zen_max = max(zen_max, float(np.nanmax(zen_sample[finite])))
            angular_steps.append(angular_step_from_grid(view_zen, view_az, step))

        if not np.isfinite([az_min, az_max, zen_min, zen_max]).all():
            return None

        deg_step = float(np.nanmedian(angular_steps)) if angular_steps else 0.05
        deg_step = max(deg_step, 1e-3)
        max_axis_pixels = int(os.environ.get("NP_PANORAMA_MAX_AXIS_PIXELS", "1600"))
        az_range = max(az_max - az_min, deg_step)
        zen_range = max(zen_max - zen_min, deg_step)
        deg_step = max(deg_step, az_range / max(max_axis_pixels - 1, 1), zen_range / max(max_axis_pixels - 1, 1))
        n_cols = int(np.ceil(az_range / deg_step)) + 1
        n_rows = int(np.ceil(zen_range / deg_step)) + 1

        dolp_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
        q_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
        u_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
        dolp_weight = np.zeros((n_rows, n_cols), dtype=np.float64)
        q_weight = np.zeros((n_rows, n_cols), dtype=np.float64)
        u_weight = np.zeros((n_rows, n_cols), dtype=np.float64)

        for aq_index, aq_name in enumerate(self.names):
            products = self.get_products(aq_name)
            view_az = products["view_az"][::step, ::step]
            view_zen = products["view_zen"][::step, ::step]
            row_idx = np.rint((view_zen - zen_min) / deg_step).astype(np.int64)
            col_idx = np.rint((view_az - az_min) / deg_step).astype(np.int64)
            center_weight = center_priority_weights(products["dolp"].shape, step, self.stitch_priority_mode())
            intensity = products["I"][::step, ::step]
            i_low = np.nanpercentile(intensity, 10)
            i_high = np.nanpercentile(intensity, 90)
            intensity_weight = np.clip((intensity - i_low) / max(i_high - i_low, 1e-6), 0.05, 1.0)
            blend_weight = center_weight * intensity_weight
            add_to_panorama_blend(dolp_sum, dolp_weight, row_idx, col_idx, products["dolp"][::step, ::step], blend_weight)
            add_to_panorama_blend(q_sum, q_weight, row_idx, col_idx, products["q"][::step, ::step], blend_weight)
            add_to_panorama_blend(u_sum, u_weight, row_idx, col_idx, products["u"][::step, ::step], blend_weight)

        pano_dolp = blended_panorama(dolp_sum, dolp_weight)
        pano_q = blended_panorama(q_sum, q_weight)
        pano_u = blended_panorama(u_sum, u_weight)

        smooth_sigma = float(os.environ.get("NP_PANORAMA_SMOOTH_SIGMA", "1.5"))
        valid_pixels = int(np.count_nonzero(np.isfinite(pano_dolp)))
        total_pixels = int(pano_dolp.size)
        coverage = valid_pixels / max(total_pixels, 1)
        mean_weight = float(np.nanmean(dolp_weight[dolp_weight > 0])) if np.any(dolp_weight > 0) else 0.0
        raw_roughness = {
            "dolp": roughness_metric(pano_dolp),
            "q": roughness_metric(pano_q),
            "u": roughness_metric(pano_u),
        }
        pano_dolp_s = nan_gaussian_filter(pano_dolp, smooth_sigma)
        pano_q_s = nan_gaussian_filter(pano_q, smooth_sigma)
        pano_u_s = nan_gaussian_filter(pano_u, smooth_sigma)
        smooth_roughness = {
            "dolp": roughness_metric(pano_dolp_s),
            "q": roughness_metric(pano_q_s),
            "u": roughness_metric(pano_u_s),
        }

        return {
            "dolp": pano_dolp_s,
            "log_dolp": np.log10(np.maximum(pano_dolp_s, 1e-6)),
            "q": pano_q_s,
            "u": pano_u_s,
            "extent": [az_min, az_max, zen_min, zen_max],
            "step_deg": deg_step,
            "sample_step_px": step,
            "smooth_sigma": smooth_sigma,
            "coverage": coverage,
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "priority_mode": self.stitch_priority_mode(),
            "priority_label": self.stitch_priority_label(),
            "blend_method": "weighted average",
            "mean_weight": mean_weight,
            "raw_roughness": raw_roughness,
            "smooth_roughness": smooth_roughness,
        }

    def get_panorama(self):
        key = (self.h5_path, self.sigma_spin.value(), self.stitch_priority_mode(), tuple(self.names))
        if self.panorama_cache is None or self.panorama_cache.get("key") != key:
            panorama = self.build_panorama()
            self.panorama_cache = {"key": key, "panorama": panorama}
        return self.panorama_cache["panorama"]

    def current_np(self, aq_name):
        row = self.table.get(aq_name, {})
        zero_row = as_float(row, "zero_row")
        zero_col = as_float(row, "zero_col")
        if np.isfinite(zero_row) and np.isfinite(zero_col):
            return row, {
                "label": "Q/U zero NP",
                "row": zero_row,
                "col": zero_col,
                "zen": as_float(row, "zero_zen_deg"),
                "az": as_float(row, "zero_az_deg"),
            }

        dolp_row = as_float(row, "dolp_row")
        dolp_col = as_float(row, "dolp_col")
        if np.isfinite(dolp_row) and np.isfinite(dolp_col):
            return row, {
                "label": "DoLP low-region NP",
                "row": dolp_row,
                "col": dolp_col,
                "zen": as_float(row, "dolp_zen_deg"),
                "az": as_float(row, "dolp_az_deg"),
            }
        return row, None

    def add_np_markers(self, ax, aq_name, row, selected, display_points=None, annotate=False):
        display_points = display_points or {}
        dolp_row = as_float(row, "dolp_row")
        dolp_col = as_float(row, "dolp_col")
        zero_row = as_float(row, "zero_row")
        zero_col = as_float(row, "zero_col")
        if np.isfinite(dolp_row) and np.isfinite(dolp_col):
            dolp_point = display_points.get("dolp", {})
            ax.scatter(dolp_col, dolp_row, marker="+", s=140, c="cyan", linewidths=2)
            if annotate:
                ax.annotate(
                    f"DoLP\nZ={dolp_point.get('zen', as_float(row, 'dolp_zen_deg')):.2f}\nA={dolp_point.get('az', as_float(row, 'dolp_az_deg')):.2f}",
                    (dolp_col, dolp_row),
                    xytext=(8, 8),
                    textcoords="offset points",
                    color="cyan",
                    fontsize=7,
                    weight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
                )
        if np.isfinite(zero_row) and np.isfinite(zero_col):
            zero_point = display_points.get("zero", {})
            ax.scatter(zero_col, zero_row, marker="x", s=120, c="lime", linewidths=2)
            if annotate:
                ax.annotate(
                    f"Q/U\nZ={zero_point.get('zen', as_float(row, 'zero_zen_deg')):.2f}\nA={zero_point.get('az', as_float(row, 'zero_az_deg')):.2f}",
                    (zero_col, zero_row),
                    xytext=(8, -30),
                    textcoords="offset points",
                    color="lime",
                    fontsize=7,
                    weight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
                )
        if selected:
            ax.scatter(
                selected["col"],
                selected["row"],
                marker="o",
                s=110,
                facecolors="none",
                edgecolors=self.label_color(aq_name),
                linewidths=2.5,
            )
        if aq_name == self.best_dolp_acquisition and np.isfinite(dolp_row) and np.isfinite(dolp_col):
            ax.scatter(
                dolp_col,
                dolp_row,
                marker="s",
                s=190,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.2,
            )
            if annotate:
                ax.annotate(
                    "BEST DoLP",
                    (dolp_col, dolp_row),
                    xytext=(8, -12),
                    textcoords="offset points",
                    color="#d62728",
                    fontsize=8,
                    weight="bold",
                )
        if aq_name == self.best_zero_acquisition and np.isfinite(zero_row) and np.isfinite(zero_col):
            ax.scatter(
                zero_col,
                zero_row,
                marker="D",
                s=180,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.2,
            )
            if annotate:
                ax.annotate(
                    "BEST Q/U",
                    (zero_col, zero_row),
                    xytext=(8, 10),
                    textcoords="offset points",
                    color="#d62728",
                    fontsize=8,
                    weight="bold",
                )

    def add_panorama_np_markers(self, ax, row, selected, display_points=None, annotate=False):
        display_points = display_points or {}
        dolp_point = display_points.get("dolp", {})
        zero_point = display_points.get("zero", {})
        dolp_zen = dolp_point.get("zen", as_float(row, "dolp_zen_deg"))
        dolp_az = dolp_point.get("az", as_float(row, "dolp_az_deg"))
        zero_zen = zero_point.get("zen", as_float(row, "zero_zen_deg"))
        zero_az = zero_point.get("az", as_float(row, "zero_az_deg"))
        if np.isfinite(dolp_zen) and np.isfinite(dolp_az):
            ax.scatter(dolp_az, dolp_zen, marker="+", s=150, c="cyan", linewidths=2.2, zorder=5)
            if annotate:
                ax.annotate(
                    f"DoLP NP\nZ={dolp_zen:.2f}\nA={dolp_az:.2f}",
                    (dolp_az, dolp_zen),
                    xytext=(8, 8),
                    textcoords="offset points",
                    color="cyan",
                    fontsize=7,
                    weight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
                    zorder=6,
                )
        if np.isfinite(zero_zen) and np.isfinite(zero_az):
            ax.scatter(zero_az, zero_zen, marker="x", s=130, c="lime", linewidths=2.2, zorder=5)
            if annotate:
                ax.annotate(
                    f"Q/U NP\nZ={zero_zen:.2f}\nA={zero_az:.2f}",
                    (zero_az, zero_zen),
                    xytext=(8, -30),
                    textcoords="offset points",
                    color="lime",
                    fontsize=7,
                    weight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
                    zorder=6,
                )
        if selected:
            ax.scatter(
                selected["az"],
                selected["zen"],
                marker="o",
                s=115,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.3,
                zorder=5,
            )

    def add_angular_np_markers(self, ax, row, selected, display_points=None, annotate=False):
        display_points = display_points or {}
        dolp_point = display_points.get("dolp", {})
        zero_point = display_points.get("zero", {})
        dolp_zen = dolp_point.get("zen", as_float(row, "dolp_zen_deg"))
        dolp_az = dolp_point.get("az", as_float(row, "dolp_az_deg"))
        zero_zen = zero_point.get("zen", as_float(row, "zero_zen_deg"))
        zero_az = zero_point.get("az", as_float(row, "zero_az_deg"))
        if np.isfinite(dolp_zen) and np.isfinite(dolp_az):
            ax.scatter(dolp_az, dolp_zen, marker="+", s=140, c="cyan", linewidths=2, zorder=5)
            if annotate:
                ax.annotate(
                    f"DoLP\nZ={dolp_zen:.2f}\nA={dolp_az:.2f}",
                    (dolp_az, dolp_zen),
                    xytext=(8, 8),
                    textcoords="offset points",
                    color="cyan",
                    fontsize=7,
                    weight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
                    zorder=6,
                )
        if np.isfinite(zero_zen) and np.isfinite(zero_az):
            ax.scatter(zero_az, zero_zen, marker="x", s=120, c="lime", linewidths=2, zorder=5)
            if annotate:
                ax.annotate(
                    f"Q/U\nZ={zero_zen:.2f}\nA={zero_az:.2f}",
                    (zero_az, zero_zen),
                    xytext=(8, -30),
                    textcoords="offset points",
                    color="lime",
                    fontsize=7,
                    weight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
                    zorder=6,
                )
        if selected:
            ax.scatter(
                selected["az"],
                selected["zen"],
                marker="o",
                s=110,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.5,
                zorder=5,
            )

    def common_legend_handles(self):
        handles = []
        handles.append(Line2D([0], [0], color="white", lw=1.8, label="Q/I = 0"))
        handles.append(Line2D([0], [0], color="magenta", lw=1.8, label="U/I = 0"))
        handles.extend([
            Line2D([0], [0], marker="+", color="cyan", lw=0, markersize=12, markeredgewidth=2, label="DoLP NP"),
            Line2D([0], [0], marker="x", color="lime", lw=0, markersize=10, markeredgewidth=2, label="Q/U zero NP"),
            Line2D([0], [0], marker="o", color="#d62728", lw=0, markerfacecolor="none", markeredgewidth=2.5, markersize=10, label="BEST"),
            Line2D([0], [0], marker="o", color="#ffb000", lw=0, markerfacecolor="none", markeredgewidth=2.5, markersize=10, label="POSSIBLE"),
            Line2D([0], [0], marker="s", color="#d62728", lw=0, markerfacecolor="none", markeredgewidth=2.2, markersize=10, label="BEST DoLP"),
            Line2D([0], [0], marker="D", color="#d62728", lw=0, markerfacecolor="none", markeredgewidth=2.2, markersize=9, label="BEST Q/U"),
        ])
        return handles

    def redraw(self):
        if self.handle is None or self.combo.currentIndex() < 0:
            return
        aq_name = self.current_name()
        if not aq_name:
            return
        self._update_roll_rotation_label(aq_name)
        products = self.get_products(aq_name)
        row, selected = self.current_np(aq_name)
        q = products["q"]
        u = products["u"]
        view_zen = products["view_zen"]
        view_az = products["view_az"]
        aq = products["aq"]
        display_points = display_np_points(row, view_zen, view_az)
        selected_display = selected
        if selected:
            point_key = "zero" if selected["label"].startswith("Q/U") else "dolp"
            if point_key in display_points:
                selected_display = dict(selected)
                selected_display["zen"] = display_points[point_key]["zen"]
                selected_display["az"] = display_points[point_key]["az"]

        self.figure.clear()
        gs = self.figure.add_gridspec(
            3,
            16,
            width_ratios=[
                2.35, 0.88, 2.35, 0.88, 2.35, 0.88, 2.35,
                1.20, 1.12, 0.78, 1.12, 0.78, 1.12, 0.72, 0.70, 0.45
            ],
            height_ratios=[2.35, 0.72, 0.72],
            wspace=0.90,
            hspace=0.82,
        )
        ax_i = self.figure.add_subplot(gs[0, 0])
        ax_dolp = self.figure.add_subplot(gs[0, 2])
        ax_q = self.figure.add_subplot(gs[0, 4])
        ax_u = self.figure.add_subplot(gs[0, 6])
        ax_stats = self.figure.add_subplot(gs[0, 8:16])
        ax_col = self.figure.add_subplot(gs[1, 0:3])
        ax_avg_q = self.figure.add_subplot(gs[1, 3:6])
        ax_row = self.figure.add_subplot(gs[2, 0:3])
        ax_avg_u = self.figure.add_subplot(gs[2, 3:6])
        ax_pano_dolp = self.figure.add_subplot(gs[1:3, 8])
        ax_pano_q = self.figure.add_subplot(gs[1:3, 10])
        ax_pano_u = self.figure.add_subplot(gs[1:3, 12])
        ax_stats.axis("off")

        color = self.label_color(aq_name)
        label = self.selection_label(aq_name)
        method_labels = self.method_best_labels(aq_name)
        avg_candidate = self.avg_candidates.get(aq_name, {})
        title_labels = [item for item in [label, *method_labels] if item]
        prefix = f"{' | '.join(title_labels)}: " if title_labels else ""
        frame_extent = angular_extent(view_zen, view_az)
        frame_image_extent = [frame_extent[0], frame_extent[1], frame_extent[3], frame_extent[2]]
        frame_x = np.linspace(frame_extent[0], frame_extent[1], q.shape[1])
        frame_y = np.linspace(frame_extent[2], frame_extent[3], q.shape[0])
        raw_frame_display = self.frame_display_combo.currentData() == "raw"

        if raw_frame_display:
            im_i = ax_i.imshow(products["I"], cmap="gray")
            self.add_np_markers(ax_i, aq_name, row, selected_display, display_points)
            label_raw_image_axes(ax_i, view_zen, view_az)
        else:
            im_i = ax_i.imshow(products["I"], cmap="gray", extent=frame_image_extent, origin="upper", aspect="auto")
            self.add_angular_np_markers(ax_i, row, selected_display, display_points)
            ax_i.set_xlabel("Azimuth [deg]")
            ax_i.set_ylabel("Zenith [deg]")
            ax_i.set_ylim(frame_extent[3], frame_extent[2])
            ax_i.tick_params(labelsize=7)
        ax_i.set_title("I", color=color, fontsize=9)
        self.figure.colorbar(im_i, ax=ax_i, fraction=0.04, pad=0.025)

        if raw_frame_display:
            im0 = ax_dolp.imshow(products["log_dolp"], cmap="viridis", vmin=-3, vmax=-0.5)
            ax_dolp.contour(q, levels=[0], colors="white", linewidths=0.8)
            ax_dolp.contour(u, levels=[0], colors="magenta", linewidths=0.8)
            self.add_np_markers(ax_dolp, aq_name, row, selected_display, display_points)
            label_raw_image_axes(ax_dolp, view_zen, view_az)
        else:
            im0 = ax_dolp.imshow(products["log_dolp"], cmap="viridis", vmin=-3, vmax=-0.5, extent=frame_image_extent, origin="upper", aspect="auto")
            ax_dolp.contour(frame_x, frame_y, q, levels=[0], colors="white", linewidths=0.8)
            ax_dolp.contour(frame_x, frame_y, u, levels=[0], colors="magenta", linewidths=0.8)
            self.add_angular_np_markers(ax_dolp, row, selected_display, display_points)
            ax_dolp.set_xlabel("Azimuth [deg]")
            ax_dolp.set_ylabel("Zenith [deg]")
            ax_dolp.set_ylim(frame_extent[3], frame_extent[2])
            ax_dolp.tick_params(labelsize=7)
        ax_dolp.set_title(f"{prefix}log10 DoLP", color=color, fontsize=9)
        self.figure.colorbar(im0, ax=ax_dolp, fraction=0.04, pad=0.025)

        norm = SymLogNorm(linthresh=1e-3, linscale=1, vmin=-0.08, vmax=0.08, base=10)
        if raw_frame_display:
            im1 = ax_q.imshow(q, cmap="coolwarm", norm=norm)
            ax_q.contour(q, levels=[0], colors="white", linewidths=0.9)
            self.add_np_markers(ax_q, aq_name, row, selected_display, display_points)
            label_raw_image_axes(ax_q, view_zen, view_az)
        else:
            im1 = ax_q.imshow(q, cmap="coolwarm", norm=norm, extent=frame_image_extent, origin="upper", aspect="auto")
            ax_q.contour(frame_x, frame_y, q, levels=[0], colors="white", linewidths=0.9)
            self.add_angular_np_markers(ax_q, row, selected_display, display_points)
            ax_q.set_xlabel("Azimuth [deg]")
            ax_q.set_ylabel("Zenith [deg]")
            ax_q.set_ylim(frame_extent[3], frame_extent[2])
            ax_q.tick_params(labelsize=7)
        ax_q.set_title("symlog Q/I", color=color, fontsize=9)
        self.figure.colorbar(im1, ax=ax_q, fraction=0.04, pad=0.025)

        if raw_frame_display:
            im2 = ax_u.imshow(u, cmap="coolwarm", norm=norm)
            ax_u.contour(u, levels=[0], colors="magenta", linewidths=0.9)
            self.add_np_markers(ax_u, aq_name, row, selected_display, display_points)
            label_raw_image_axes(ax_u, view_zen, view_az)
        else:
            im2 = ax_u.imshow(u, cmap="coolwarm", norm=norm, extent=frame_image_extent, origin="upper", aspect="auto")
            ax_u.contour(frame_x, frame_y, u, levels=[0], colors="magenta", linewidths=0.9)
            self.add_angular_np_markers(ax_u, row, selected_display, display_points)
            ax_u.set_xlabel("Azimuth [deg]")
            ax_u.set_ylabel("Zenith [deg]")
            ax_u.set_ylim(frame_extent[3], frame_extent[2])
            ax_u.tick_params(labelsize=7)
        ax_u.set_title("symlog U/I", color=color, fontsize=9)
        self.figure.colorbar(im2, ax=ax_u, fraction=0.04, pad=0.025)

        self.figure.legend(
            handles=self.common_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=8,
            fontsize=10,
            framealpha=0.95,
        )

        sun_alt = aq.attrs.get("Sun Position Altitude", np.nan)
        sun_az = aq.attrs.get("Sun Position Azimuth", np.nan)
        actual_sun_zen = self.acquisition_actual_solar_zenith(aq_name)
        sza_reference = self.selected_reference_sza()
        try:
            sun_text = f"{float(sun_alt):.3f}, {float(sun_az):.3f}"
        except Exception:
            sun_text = f"{sun_alt}, {sun_az}"

        if selected_display:
            col = int(np.clip(round(selected_display["col"]), 0, q.shape[1] - 1))
            row_idx = int(np.clip(round(selected_display["row"]), 0, q.shape[0] - 1))
            profile_source = "NP"
        else:
            row_idx = q.shape[0] // 2
            col = q.shape[1] // 2
            profile_source = "center"

        avg_q = np.nanmean(q, axis=1)
        avg_q_zen = np.nanmean(view_zen, axis=1)
        avg_u = np.nanmean(u, axis=0)
        avg_u_az = np.nanmean(view_az, axis=0)
        avg_q_zero = zero_crossing_1d(avg_q, avg_q_zen, preferred_index=row_idx)
        avg_u_zero = zero_crossing_1d(avg_u, avg_u_az, preferred_index=col)
        dolp_display_zen = display_points.get("dolp", {}).get("zen", as_float(row, "dolp_zen_deg"))
        dolp_display_az = display_points.get("dolp", {}).get("az", as_float(row, "dolp_az_deg"))
        zero_display_zen = display_points.get("zero", {}).get("zen", as_float(row, "zero_zen_deg"))
        zero_display_az = display_points.get("zero", {}).get("az", as_float(row, "zero_az_deg"))
        dolp_ref_zen = self.reference_adjusted_zenith(dolp_display_zen, sza_reference)
        zero_ref_zen = self.reference_adjusted_zenith(zero_display_zen, sza_reference)
        avg_q_ref_zen = (
            self.reference_adjusted_zenith(avg_q_zero["coord"], sza_reference)
            if avg_q_zero else np.nan
        )
        avg_fit_ref_zen = self.reference_adjusted_zenith(
            avg_candidate.get("avg_zen_deg", np.nan), sza_reference)

        corr_applied = self._active_tilt_correction_deg()
        res = self.tilt_correction_result
        if res is not None:
            corr_val = res["correction_deg"]
            n_used = res["n_used"]
            n_total = res.get("n_total", "?")
            corr_summary = (
                f"{corr_val:+.4f} deg  ({n_used}/{n_total} calib frames)"
                if np.isfinite(corr_val) else f"no usable calib frames (0/{n_total})"
            )
            detail_lines = []
            for d in res.get("details", []):
                if "skip" in d:
                    detail_lines.append(f"    {d['name']}: SKIPPED ({d['skip']})")
                else:
                    detail_lines.append(
                        f"    {d['name']}: dtilt={d['dtilt_cmd_deg']:+.2f} "
                        f"({d.get('dtilt_source', 'unknown')})  "
                        f"cy={d['cy_px']:.0f} cy_exp={d['cy_expected_px']:.0f}  "
                        f"dy={d['dy_px']:+.0f}px  "
                        f"corr@frame={d['corr_at_frame_deg']:+.4f}  "
                        f"solar_mot={d['solar_motion_deg']:+.4f}  "
                        f"→ corr@aq0={d['corr_at_aq0_deg']:+.4f}")
            tilt_corr_block = (
                ["", "Calibration tilt correction:", f"  result: {corr_summary}",
                 f"  applied: {corr_applied:+.4f} deg"]
                + detail_lines
            )
        else:
            tilt_corr_block = ["", "Calibration tilt correction: not available (no calibration_acqui_ groups)"]

        stats = [
            aq_name,
            f"Selection: {label if label else 'not selected'} | Method best: {', '.join(method_labels) if method_labels else 'no'}",
            f"UTC: {aq.attrs.get('Timestamp UTC', '')} | Sun alt/az: {sun_text}",
            f"Acquisition sun zenith: {actual_sun_zen:.4f}",
            f"NP zenith ref: {sza_reference['label']} | ref SZA {sza_reference['sza']:.4f} | dSZA {sza_reference['delta']:+.4f}",
            f"Circular crop: radius fraction {products.get('crop_radius_fraction', np.nan):.3f}, shrink {products.get('crop_radius_shrink_px', np.nan):.0f}px | valid {100 * np.count_nonzero(np.isfinite(products['I'])) / products['I'].size:.1f}%",
            f"Sun zen filter: {as_float(row, 'sun_zen_for_filter_deg'):.3f} | Min sep: {as_float(row, 'min_sun_zen_separation_deg'):.3f}",
            "",
            f"Conf DoLP/Zero/Total: {as_float(row, 'dolp_confidence'):.3f} / {as_float(row, 'zero_confidence'):.3f} / {as_float(row, 'total_confidence'):.3f}",
            f"Sun sep DoLP/Zero: {as_float(row, 'dolp_sun_zen_separation_deg'):.3f} ({row.get('dolp_sun_zen_separation_ok', '')}) / {as_float(row, 'zero_sun_zen_separation_deg'):.3f} ({row.get('zero_sun_zen_separation_ok', '')})",
            f"Zero contour: Q/I=0, U/I=0 after Gaussian sigma {self.sigma_spin.value():.2f}px",
            f"Zero line in FOV Q/U: {row.get('q_line_in_fov', '')} ({row.get('q_zero_pixel_count', '')} px) / {row.get('u_line_in_fov', '')} ({row.get('u_zero_pixel_count', '')} px)",
            "",
            f"DoLP NP px: {as_float(row, 'dolp_row'):.1f}, {as_float(row, 'dolp_col'):.1f} | Z/A: {dolp_display_zen:.4f}, {dolp_display_az:.4f} | RefZ: {dolp_ref_zen:.4f} | min: {as_float(row, 'dolp_min'):.6g}",
            f"Q/U NP px: {as_float(row, 'zero_row'):.1f}, {as_float(row, 'zero_col'):.1f} | Z/A: {zero_display_zen:.4f}, {zero_display_az:.4f} | RefZ: {zero_ref_zen:.4f} | residual: {as_float(row, 'zero_residual'):.6g}",
        ]
        if selected_display:
            selected_ref_zen = self.reference_adjusted_zenith(selected_display["zen"], sza_reference)
            stats.extend([
                "",
                f"Selected: {selected_display['label']} | px: {selected_display['row']:.1f}, {selected_display['col']:.1f} | Z/A: {selected_display['zen']:.4f}, {selected_display['az']:.4f} | RefZ: {selected_ref_zen:.4f}",
            ])
        stats.extend([
            "",
            "Average Q/U profile:",
            f"  source row,col: {row_idx}, {col} ({profile_source})",
            (
                f"  avg Q=0 zen: {avg_q_zero['coord']:.4f} "
                f"| RefZ {avg_q_ref_zen:.4f} "
                f"| idx {avg_q_zero['index']:.1f} | crossing {avg_q_zero['crossed']} "
                f"| residual {avg_q_zero['residual']:.3g}"
                if avg_q_zero else "  avg Q=0 zen: unavailable"
            ),
            (
                f"  avg U=0 az: {avg_u_zero['coord']:.4f} "
                f"| idx {avg_u_zero['index']:.1f} | crossing {avg_u_zero['crossed']} "
                f"| residual {avg_u_zero['residual']:.3g}"
                if avg_u_zero else "  avg U=0 az: unavailable"
            ),
            (
                f"  avg Q/U Z/A: {avg_q_zero['coord']:.4f}, {avg_u_zero['coord']:.4f}"
                if avg_q_zero and avg_u_zero else "  avg Q/U Z/A: unavailable"
            ),
            (
                f"  linearfit best: {'yes' if aq_name == self.best_avg_acquisition else 'no'} "
                f"| score {avg_candidate.get('combined_score', np.nan):.4f} "
                f"| fit Z/A {avg_candidate.get('avg_zen_deg', np.nan):.4f}, {avg_candidate.get('avg_az_deg', np.nan):.4f} "
                f"| RefZ {avg_fit_ref_zen:.4f}"
                if avg_candidate else "  linearfit best: unavailable"
            ),
        ])

        panorama = self.get_panorama()
        if panorama is not None:
            raw_r = panorama["raw_roughness"]
            smooth_r = panorama["smooth_roughness"]
            stats.extend([
                "",
                f"Panorama: {panorama['priority_label']} | {panorama['blend_method']} | coverage {100 * panorama['coverage']:.1f}% | grid {panorama['step_deg']:.4f} deg | smooth {panorama['smooth_sigma']:.2f}px",
                f"Rough DoLP/Q/U: {raw_r['dolp']:.3g}->{smooth_r['dolp']:.3g} / {raw_r['q']:.3g}->{smooth_r['q']:.3g} / {raw_r['u']:.3g}->{smooth_r['u']:.3g}",
            ])
        stats.extend(tilt_corr_block)
        ax_stats.text(0.01, 0.98, "\n".join(stats), va="top", ha="left", family="monospace", fontsize=7.8)

        ax_col.plot(q[:, col], view_zen[:, col], color="tab:blue")
        ax_col.axvline(0, color="black", linewidth=1)
        ax_col.scatter(q[row_idx, col], view_zen[row_idx, col], c="red", s=45, zorder=3)
        ax_col.set_xscale("symlog", linthresh=1e-3)
        ax_col.invert_yaxis()
        ax_col.set_xlabel("Q/I (symlog)")
        ax_col.set_ylabel("Zenith [deg]")
        ax_col.set_title(f"Q/I vs zenith | {profile_source} col {col}", fontsize=9)
        ax_col.tick_params(labelsize=8)
        ax_col.grid(True, alpha=0.3)

        ax_avg_q.plot(avg_q, avg_q_zen, color="tab:blue")
        ax_avg_q.axvline(0, color="black", linewidth=1)
        ax_avg_q.scatter(avg_q[row_idx], avg_q_zen[row_idx], c="red", s=45, zorder=3)
        if avg_q_zero:
            ax_avg_q.axhline(avg_q_zero["coord"], color="#d62728", linewidth=1, alpha=0.7)
            ax_avg_q.scatter(0, avg_q_zero["coord"], marker="D", c="#d62728", s=42, zorder=4)
        ax_avg_q.set_xscale("symlog", linthresh=1e-3)
        ax_avg_q.invert_yaxis()
        ax_avg_q.set_xlabel("avg Q/I")
        ax_avg_q.set_ylabel("Zenith [deg]")
        ax_avg_q.set_title("Average Q/I vs zenith", fontsize=9)
        ax_avg_q.tick_params(labelsize=8)
        ax_avg_q.grid(True, alpha=0.3)

        ax_row.plot(u[row_idx, :], view_az[row_idx, :], color="tab:orange")
        ax_row.axvline(0, color="black", linewidth=1)
        ax_row.scatter(u[row_idx, col], view_az[row_idx, col], c="red", s=45, zorder=3)
        ax_row.set_xscale("symlog", linthresh=1e-3)
        ax_row.set_xlabel("U/I (symlog)")
        ax_row.set_ylabel("Azimuth [deg]")
        ax_row.set_title(f"U/I vs azimuth | {profile_source} row {row_idx}", fontsize=9)
        ax_row.tick_params(labelsize=8)
        ax_row.grid(True, alpha=0.3)

        ax_avg_u.plot(avg_u, avg_u_az, color="tab:orange")
        ax_avg_u.axvline(0, color="black", linewidth=1)
        ax_avg_u.scatter(avg_u[col], avg_u_az[col], c="red", s=45, zorder=3)
        if avg_u_zero:
            ax_avg_u.axhline(avg_u_zero["coord"], color="#d62728", linewidth=1, alpha=0.7)
            ax_avg_u.scatter(0, avg_u_zero["coord"], marker="D", c="#d62728", s=42, zorder=4)
        ax_avg_u.set_xscale("symlog", linthresh=1e-3)
        ax_avg_u.set_xlabel("avg U/I")
        ax_avg_u.set_ylabel("Azimuth [deg]")
        ax_avg_u.set_title("Average U/I vs azimuth", fontsize=9)
        ax_avg_u.tick_params(labelsize=8)
        ax_avg_u.grid(True, alpha=0.3)

        if panorama is None:
            for ax in [ax_pano_dolp, ax_pano_q, ax_pano_u]:
                ax.text(0.5, 0.5, "No panorama", ha="center", va="center")
                ax.axis("off")
        else:
            extent = panorama["extent"]
            pano_x = np.linspace(extent[0], extent[1], panorama["q"].shape[1])
            pano_y = np.linspace(extent[2], extent[3], panorama["q"].shape[0])
            pano_norm = SymLogNorm(linthresh=1e-3, linscale=1, vmin=-0.08, vmax=0.08, base=10)
            im_p0 = ax_pano_dolp.imshow(
                panorama["log_dolp"],
                cmap="viridis",
                vmin=-3,
                vmax=-0.5,
                extent=extent,
                origin="lower",
                aspect="equal")
            ax_pano_dolp.set_title("Panorama DoLP", fontsize=9)
            self.add_panorama_np_markers(ax_pano_dolp, row, selected_display, display_points)
            self.figure.colorbar(im_p0, ax=ax_pano_dolp, fraction=0.12, pad=0.06)

            im_p1 = ax_pano_q.imshow(
                panorama["q"],
                cmap="coolwarm",
                norm=pano_norm,
                extent=extent,
                origin="lower",
                aspect="equal")
            ax_pano_q.contour(pano_x, pano_y, panorama["q"], levels=[0], colors="white", linewidths=0.8)
            ax_pano_q.set_title("Panorama Q/I", fontsize=9)
            self.add_panorama_np_markers(ax_pano_q, row, selected_display, display_points)
            self.figure.colorbar(im_p1, ax=ax_pano_q, fraction=0.12, pad=0.06)

            im_p2 = ax_pano_u.imshow(
                panorama["u"],
                cmap="coolwarm",
                norm=pano_norm,
                extent=extent,
                origin="lower",
                aspect="equal")
            ax_pano_u.contour(pano_x, pano_y, panorama["u"], levels=[0], colors="magenta", linewidths=0.8)
            ax_pano_u.set_title("Panorama U/I", fontsize=9)
            self.add_panorama_np_markers(ax_pano_u, row, selected_display, display_points)
            self.figure.colorbar(im_p2, ax=ax_pano_u, fraction=0.12, pad=0.06)

            for ax in [ax_pano_dolp, ax_pano_q, ax_pano_u]:
                ax.set_xlabel("Azimuth [deg]")
                ax.set_ylabel("Zenith [deg]")
                ax.set_ylim(extent[3], extent[2])
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=8)
                ax.grid(False)

        self.figure.suptitle(os.path.basename(self.h5_path))
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    initial_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    win = RobustNPQtApp(initial, initial_index=initial_index)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

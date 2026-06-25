"""
compute_u_neutral_line_calibration.py
======================================
Calibrate ULTRASIP pointing using the Stokes-U neutral line.

Physics
-------
In Rayleigh sky polarimetry, U = 0 along the solar meridian (a vertical line
in the image when the camera is correctly oriented).  Column offset from the
image centre encodes azimuth pointing error; the slope of the U=0 line from
vertical encodes camera roll.

Stokes rotation
---------------
  U'(φ) = -Q·sin(2φ) + U·cos(2φ)

At φ = camera_roll, the U'=0 neutral line becomes exactly vertical.
Algorithm: sweep φ ∈ [-PHI_HALF_DEG, +PHI_HALF_DEG], find per-row zero
crossings of U'(φ), fit (row, col) with a line, pick φ that minimises slope².
The column position of that vertical line at the image centre gives az_offset.

Pan-Offset convention (from process_calibration_v4.py)
-------------------------------------------------------
  True sky pointing (az)  = Moog_actual_pan + Pan_Offset
  Sun pan in Moog coords   = sun_az_met - 180
  Residual u_x = Moog_actual_pan + Pan_Offset - sun_pan  ← ≈ 0 when correct
  Any measured column offset = residual pointing error after Pan-Offset correction.

Multi-acquisition pointing model
---------------------------------
  az_offset_deg(pan) = Δpitch·cos(pan_rad) + Δtilt_az·sin(pan_rad) + C_az

Usage
-----
  python compute_u_neutral_line_calibration.py  [data_dir]

  data_dir defaults to '/Volumes/LaCie/Level2 data/2026_06_05'

Output
------
  * Console table: per-acquisition results
  * CSV:  u_neutral_line_results.csv  (saved in data_dir)
  * Console summary: fitted Δpitch, Δtilt_az, C_az, Δroll
"""

import sys, csv, warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion

# ── Configuration ────────────────────────────────────────────────────────────
BAND_HALF_ROWS    = 300    # half-height of centre band loaded for sweep
SIGMA_SMOOTH      = 8      # Gaussian σ (pixels) applied to U, Q, I
I_PERCENTILE_MASK = 20     # mask pixels below this intensity percentile
MAX_DTILT_DEG     = 1.5    # only process acquisitions with |Δtilt| < this
PIX_SCALE         = 0.002  # deg / pixel (nominal ULTRASIP)
CLUSTER_PIX       = 200    # per-row crossing outlier rejection threshold (px)
PHI_HALF_DEG      = 10.0   # roll search range: ±PHI_HALF_DEG degrees
PHI_COARSE_STEPS  = 41     # coarse sweep steps (step ≈ 0.5°)
PHI_FINE_STEPS    = 101    # fine sweep steps around coarse min (step ≈ 0.02°)
PHI_FINE_HALF     = 1.0    # fine sweep half-width (°)
DISK_HALF_ROWS    = 600    # half-height of band loaded for disk circle fit
# ─────────────────────────────────────────────────────────────────────────────


def azimuth_to_moog_pan(az_deg: float) -> float:
    """Meteorological azimuth (N=0, E=90) → Moog pan (S=0)."""
    return (float(az_deg) % 360.0) - 180.0


def resolve_sun_pan(attrs: dict) -> float:
    """
    Return sun azimuth in Moog-pan convention.

    Two dataset conventions:
      * 2026-style: Sun Position Azimuth is meteorological → apply azimuth_to_moog_pan().
      * 2025-style: Sun Position Azimuth already Moog (≈ Pan) → use directly.
    Detection: if |Pan - Sun_Position_Azimuth| < 2°, already Moog.
    """
    pan    = float(attrs.get("Moog Actual Pan [deg]", attrs["Pan"]))
    sun_az = float(attrs["Sun Position Azimuth"])
    if abs(pan - sun_az) < 2.0:
        return sun_az
    return azimuth_to_moog_pan(sun_az)


def select_tilt(attrs: dict) -> float:
    """Priority: Post Capture Actual → Actual → Geometry Tilt Used → Tilt."""
    for key in ("Moog Post Capture Actual Tilt [deg]",
                "Moog Actual Tilt [deg]",
                "Geometry Tilt Used [deg]",
                "Tilt"):
        if key in attrs:
            return float(attrs[key])
    raise KeyError("No tilt key found")


def _eval_phi(
    U_s: np.ndarray, Q_s: np.ndarray, I_s: np.ndarray,
    cy_in_band: int, phi_deg: float,
    lo: int, hi: int, cx: int, thresh: float,
) -> tuple[float, float, int]:
    """
    Evaluate slope² of the U'=0 crossing locus at a single roll angle φ.

    Returns (slope², col_at_cy, n_rows_used).  slope²=inf if not enough rows.
    """
    phi    = np.radians(phi_deg)
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    U_prime = -Q_s * s2 + U_s * c2
    u = np.where(I_s > thresh, U_prime / np.maximum(I_s, 1e-9), np.nan)

    u_l = u[:, lo : hi - 1]
    u_r = u[:, lo + 1 : hi]
    sc  = np.isfinite(u_l) & np.isfinite(u_r) & (u_l * u_r < 0)
    has = sc.any(axis=1)
    valid_rows = np.where(has)[0]
    if len(valid_rows) < 10:
        return np.inf, float(cx), 0

    cross_cols = []
    for r in valid_rows:
        idxs = np.where(sc[r])[0]
        bi   = idxs[np.argmin(np.abs(idxs + lo - cx))]
        ul   = abs(u[r, lo + bi])
        ur_  = abs(u[r, lo + bi + 1])
        dn   = ul + ur_
        col  = lo + bi + (ul / dn if dn > 0 else 0.5)
        cross_cols.append((float(r), col))

    rs = np.array([x[0] for x in cross_cols])
    cs = np.array([x[1] for x in cross_cols])
    med_c = np.median(cs)
    keep  = np.abs(cs - med_c) < CLUSTER_PIX
    if keep.sum() < 10:
        return np.inf, float(cx), 0
    rs, cs = rs[keep], cs[keep]

    slope, intercept = np.polyfit(rs, cs, 1)
    return slope ** 2, float(slope * cy_in_band + intercept), int(keep.sum())


def sweep_roll_and_az(
    U_s: np.ndarray,
    Q_s: np.ndarray,
    I_s: np.ndarray,
    cy_in_band: int,
) -> tuple[float | None, float | None, int]:
    """
    Find camera roll angle and azimuth offset by rotating Stokes parameters
    until the U'=0 neutral line is most vertical.

        U'(φ) = -Q·sin(2φ) + U·cos(2φ)

    Two-pass coarse→fine search with parabolic interpolation:
      1. Coarse: PHI_COARSE_STEPS values over ±PHI_HALF_DEG  (≈0.5° step)
      2. Fine:   PHI_FINE_STEPS  values over ±PHI_FINE_HALF around coarse min (≈0.02° step)
      3. Parabolic interpolation on the three points around the fine minimum
         → sub-step precision (< 0.01°)

    Returns
    -------
    roll_deg : camera roll angle (°), or None.
    az_col   : column of the U'=0 line at the image centre row, or None.
    n_used   : number of rows contributing to the best fit.
    """
    n_rows, n_cols = U_s.shape
    cx        = n_cols // 2
    half_srch = n_cols // 4
    lo        = cx - half_srch
    hi        = cx + half_srch
    thresh    = np.nanpercentile(I_s, I_PERCENTILE_MASK)

    # ── Pass 1: coarse sweep ──────────────────────────────────────────────
    phi_coarse = np.linspace(-PHI_HALF_DEG, PHI_HALF_DEG, PHI_COARSE_STEPS)
    s2_coarse  = [_eval_phi(U_s, Q_s, I_s, cy_in_band, p,
                            lo, hi, cx, thresh)[0] for p in phi_coarse]
    idx_c      = int(np.argmin(s2_coarse))
    phi_c      = float(phi_coarse[idx_c])

    # ── Pass 2: fine sweep around coarse minimum ──────────────────────────
    phi_fine  = np.linspace(phi_c - PHI_FINE_HALF, phi_c + PHI_FINE_HALF,
                             PHI_FINE_STEPS)
    fine_eval = [_eval_phi(U_s, Q_s, I_s, cy_in_band, p,
                           lo, hi, cx, thresh) for p in phi_fine]
    s2_fine   = [e[0] for e in fine_eval]
    idx_f     = int(np.argmin(s2_fine))
    phi_f     = float(phi_fine[idx_f])
    col_f     = fine_eval[idx_f][1]
    n_f       = fine_eval[idx_f][2]

    if n_f == 0:
        return None, None, 0

    # ── Pass 3: parabolic interpolation for sub-step precision ────────────
    phi_opt = phi_f
    if 0 < idx_f < len(phi_fine) - 1:
        x1, y1 = phi_fine[idx_f - 1], s2_fine[idx_f - 1]
        x2, y2 = phi_fine[idx_f],     s2_fine[idx_f]
        x3, y3 = phi_fine[idx_f + 1], s2_fine[idx_f + 1]
        denom  = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(denom) > 1e-12:
            phi_interp = (x1**2*(y2 - y3) + x2**2*(y3 - y1) + x3**2*(y1 - y2)) / denom
            # Accept only if within the fine range (guards against noisy parabola)
            if phi_c - PHI_FINE_HALF <= phi_interp <= phi_c + PHI_FINE_HALF:
                phi_opt = phi_interp
                # Re-evaluate col_at_cy at the interpolated angle
                _, col_f, _ = _eval_phi(U_s, Q_s, I_s, cy_in_band, phi_opt,
                                        lo, hi, cx, thresh)

    return phi_opt, col_f, n_f


def measure_pointing(h5_path: str,
                     aq_name: str,
                     pix_scale: float = PIX_SCALE) -> dict | None:
    """
    Measure azimuth offset and roll for one acquisition.

    Loads U, Q, I in a centre band (±BAND_HALF_ROWS rows), smooths them,
    then calls sweep_roll_and_az.  Q_corrected is used if present; otherwise
    Q is treated as zero (graceful fallback for datasets without Q).
    """
    with h5py.File(h5_path, "r") as h:
        grp      = h[aq_name]
        attrs    = dict(grp.attrs)
        uv       = grp["UV Image Data"]
        h_img, w = uv["U_corrected"].shape[:2]
        cy       = h_img // 2
        cx       = w // 2
        r0       = max(0, cy - BAND_HALF_ROWS)
        r1       = min(h_img, cy + BAND_HALF_ROWS)
        U_b      = uv["U_corrected"][r0:r1, :]
        I_b      = uv["I_corrected"][r0:r1, :]
        Q_b      = (uv["Q_corrected"][r0:r1, :]
                    if "Q_corrected" in uv else np.zeros_like(U_b))

    try:
        moog_pan   = float(attrs.get("Moog Actual Pan [deg]", attrs["Pan"]))
        pan_offset = float(attrs.get("Pan Offset", 0.0))
        moog_tilt  = select_tilt(attrs)
        tilt_offset = float(attrs.get("Tilt Offset", 0.0))
        sun_alt    = float(attrs["Sun Position Altitude"])
    except (KeyError, ValueError):
        return None

    sun_pan = resolve_sun_pan(attrs)
    u_x_exp = moog_pan + pan_offset - sun_pan
    u_y_exp = moog_tilt + tilt_offset - sun_alt
    dtilt   = moog_tilt - sun_alt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U_s = gaussian_filter(U_b.astype(np.float64), SIGMA_SMOOTH)
        Q_s = gaussian_filter(Q_b.astype(np.float64), SIGMA_SMOOTH)
        I_s = gaussian_filter(I_b.astype(np.float64), SIGMA_SMOOTH)

    cy_in_band = cy - r0

    roll_deg, az_col, n_used = sweep_roll_and_az(U_s, Q_s, I_s, cy_in_band)

    detected  = az_col is not None
    az_offset = (az_col - cx) * pix_scale if detected else None

    return {
        "moog_pan":      moog_pan,
        "pan_offset":    pan_offset,
        "moog_tilt":     moog_tilt,
        "sun_alt":       sun_alt,
        "dtilt":         dtilt,
        "u_x_expected":  u_x_exp,
        "u_y_expected":  u_y_exp,
        "detected":      detected,
        "az_offset_deg": az_offset,
        "roll_deg":      roll_deg,
        "n_rows_used":   n_used,
    }


def process_file(h5_path: Path,
                 max_dtilt: float = MAX_DTILT_DEG) -> list[dict]:
    """Process all Aquistion_* groups with |Δtilt| < max_dtilt."""
    results = []
    with h5py.File(h5_path, "r") as h:
        aq_names = sorted(
            [k for k in h.keys() if k.startswith("Aquistion_")],
            key=lambda x: int(x.split("_")[-1]),
        )

    for name in aq_names:
        with h5py.File(h5_path, "r") as h:
            attrs = dict(h[name].attrs)
        try:
            moog_tilt = select_tilt(attrs)
            sun_alt   = float(attrs["Sun Position Altitude"])
        except (KeyError, ValueError):
            continue

        if abs(moog_tilt - sun_alt) > max_dtilt:
            continue

        res = measure_pointing(str(h5_path), name)
        if res is None:
            continue

        row = {
            "h5_file":       h5_path.name,
            "acquisition":   name,
            "moog_pan":      round(res["moog_pan"], 3),
            "pan_offset":    round(res["pan_offset"], 3),
            "moog_tilt":     round(res["moog_tilt"], 3),
            "sun_alt":       round(res["sun_alt"], 3),
            "dtilt":         round(res["dtilt"], 3),
            "u_x_expected":  round(res["u_x_expected"], 4),
            "u_y_expected":  round(res["u_y_expected"], 4),
            "detected":      res["detected"],
            "az_offset_deg": round(res["az_offset_deg"], 4)
                             if res["az_offset_deg"] is not None else None,
            "roll_deg":      round(res["roll_deg"], 4)
                             if res["roll_deg"] is not None else None,
            "n_rows_used":   res["n_rows_used"],
        }
        results.append(row)
    return results


def fit_pointing_model(results: list[dict]) -> dict:
    """
    Fit multi-acquisition azimuth pointing model:
        az_offset_deg = Δpitch·cos(pan) + Δtilt_az·sin(pan) + C_az

    Standard alt-az convention:
        az_err  = -δ_N·cos(pan) + δ_W·sin(pan)/cos(el) + C_az
    so  Δpitch = -δ_N,  Δtilt_az = δ_W / cos(el_mean).
    """
    valid = [r for r in results if r.get("az_offset_deg") is not None]
    fit   = {"n": len(valid)}

    if len(valid) >= 3:
        pan_rad = np.radians([r["moog_pan"] for r in valid])
        az_meas = np.array([r["az_offset_deg"] for r in valid])
        A = np.column_stack([np.cos(pan_rad), np.sin(pan_rad),
                             np.ones_like(pan_rad)])
        coeff, _, _, _ = np.linalg.lstsq(A, az_meas, rcond=None)
        pred  = A @ coeff
        resid = az_meas - pred
        fit.update({
            "delta_pitch":   float(coeff[0]),
            "delta_tilt_az": float(coeff[1]),
            "C_az":          float(coeff[2]),
            "az_rms":        float(np.sqrt(np.mean(resid ** 2))),
            "az_meas":       az_meas.tolist(),
            "az_pred":       pred.tolist(),
            "az_resid":      resid.tolist(),
            "pans":          [r["moog_pan"] for r in valid],
        })

    return fit


def fit_roll_model(results: list[dict]) -> dict:
    """
    Fit sinusoidal model to φ_optimal(pan):
        φ(pan) = B1·sin(pan) + B2·cos(pan) + C_roll

    Physical interpretation (alt-az pointing model):
        roll(pan) = δ_N·cos(pan) - δ_W·sin(pan) + δ_cam
    so  B2 ≈  δ_N ≈ -Δpitch  (cross-check with az_offset fit)
        B1 ≈ -δ_W
        C_roll = δ_cam  (constant mechanical camera roll)

    Uses iterative 2.5-sigma outlier rejection so a single anomalous
    acquisition (e.g. cloud cover disrupting the neutral-line pattern)
    does not bias the sinusoidal fit.

    Returns
    -------
    B1_sin, B2_cos, C_roll : fit coefficients
    amplitude, phase_deg   : A·sin(pan + ψ) form
    roll_rms               : fit RMS (deg)
    n_outliers             : number of points rejected
    """
    valid = [r for r in results if r.get("roll_deg") is not None]
    if len(valid) < 3:
        return {}

    pan_rad = np.radians([r["moog_pan"] for r in valid])
    rolls   = np.array([r["roll_deg"] for r in valid])
    pans    = np.array([r["moog_pan"] for r in valid])

    def _do_fit(pr, rs):
        A = np.column_stack([np.sin(pr), np.cos(pr), np.ones_like(pr)])
        coeff, _, _, _ = np.linalg.lstsq(A, rs, rcond=None)
        return A, coeff

    # Iterative 2.5-sigma outlier rejection (max 3 passes)
    mask = np.ones(len(rolls), dtype=bool)
    n_outliers = 0
    for _pass in range(3):
        if mask.sum() < 3:
            break
        A_m, coeff = _do_fit(pan_rad[mask], rolls[mask])
        A_all = np.column_stack([np.sin(pan_rad), np.cos(pan_rad),
                                 np.ones_like(pan_rad)])
        pred_all  = A_all @ coeff
        resid_all = rolls - pred_all
        rms_in    = float(np.sqrt(np.mean(resid_all[mask] ** 2)))
        new_mask  = mask & (np.abs(resid_all) <= 2.5 * rms_in)
        if new_mask.sum() == mask.sum():
            break   # converged
        n_outliers += int((mask & ~new_mask).sum())
        mask = new_mask

    A_m, coeff = _do_fit(pan_rad[mask], rolls[mask])
    B1, B2, C_roll = coeff

    A_all = np.column_stack([np.sin(pan_rad), np.cos(pan_rad),
                             np.ones_like(pan_rad)])
    pred_all  = A_all @ coeff
    resid_all = rolls - pred_all

    amplitude = float(np.sqrt(B1 ** 2 + B2 ** 2))
    phase_deg = float(np.degrees(np.arctan2(B1, B2)))

    return {
        "B1_sin":      float(B1),
        "B2_cos":      float(B2),
        "C_roll":      float(C_roll),
        "amplitude":   amplitude,
        "phase_deg":   phase_deg,
        "roll_rms":    float(np.sqrt(np.mean(resid_all[mask] ** 2))),
        "n":           int(mask.sum()),
        "n_outliers":  n_outliers,
        "pans":        pans.tolist(),
        "rolls_meas":  rolls.tolist(),
        "rolls_pred":  pred_all.tolist(),
        "inlier_mask": mask.tolist(),
    }


def compute_corrections(results: list[dict],
                        az_fit: dict,
                        roll_fit: dict) -> list[dict]:
    """
    Derive δ_N, δ_W from az_fit; compute tilt_corr and pan_corr.

    Tilt (elevation) error model derived from az_fit:
        tilt_err(pan) = δ_N·sin(pan) + δ_W·cos(pan)
                      = -Δpitch·sin(pan) + Δtilt_az·cos(el_mean)·cos(pan)

    Pan correction for pan-axis north tilt δ_N:
        pan_corr = pan_raw + δ_N·cos(pan) / cos(tilt)   [diagram formula]

    Camera roll δ_cam from roll_fit["C_roll"] — reported separately;
    it rotates the image but does not shift the pointing direction.
    """
    if "delta_pitch" not in az_fit or "C_roll" not in roll_fit:
        return []

    delta_pitch   = az_fit["delta_pitch"]     # = -δ_N
    delta_tilt_az = az_fit["delta_tilt_az"]   # = δ_W / cos(el_mean)
    delta_cam     = roll_fit["C_roll"]         # constant image roll

    el_mean = float(np.mean([r["sun_alt"] for r in results]))
    delta_N = -delta_pitch
    delta_W = delta_tilt_az * np.cos(np.radians(el_mean))

    corrections = []
    for r in results:
        pan_rad  = np.radians(r["moog_pan"])
        tilt_deg = r["moog_tilt"]
        cos_tilt = max(np.cos(np.radians(tilt_deg)), 0.1)

        tilt_err_model = (delta_N * np.sin(pan_rad)
                          + delta_W * np.cos(pan_rad))
        tilt_corr = tilt_deg - tilt_err_model
        pan_corr  = r["moog_pan"] + delta_N * np.cos(pan_rad) / cos_tilt

        corrections.append({
            "h5_file":        r["h5_file"],
            "acquisition":    r["acquisition"],
            "moog_pan":       r["moog_pan"],
            "moog_tilt":      tilt_deg,
            "sun_alt":        r["sun_alt"],
            "tilt_err_model": round(tilt_err_model, 4),
            "tilt_corr":      round(tilt_corr, 4),
            "pan_corr":       round(pan_corr, 4),
            "delta_cam_roll": round(delta_cam, 4),
        })

    return corrections


def _fit_circle_from_points(rows: np.ndarray, cols: np.ndarray) -> dict:
    """
    Algebraic least-squares circle fit.
      2x·cx + 2y·cy + c = x² + y²

    Returns dict with keys: x (cx), y (cy), radius, rmse, n_points.
    """
    x, y = cols.astype(float), rows.astype(float)
    A    = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b    = x * x + y * y
    res  = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = res[0]
    r2        = c + cx * cx + cy * cy
    if r2 < 0:
        return {"x": cx, "y": cy, "radius": 0.0, "rmse": np.inf, "n_points": len(rows)}
    radius   = float(np.sqrt(r2))
    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    rmse     = float(np.sqrt(np.mean(residual ** 2)))
    return {"x": float(cx), "y": float(cy), "radius": radius,
            "rmse": rmse, "n_points": int(len(rows))}


def measure_disk_centroid(h5_path: str,
                          aq_name: str,
                          pix_scale: float = PIX_SCALE) -> dict | None:
    """
    Find solar disk centre using algebraic circle fit on I_corrected.

    Adaptively thresholds bright pixels (fraction of max_intensity) from
    0.95 down to 0.70, accepts the first result that passes quality checks:
      n_points >= 20, 20 <= radius <= 500 px, rmse <= max(25, 0.35*radius)

    Coordinates are converted to pointing offsets from image centre:
      zen_offset_deg > 0  → sun below image centre (camera over-elevated)
      az_offset_disk > 0  → sun right of image centre (camera pointing left)
    """
    with h5py.File(h5_path, "r") as h:
        grp      = h[aq_name]
        uv       = grp["UV Image Data"]
        h_img, w = uv["I_corrected"].shape[:2]
        cy_full  = h_img // 2
        cx_full  = w // 2
        r0       = max(0, cy_full - DISK_HALF_ROWS)
        r1       = min(h_img, cy_full + DISK_HALF_ROWS)
        I_b      = uv["I_corrected"][r0:r1, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        I_s = gaussian_filter(I_b.astype(np.float64), sigma=5)

    I_max = float(np.nanmax(I_s))
    if I_max <= 0:
        return None

    best_fit = None
    for frac in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        thresh   = frac * I_max
        mask     = I_s > thresh
        # Use only the boundary of the mask (limb pixels) so the algebraic
        # circle fit sees the disk EDGE, not the filled interior.
        # Border = mask AND NOT eroded(mask)
        border = mask & ~binary_erosion(mask)
        rows_px, cols_px = np.where(border)
        if len(rows_px) < 20:
            continue
        fit = _fit_circle_from_points(rows_px, cols_px)
        r   = fit["radius"]
        if (fit["n_points"] >= 20
                and 20 <= r <= 500
                and fit["rmse"] <= max(25.0, 0.35 * r)):
            best_fit = fit
            break   # accept first passing threshold

    if best_fit is None:
        return None

    # Convert to full-image coordinates (band starts at r0)
    cx_sun = best_fit["x"]                    # column in band = column in image
    cy_sun = best_fit["y"] + r0               # row in full image

    return {
        "row_centroid":   cy_sun,
        "col_centroid":   cx_sun,
        "disk_radius_px": best_fit["radius"],
        "disk_rmse_px":   best_fit["rmse"],
        "zen_offset_deg": (cy_sun - cy_full) * pix_scale,
        "az_offset_disk": (cx_sun - cx_full) * pix_scale,
        "disk_detected":  True,
    }


def write_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_table(results: list[dict]) -> None:
    print()
    hdr = (f"{'File':<28} {'Aq':>5} {'pan':>6} {'Dtilt':>6} "
           f"{'u_x_exp':>8} {'az_off':>8} {'roll':>7} {'n_rows':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        az  = f"{r['az_offset_deg']:+8.4f}" if r["az_offset_deg"] is not None else "     ---"
        rol = f"{r['roll_deg']:+7.3f}"       if r["roll_deg"]      is not None else "    ---"
        aq_n = r["acquisition"].split("_")[-1]
        print(f"{r['h5_file'][-28:]:<28} {aq_n:>5} {r['moog_pan']:6.1f} "
              f"{r['dtilt']:+6.2f} {r['u_x_expected']:+8.3f} "
              f"{az} {rol} {r['n_rows_used']:>6}")


def main(data_dir: str | None = None) -> None:
    if data_dir is None:
        data_dir = "/Volumes/LaCie/Level2 data/2026_06_05"
    data_path = Path(data_dir)

    h5_files = sorted(
        f for f in data_path.glob("*.h5")
        if "_calibration" not in f.name and not f.name.startswith("._")
    )
    if not h5_files:
        print(f"No H5 files found in {data_path}")
        return

    # ── Step 1: per-acquisition U rotation sweep ─────────────────────────────
    print(f"Processing {len(h5_files)} files  "
          f"(|Δtilt| < {MAX_DTILT_DEG}°, roll sweep ±{PHI_HALF_DEG}°, "
          f"coarse {PHI_COARSE_STEPS} + fine {PHI_FINE_STEPS} steps) …")
    all_results = []
    for h5f in h5_files:
        rows = process_file(h5f)
        n_ok = sum(1 for r in rows if r["detected"])
        print(f"  {h5f.name[-32:]}: {len(rows)} candidate(s), {n_ok} detected")
        all_results.extend(rows)

    if not all_results:
        print("No results found.")
        return

    print_table(all_results)

    out_csv = data_path / "u_neutral_line_results.csv"
    write_csv(all_results, out_csv)
    print(f"\nResults written to: {out_csv}")

    # ── Step 2a: azimuth sinusoidal fit ──────────────────────────────────────
    az_fit = fit_pointing_model(all_results)

    print("\n" + "=" * 60)
    print("AZIMUTH FIT   az_err = Δpitch·cos(pan) + Δtilt_az·sin(pan) + C")
    print("=" * 60)
    if "delta_pitch" in az_fit:
        print(f"  n={az_fit['n']}   RMS={az_fit['az_rms']:.4f} deg")
        print(f"  Δpitch   = {az_fit['delta_pitch']:+.4f} deg  → δ_N = {-az_fit['delta_pitch']:+.4f} deg")
        print(f"  Δtilt_az = {az_fit['delta_tilt_az']:+.4f} deg")
        print(f"  C_az     = {az_fit['C_az']:+.4f} deg")
        print()
        for pan, meas, pred, resid in zip(
            az_fit["pans"], az_fit["az_meas"],
            az_fit["az_pred"], az_fit["az_resid"]
        ):
            print(f"  pan={pan:7.1f}  meas={meas:+.4f}  pred={pred:+.4f}  "
                  f"resid={resid:+.4f}")

    # ── Step 2b: roll sinusoidal fit ─────────────────────────────────────────
    roll_fit = fit_roll_model(all_results)

    print("\n" + "=" * 60)
    print("ROLL FIT   φ(pan) = B1·sin(pan) + B2·cos(pan) + C_roll")
    print("=" * 60)
    if "C_roll" in roll_fit:
        n_out = roll_fit.get("n_outliers", 0)
        print(f"  n={roll_fit['n']} (rejected {n_out} outlier(s) at 2.5σ)  "
              f"RMS={roll_fit['roll_rms']:.4f} deg")
        print(f"  B1 (sin) = {roll_fit['B1_sin']:+.4f} deg  ≈ -δ_W")
        print(f"  B2 (cos) = {roll_fit['B2_cos']:+.4f} deg  ≈  δ_N  "
              f"(az-fit gives δ_N={-az_fit.get('delta_pitch',0):+.4f})")
        print(f"  C_roll   = {roll_fit['C_roll']:+.4f} deg  ← constant camera roll δ_cam")
        print(f"  Amplitude= {roll_fit['amplitude']:.4f} deg,  Phase={roll_fit['phase_deg']:+.1f}°")
        print()
        inlier_mask = roll_fit.get("inlier_mask", [True] * len(roll_fit["pans"]))
        for pan, meas, pred, inlier in zip(
            roll_fit["pans"], roll_fit["rolls_meas"], roll_fit["rolls_pred"], inlier_mask
        ):
            flag = "" if inlier else "  ← OUTLIER"
            print(f"  pan={pan:7.1f}  φ_meas={meas:+7.3f}  φ_pred={pred:+7.3f}  "
                  f"resid={meas-pred:+.3f}{flag}")

    # ── Step 3: compute corrections ───────────────────────────────────────────
    corrections = compute_corrections(all_results, az_fit, roll_fit)

    print("\n" + "=" * 60)
    print("CORRECTIONS   (tilt_err = δ_N·sin(pan) + δ_W·cos(pan))")
    print("=" * 60)
    if corrections:
        hdr = (f"{'File':<28} {'pan':>6} {'tilt_raw':>9} {'sun_alt':>8} "
               f"{'tilt_err':>9} {'tilt_corr':>10} {'pan_corr':>9}")
        print(hdr)
        print("-" * len(hdr))
        for c in corrections:
            print(f"{c['h5_file'][-28:]:<28} {c['moog_pan']:6.1f} "
                  f"{c['moog_tilt']:9.3f} {c['sun_alt']:8.3f} "
                  f"{c['tilt_err_model']:+9.4f} {c['tilt_corr']:10.4f} "
                  f"{c['pan_corr']:9.3f}")
        corr_csv = data_path / "u_neutral_line_corrections.csv"
        with open(corr_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(corrections[0].keys()))
            writer.writeheader()
            writer.writerows(corrections)
        print(f"\nCorrections written to: {corr_csv}")

    # ── Step 4: disk centroid validation (zenith) ─────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION: disk centroid zen_offset vs tilt_err_model")
    print("=" * 60)
    hdr2 = (f"{'File':<28} {'pan':>6} {'zen_disk':>9} "
            f"{'tilt_model':>11} {'resid':>7} {'az_disk':>8}")
    print(hdr2)
    print("-" * len(hdr2))
    zen_resids = []
    for r, c in zip(all_results, corrections):
        if not r["detected"]:
            continue
        disk = measure_disk_centroid(
            str(data_path / r["h5_file"]), r["acquisition"]
        )
        if disk is None:
            print(f"  {r['h5_file'][-28:]:<28}  disk not detected")
            continue
        resid = disk["zen_offset_deg"] - c["tilt_err_model"]
        zen_resids.append(resid)
        print(f"{r['h5_file'][-28:]:<28} {r['moog_pan']:6.1f} "
              f"{disk['zen_offset_deg']:+9.4f} "
              f"{c['tilt_err_model']:+11.4f} "
              f"{resid:+7.4f} "
              f"{disk['az_offset_disk']:+8.4f}")

    if zen_resids:
        print(f"\n  Disk zen residuals — "
              f"mean={np.mean(zen_resids):+.4f}°  "
              f"std={np.std(zen_resids):.4f}°  "
              f"RMS={np.sqrt(np.mean(np.array(zen_resids)**2)):.4f}°")
    print("=" * 60)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_dir)

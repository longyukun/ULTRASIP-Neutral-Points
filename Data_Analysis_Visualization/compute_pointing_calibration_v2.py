import argparse
import csv
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

from robust_np_all_acquisitions import (
    UV_DEG_PER_PIXEL,
    calibration_dtilt_from_attrs,
    detect_sun_center_raw,
    solar_altitude_deg,
    sun_is_complete,
)


DEFAULT_DIR = Path("/Volumes/LaCie/Level2 data/2026_06_05")


def is_real_h5(path):
    return path.is_file() and path.suffix.lower() in (".h5", ".hdf5") and not path.name.startswith("._")


def parse_timestamp(attrs):
    ts = str(attrs.get("Timestamp Local New") or attrs.get("Timestamp UTC New", ""))
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def acquisition_names(handle):
    return sorted(
        (n for n in handle.keys() if n.startswith("Aquistion_")),
        key=lambda name: int(name.split("_")[-1]),
    )


def calibration_point(handle):
    """Return one (pan, tilt_err) calibration point for this H5 file.

    Prefers the calibration_acqui_* frames (same approach as
    compute_calibration_tilt_correction). If a folder/file has no
    calibration frames, falls back to Aquistion_1 as the calibration frame.
    """
    meta = handle["Measurement_Metadata"]
    aq0 = handle["Aquistion_0"]
    lat = float(meta.attrs["Latitude"])
    lon = float(meta.attrs["Longitude"])
    t0 = parse_timestamp(aq0.attrs)
    alt0 = solar_altitude_deg(t0, lat, lon)
    px_per_deg = 1.0 / UV_DEG_PER_PIXEL

    calib_names = sorted(n for n in handle.keys() if n.startswith("calibration_acqui_"))
    source = "calibration"
    if not calib_names:
        calib_names = ["Aquistion_1"]
        source = "acquisition_1"

    corrections = []
    pans = []
    for name in calib_names:
        grp = handle.get(name)
        if grp is None:
            continue
        uv = grp.get("UV Image Data")
        raw_ds = uv.get("UV Raw Images") if uv is not None else None
        if raw_ds is None:
            continue
        raw = raw_ds[:]
        frame = raw[0] if raw.ndim == 3 else raw if raw.ndim == 2 else None
        if frame is None:
            continue
        h, w = frame.shape
        cy0 = h / 2.0
        result = detect_sun_center_raw(frame)
        if result is None:
            continue
        cx, cy = result
        if not sun_is_complete(cx, cy, w, h):
            continue
        dtilt, _ = calibration_dtilt_from_attrs(name, aq0.attrs, grp.attrs)
        if not np.isfinite(dtilt):
            continue
        t_i = parse_timestamp(grp.attrs)
        solar_motion = solar_altitude_deg(t_i, lat, lon) - alt0
        cy_expected = cy0 + dtilt * px_per_deg
        dy = cy - cy_expected
        corr_at_frame = dy * UV_DEG_PER_PIXEL
        corrections.append(corr_at_frame - solar_motion)
        pans.append(float(grp.attrs.get("Geometry Pan Used [deg]", aq0.attrs.get("Geometry Pan Used [deg]"))))

    if not corrections:
        return None

    return {
        "pan_deg": float(np.mean(pans)),
        "tilt_err_deg": float(np.mean(corrections)),
        "source": source,
        "n_used": len(corrections),
        "n_total": len(calib_names),
    }


def sinusoid(pan_deg, A, phi, C):
    return A * np.sin(np.radians(pan_deg) + phi) + C


def fit_sinusoid(pans, tilt_errs):
    pans = np.asarray(pans, dtype=float)
    tilt_errs = np.asarray(tilt_errs, dtype=float)
    p0 = [0.1, 0.0, float(np.mean(tilt_errs))]
    popt, pcov = curve_fit(sinusoid, pans, tilt_errs, p0=p0, maxfev=20000)
    pred = sinusoid(pans, *popt)
    ss_res = float(np.sum((tilt_errs - pred) ** 2))
    ss_tot = float(np.sum((tilt_errs - np.mean(tilt_errs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return popt, pcov, r2


def apply_frame_correction(attrs, A, phi, C, delta_roll, tilt_err_v3):
    """Compute v2 (folder-wide sinusoid) and v3 (this file's calibration frames only)
    corrected geometry for one acquisition.

    Sign convention: zenith_vN = zenith_raw - tilt_err_vN, tilt_vN = tilt_raw + tilt_err_vN
    (consistent with the existing "Apply tilt corr" checkbox, which does
    view_zen = view_zen - calib_tilt_correction_deg, the same quantity as tilt_err_pred).

    v3 uses this file's own calibration-derived zenith offset (tilt_err_v3) instead
    of the folder-wide sinusoid prediction, but reuses the folder-wide roll/pan
    correction (camera_roll, pan_v) since that requires the cross-file fit.
    """
    pan_raw = float(attrs.get("Geometry Pan Used [deg]"))
    tilt_raw = float(attrs.get("Geometry Tilt Used [deg]"))
    pan_rad = np.radians(pan_raw)
    tilt_rad = np.radians(tilt_raw)

    tilt_err_pred = float(sinusoid(pan_raw, A, phi, C))
    tilt_v2 = tilt_raw + tilt_err_pred
    zenith_v2 = 90.0 - tilt_v2
    pan_v2 = pan_raw + delta_roll * np.cos(pan_rad) / np.cos(tilt_rad)
    camera_roll_v2 = A * np.cos(pan_rad + phi) / np.cos(tilt_rad)

    tilt_v3 = tilt_raw + tilt_err_v3
    zenith_v3 = 90.0 - tilt_v3
    # v3 reuses the cross-file roll/pan correction (it cannot be derived from a
    # single per-file calibration point), but uses this file's own zenith offset.
    pan_v3 = pan_v2
    camera_roll_v3 = camera_roll_v2

    return {
        "pan_raw_deg": pan_raw,
        "tilt_raw_deg": tilt_raw,
        "zenith_raw_deg": 90.0 - tilt_raw,
        "tilt_err_pred_deg": tilt_err_pred,
        "tilt_v2_deg": tilt_v2,
        "zenith_v2_deg": zenith_v2,
        "pan_v2_deg": float(pan_v2),
        "camera_roll_v2_deg": float(camera_roll_v2),
        "tilt_err_v3_deg": tilt_err_v3,
        "tilt_v3_deg": tilt_v3,
        "zenith_v3_deg": zenith_v3,
        "pan_v3_deg": float(pan_v3),
        "camera_roll_v3_deg": float(camera_roll_v3),
    }


def write_frame_attrs(grp, values):
    grp.attrs["Tilt Error v2 Pred [deg]"] = values["tilt_err_pred_deg"]
    grp.attrs["Tilt_v2 [deg]"] = values["tilt_v2_deg"]
    grp.attrs["Zenith_v2 [deg]"] = values["zenith_v2_deg"]
    grp.attrs["Pan_v2 [deg]"] = values["pan_v2_deg"]
    grp.attrs["Camera Roll v2 [deg]"] = values["camera_roll_v2_deg"]
    grp.attrs["Tilt Error v3 Pred [deg]"] = values["tilt_err_v3_deg"]
    grp.attrs["Tilt_v3 [deg]"] = values["tilt_v3_deg"]
    grp.attrs["Zenith_v3 [deg]"] = values["zenith_v3_deg"]
    grp.attrs["Pan_v3 [deg]"] = values["pan_v3_deg"]
    grp.attrs["Camera Roll v3 [deg]"] = values["camera_roll_v3_deg"]


def write_calibration_group(handle, A, phi, C, delta_pitch, delta_roll, r2, n_points, point):
    if "Pointing_Calibration_v2" in handle:
        del handle["Pointing_Calibration_v2"]
    g = handle.create_group("Pointing_Calibration_v2")
    g.attrs["model"] = "tilt_err(pan_deg) = A * sin(radians(pan_deg) + phi_rad) + C"
    g.attrs["A_deg"] = A
    g.attrs["phi_rad"] = phi
    g.attrs["C_deg"] = C
    g.attrs["delta_pitch_deg"] = delta_pitch
    g.attrs["delta_roll_deg"] = delta_roll
    g.attrs["fit_r2"] = r2
    g.attrs["fit_n_points"] = n_points
    g.attrs["this_file_calibration_source"] = point["source"]
    g.attrs["this_file_calibration_pan_deg"] = point["pan_deg"]
    g.attrs["this_file_calibration_tilt_err_deg"] = point["tilt_err_deg"]
    g.attrs["this_file_calibration_n_used"] = point["n_used"]
    g.attrs["this_file_calibration_n_total"] = point["n_total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a folder-wide sinusoidal tilt_err(pan) calibration from calibration "
            "frames (or Aquistion_1 if a file has none), then write per-frame v2 "
            "(folder-wide sinusoid) and v3 (this file's own calibration frames only "
            "for the zenith offset, reusing the folder-wide roll/pan correction) "
            "corrected zenith/tilt/pan and camera roll back into each H5 file."
        )
    )
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Compute and report only; do not modify H5 files")
    return parser.parse_args()


def main():
    args = parse_args()
    directory = args.directory.expanduser()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Directory does not exist: {directory}")

    files = sorted(p for p in directory.iterdir() if is_real_h5(p))
    if not files:
        raise SystemExit(f"No H5 files in {directory}")

    points = {}
    for path in files:
        with h5py.File(path, "r") as handle:
            point = calibration_point(handle)
        if point is None:
            print(f"WARNING: no usable calibration point for {path.name}", flush=True)
            continue
        points[path] = point
        print(
            f"{path.name}: pan={point['pan_deg']:.3f} deg  tilt_err={point['tilt_err_deg']:.4f} deg  "
            f"source={point['source']} ({point['n_used']}/{point['n_total']})",
            flush=True,
        )

    if len(points) < 3:
        raise SystemExit("Not enough calibration points to fit sinusoid (need >= 3)")

    pans = [p["pan_deg"] for p in points.values()]
    tilt_errs = [p["tilt_err_deg"] for p in points.values()]
    (A, phi, C), pcov, r2 = fit_sinusoid(pans, tilt_errs)
    delta_pitch = A * np.cos(phi)
    delta_roll = A * np.sin(phi)

    print(
        f"\nFit: A={A:.4f} deg  phi={np.degrees(phi):.2f} deg  C={C:.4f} deg  "
        f"R2={r2:.3f}  n={len(points)}",
        flush=True,
    )
    print(f"delta_pitch={delta_pitch:.4f} deg  delta_roll={delta_roll:.4f} deg", flush=True)

    fit_csv = directory / "pointing_calibration_v2_fit.csv"
    with open(fit_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["h5_file", "pan_deg", "tilt_err_deg", "source", "n_used", "n_total"])
        for path, point in points.items():
            writer.writerow([
                path.name, point["pan_deg"], point["tilt_err_deg"],
                point["source"], point["n_used"], point["n_total"],
            ])
        writer.writerow([])
        writer.writerow(["A_deg", "phi_deg", "C_deg", "delta_pitch_deg", "delta_roll_deg", "r2", "n_points"])
        writer.writerow([A, np.degrees(phi), C, delta_pitch, delta_roll, r2, len(points)])
    print(f"Wrote {fit_csv}", flush=True)

    per_frame_rows = []
    for path in files:
        mode = "r" if args.dry_run else "r+"
        with h5py.File(path, mode) as handle:
            point = points.get(path)
            names = acquisition_names(handle)
            calib_names = sorted(n for n in handle.keys() if n.startswith("calibration_acqui_"))
            tilt_err_v3 = point["tilt_err_deg"] if point is not None else float("nan")
            for name in names + calib_names:
                grp = handle[name]
                values = apply_frame_correction(grp.attrs, A, phi, C, delta_roll, tilt_err_v3)
                if not args.dry_run:
                    write_frame_attrs(grp, values)
                row = {"h5_file": path.name, "acquisition": name}
                row.update(values)
                per_frame_rows.append(row)
            if not args.dry_run and point is not None:
                write_calibration_group(handle, A, phi, C, delta_pitch, delta_roll, r2, len(points), point)
        print(f"Processed {path.name} ({len(names) + len(calib_names)} frames)", flush=True)

    per_frame_csv = directory / "pointing_calibration_v2_per_frame.csv"
    with open(per_frame_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)
    print(f"Wrote {per_frame_csv}", flush=True)


if __name__ == "__main__":
    main()

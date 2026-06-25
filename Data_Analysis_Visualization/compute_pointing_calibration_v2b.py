#!/usr/bin/env python3
"""Apply v2b pointing calibration from folder-wide Aquistion_0 raw sun centers.

v2b differs from v2/v3 by using only each measurement file's Aquistion_0 raw
UV images as the calibration observation. It does not require Level0/1/2
products to estimate the calibration point: the sun center is found directly
from raw frames, compared with the raw frame center, and converted to a
boresight zenith error. A folder-wide sinusoid tilt_err(pan) is then fitted and
per-frame v2b attributes are written into each measurement H5.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

from process_single_level0_1_2 import IMG_X, IMG_Y, select_tilt
from robust_np_all_acquisitions import UV_DEG_PER_PIXEL, detect_sun_center_raw


DEFAULT_DIR = Path("/Volumes/LaCie/Level2 data/2026_06_10")


def default_worker_count(item_count):
    cpu_count = os.cpu_count() or 1
    return max(1, min(item_count, max(1, cpu_count - 2)))


def is_measurement_h5(path):
    return (
        path.is_file()
        and path.suffix.lower() in (".h5", ".hdf5")
        and not path.name.startswith("._")
        and not path.name.endswith("_calibration.h5")
    )


def finite_attr(attrs, key):
    try:
        value = float(attrs.get(key, np.nan))
    except Exception:
        value = np.nan
    return value if np.isfinite(value) else np.nan


def select_pan(attrs):
    for key in ("Moog Post Capture Actual Pan [deg]", "Moog Actual Pan [deg]", "Pan"):
        value = finite_attr(attrs, key)
        if np.isfinite(value):
            return value, key
    raise KeyError("No usable pan attribute found")


def acquisition_names(handle):
    names = [n for n in handle.keys() if n.startswith("Aquistion_")]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def frame_group_names(handle):
    names = [
        n for n in handle.keys()
        if n.startswith("Aquistion_") or n.startswith("calibration_acqui_")
    ]

    def sort_key(name):
        if name.startswith("Aquistion_"):
            return (0, int(name.split("_")[-1]))
        parts = name.split("_")
        direction = parts[2] if len(parts) > 2 else ""
        amount = parts[3].replace("p", ".") if len(parts) > 3 else "0"
        try:
            amount_value = float(amount)
        except ValueError:
            amount_value = 0.0
        return (1, 0 if direction == "down" else 1, amount_value)

    return sorted(names, key=sort_key)


def raw_sun_frame(grp):
    uv = grp.get("UV Image Data")
    raw_ds = uv.get("UV Raw Images") if uv is not None else None
    if raw_ds is None:
        return None
    raw = raw_ds[:]
    if raw.ndim == 1 and raw.size == 4 * IMG_Y * IMG_X:
        raw = raw.reshape(4, IMG_Y, IMG_X)
    if raw.ndim == 3:
        return np.nanmean(raw.astype(np.float32), axis=0)
    if raw.ndim == 2:
        return raw.astype(np.float32)
    return None


def calibration_point_from_aq0(handle):
    aq0 = handle.get("Aquistion_0")
    if aq0 is None:
        return None
    frame = raw_sun_frame(aq0)
    if frame is None:
        return None
    result = detect_sun_center_raw(frame)
    if result is None:
        return None
    cx, cy = result
    h, w = frame.shape
    cy0 = h / 2.0
    dy = cy - cy0
    tilt_err = dy * UV_DEG_PER_PIXEL
    pan, pan_source = select_pan(aq0.attrs)
    tilt, tilt_source = select_tilt(aq0.attrs)
    return {
        "pan_deg": float(pan),
        "tilt_deg": float(tilt),
        "tilt_err_deg": float(tilt_err),
        "cx_px": float(cx),
        "cy_px": float(cy),
        "cy_expected_px": float(cy0),
        "dy_px": float(dy),
        "pan_source": pan_source,
        "tilt_source": tilt_source,
        "source": "Aquistion_0 raw UV",
    }


def calibration_point_for_path(path):
    with h5py.File(path, "r") as handle:
        return calibration_point_from_aq0(handle)


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


def frame_pan_tilt_attrs(grp):
    attrs = grp.attrs
    pan = finite_attr(attrs, "Geometry Pan Used [deg]")
    tilt = finite_attr(attrs, "Geometry Tilt Used [deg]")
    pan_source = attrs.get("Geometry Pan Source", "")
    tilt_source = attrs.get("Geometry Tilt Source", "")
    if not np.isfinite(pan):
        pan, pan_source = select_pan(attrs)
        attrs["Geometry Pan Used [deg]"] = float(pan)
        attrs["Geometry Pan Source"] = pan_source
    if not np.isfinite(tilt):
        tilt, tilt_source = select_tilt(attrs)
        attrs["Geometry Tilt Used [deg]"] = float(tilt)
        attrs["Geometry Tilt Source"] = tilt_source
    return float(pan), float(tilt)


def apply_frame_correction(grp, A, phi, C, delta_roll):
    pan_raw, tilt_raw = frame_pan_tilt_attrs(grp)
    pan_rad = np.radians(pan_raw)
    tilt_rad = np.radians(tilt_raw)
    tilt_err_pred = float(sinusoid(pan_raw, A, phi, C))
    tilt_v2b = tilt_raw + tilt_err_pred
    zenith_v2b = 90.0 - tilt_v2b
    pan_v2b = pan_raw + delta_roll * np.cos(pan_rad) / max(np.cos(tilt_rad), 1e-3)
    camera_roll_v2b = A * np.cos(pan_rad + phi) / max(np.cos(tilt_rad), 1e-3)
    return {
        "pan_raw_deg": pan_raw,
        "tilt_raw_deg": tilt_raw,
        "zenith_raw_deg": 90.0 - tilt_raw,
        "tilt_err_pred_deg": tilt_err_pred,
        "tilt_v2b_deg": tilt_v2b,
        "zenith_v2b_deg": zenith_v2b,
        "pan_v2b_deg": float(pan_v2b),
        "camera_roll_v2b_deg": float(camera_roll_v2b),
    }


def write_frame_attrs(grp, values):
    grp.attrs["Tilt Error v2b Pred [deg]"] = values["tilt_err_pred_deg"]
    grp.attrs["Tilt_v2b [deg]"] = values["tilt_v2b_deg"]
    grp.attrs["Zenith_v2b [deg]"] = values["zenith_v2b_deg"]
    grp.attrs["Az_v2b [deg]"] = values["pan_v2b_deg"]  # corrected az at image centre
    grp.attrs["Pan_v2b [deg]"] = values["pan_v2b_deg"]
    grp.attrs["Camera Roll v2b [deg]"] = values["camera_roll_v2b_deg"]


def write_calibration_group(handle, A, phi, C, delta_pitch, delta_roll, r2, points, point):
    if "Pointing_Calibration_v2b" in handle:
        del handle["Pointing_Calibration_v2b"]
    g = handle.create_group("Pointing_Calibration_v2b")
    g.attrs["model"] = "tilt_err(pan_deg) from folder-wide Aquistion_0 raw sun centers"
    g.attrs["A_deg"] = A
    g.attrs["phi_rad"] = phi
    g.attrs["C_deg"] = C
    g.attrs["delta_pitch_deg"] = delta_pitch
    g.attrs["delta_roll_deg"] = delta_roll
    g.attrs["fit_r2"] = r2
    g.attrs["fit_n_points"] = len(points)
    g.attrs["this_file_calibration_source"] = point["source"]
    g.attrs["this_file_calibration_pan_deg"] = point["pan_deg"]
    g.attrs["this_file_calibration_tilt_err_deg"] = point["tilt_err_deg"]
    g.attrs["this_file_calibration_sun_cx_px"] = point["cx_px"]
    g.attrs["this_file_calibration_sun_cy_px"] = point["cy_px"]
    g.attrs["this_file_calibration_dy_px"] = point["dy_px"]
    g.attrs["this_file_calibration_pan_source"] = point["pan_source"]
    g.attrs["this_file_calibration_tilt_source"] = point["tilt_source"]


def process_one_file(path, point, A, phi, C, delta_pitch, delta_roll, r2,
                     points, dry_run=False, position=1):
    mode = "r" if dry_run else "r+"
    per_frame_rows = []
    with h5py.File(path, mode) as handle:
        names = frame_group_names(handle)
        iterator = tqdm(
            names,
            desc=f"{path.name}",
            unit="frame",
            position=position,
            leave=False,
            dynamic_ncols=True,
        )
        for name in iterator:
            values = apply_frame_correction(handle[name], A, phi, C, delta_roll)
            if not dry_run:
                write_frame_attrs(handle[name], values)
            row = {"h5_file": path.name, "acquisition": name}
            row.update(values)
            per_frame_rows.append(row)
        if not dry_run:
            write_calibration_group(handle, A, phi, C, delta_pitch, delta_roll, r2, points, point)
    return per_frame_rows, len(names)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write v2b pointing calibration from folder-wide Aquistion_0 raw sun centers.")
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Compute only; do not modify H5 files")
    parser.add_argument(
        "--only-file",
        type=Path,
        default=None,
        help="Write v2b attributes only to this H5 file; folder-wide fit still uses all usable H5 files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker threads. Default: max(1, CPU count - 2), capped by file count.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    directory = args.directory.expanduser()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Directory does not exist: {directory}")

    files = sorted(p for p in directory.iterdir() if is_measurement_h5(p))
    if not files:
        raise SystemExit(f"No measurement H5 files in {directory}")
    target_files = files
    if args.only_file is not None:
        only = args.only_file.expanduser()
        if not only.is_absolute():
            only = directory / only
        only = only.resolve()
        target_files = [p for p in files if p.resolve() == only]
        if not target_files:
            raise SystemExit(f"--only-file is not a measurement H5 in {directory}: {only}")
    workers = (
        max(1, min(args.workers, len(files)))
        if args.workers and args.workers > 0
        else default_worker_count(len(files))
    )
    target_workers = max(1, min(workers, len(target_files)))
    print(
        f"Using {workers} worker(s) for fit points and {target_workers} worker(s) for writes",
        flush=True,
    )

    points = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(calibration_point_for_path, path): path for path in files}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc="Fit points",
            unit="file",
            position=0,
            dynamic_ncols=True,
        ):
            path = future_to_path[future]
            try:
                point = future.result()
            except Exception as exc:
                print(f"WARNING: failed Aquistion_0 raw point for {path.name}: {exc}", flush=True)
                continue
            if point is None:
                print(f"WARNING: no usable Aquistion_0 raw sun point for {path.name}", flush=True)
                continue
            points[path] = point
            print(
                f"{path.name}: pan={point['pan_deg']:.3f} deg  "
                f"tilt_err={point['tilt_err_deg']:.4f} deg  "
                f"sun=({point['cx_px']:.1f},{point['cy_px']:.1f}) raw aq0",
                flush=True,
            )

    if len(points) < 3:
        raise SystemExit("Not enough Aquistion_0 raw sun points to fit v2b sinusoid (need >= 3)")

    pans = [p["pan_deg"] for p in points.values()]
    tilt_errs = [p["tilt_err_deg"] for p in points.values()]
    (A, phi, C), pcov, r2 = fit_sinusoid(pans, tilt_errs)
    delta_pitch = A * np.cos(phi)
    delta_roll = A * np.sin(phi)

    print(
        f"\nv2b Fit: A={A:.4f} deg  phi={np.degrees(phi):.2f} deg  "
        f"C={C:.4f} deg  R2={r2:.3f}  n={len(points)}",
        flush=True,
    )
    print(f"v2b delta_pitch={delta_pitch:.4f} deg  delta_roll={delta_roll:.4f} deg", flush=True)

    fit_csv = directory / "pointing_calibration_v2b_fit.csv"
    if not args.dry_run:
        with open(fit_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "h5_file", "pan_deg", "tilt_deg", "tilt_err_deg",
                "cx_px", "cy_px", "cy_expected_px", "dy_px",
                "pan_source", "tilt_source",
            ])
            for path, point in points.items():
                writer.writerow([
                    path.name, point["pan_deg"], point["tilt_deg"], point["tilt_err_deg"],
                    point["cx_px"], point["cy_px"], point["cy_expected_px"], point["dy_px"],
                    point["pan_source"], point["tilt_source"],
                ])
            writer.writerow([])
            writer.writerow(["A_deg", "phi_deg", "C_deg", "delta_pitch_deg", "delta_roll_deg", "r2", "n_points"])
            writer.writerow([A, np.degrees(phi), C, delta_pitch, delta_roll, r2, len(points)])
        print(f"Wrote {fit_csv}", flush=True)
    else:
        print(f"Dry run: not writing {fit_csv}", flush=True)

    per_frame_rows = []
    write_inputs = [(path, points.get(path), idx + 1) for idx, path in enumerate(target_files) if points.get(path)]
    with ThreadPoolExecutor(max_workers=target_workers) as executor:
        future_to_path = {
            executor.submit(
                process_one_file,
                path,
                point,
                A,
                phi,
                C,
                delta_pitch,
                delta_roll,
                r2,
                points,
                args.dry_run,
                position,
            ): path
            for path, point, position in write_inputs
        }
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc="Write files",
            unit="file",
            position=0,
            dynamic_ncols=True,
        ):
            path = future_to_path[future]
            try:
                rows, frame_count = future.result()
            except BlockingIOError:
                print(f"{path.name}: SKIPPED - file is open in another program (close it and rerun)", flush=True)
                continue
            except Exception as exc:
                print(f"{path.name}: FAILED - {exc}", flush=True)
                continue
            per_frame_rows.extend(rows)
            print(f"Processed {path.name} ({frame_count} frames)", flush=True)

    per_frame_csv = directory / "pointing_calibration_v2b_per_frame.csv"
    if per_frame_rows and not args.dry_run:
        with open(per_frame_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_frame_rows)
        print(f"Wrote {per_frame_csv}", flush=True)
    elif args.dry_run:
        print(f"Dry run: not writing {per_frame_csv}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply v4 raster-scan pointing calibration to measurement H5 files.

Reads the calibration result (delta_pitch, delta_pan, and the affine matrix M
with o = M @ u) from the Pointing_Calibration_v4 group of a *_calibration.h5
file processed by process_calibration_v4.py, then writes per-acquisition
v4-corrected pointing attributes AND corrected per-pixel geometry datasets
into every measurement H5 file in the target folder.

Per-frame geometry correction. The mapping from assumed (az pan-deg, alt)
offsets to true offsets at the calibration tilt T_cal is

    C_cal = -inv(M) @ diag(1, -1)

The mechanical camera/mount relation is tilt independent; only the
cos(tilt) projection between mount pan and sky azimuth changes, so at an
acquisition tilt T the correction becomes

    C(T) = diag(cos(T_cal) / cos(T), 1) @ C_cal

and the per-frame camera roll is the polar-decomposition rotation angle of
C(T) - it varies with tilt (larger near zenith). Only this rotation (not the
scale/shear of C) is applied to the saved grids, to stay consistent with the
existing Level 1 geometry conventions.

Per acquisition this script writes:

    Tilt Error v4 Pred [deg]    = delta_pitch (shift to subtract from view_zen)
    Tilt_v4 [deg]               = Geometry Tilt Used + delta_pitch
    Zenith_v4 [deg]             = 90 - Tilt_v4
    Pan_v4 [deg]                = Geometry Pan Used + delta_pan
    Azimuth_v4 [deg]            = compass azimuth of Pan_v4
    Camera Roll v4 [deg]        = per-frame rotation of C(T)
    Geometry Correction Matrix v4 = C(T) (2x2)

and the corrected per-pixel datasets (unless --attrs-only):

    UV Image Data/view_az_v4    = rotate(view_az, view_zen) by roll about the
    UV Image Data/view_zen_v4     geometry anchor, then shift by delta_pan /
                                  delta_pitch

The attribute names mirror v2/v3 so robust_np_qt_app applies them the same
way (view_zen - tilt_err, view_az + (Pan_v4 - pan_raw), grid + Q/U rotation
by the per-frame camera roll). A Pointing_Calibration_v4 group with the
constants, fit statistics, and the source calibration file is also written
into each file.

Usage:
    python apply_calibration_v4.py FOLDER [--calibration CAL.h5] [--dry-run] [--attrs-only]
"""

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


G_SIGN = np.diag([1.0, -1.0])


V4_SUMMARY_KEYS = (
    "script_version",
    "processed_utc",
    "fit_n_points_total",
    "fit_n_points_used",
    "delta_pan_deg",
    "delta_pitch_deg",
    "delta_roll_deg",
    "pan_offset_used_deg",
    "tilt_offset_used_deg",
    "suggested_pan_offset_deg",
    "suggested_tilt_offset_deg",
    "pan_image_direction",
    "scale_pan",
    "scale_tilt",
    "mean_sun_altitude_deg",
    "residual_rms_x_deg",
    "residual_rms_y_deg",
)


def is_real_h5(path):
    return path.is_file() and path.suffix.lower() in (".h5", ".hdf5") and not path.name.startswith("._")


def find_calibration_file(directory):
    candidates = sorted(directory.glob("*_calibration.h5"))
    usable = []
    for path in candidates:
        if path.name.startswith("._"):
            continue
        try:
            with h5py.File(path, "r") as handle:
                if "Pointing_Calibration_v4" in handle:
                    usable.append(path)
        except OSError:
            continue
    if not usable:
        raise SystemExit(
            f"No *_calibration.h5 with a Pointing_Calibration_v4 group found in {directory}; "
            "run process_calibration_v4.py first or pass --calibration"
        )
    if len(usable) > 1:
        print(f"WARNING: multiple processed calibration files; using {usable[-1].name}", flush=True)
    return usable[-1]


def read_calibration(path):
    with h5py.File(path, "r") as handle:
        if "Pointing_Calibration_v4" not in handle:
            raise SystemExit(f"{path} has no Pointing_Calibration_v4 group; run process_calibration_v4.py first")
        attrs = handle["Pointing_Calibration_v4"].attrs
        constants = {key: attrs[key] for key in V4_SUMMARY_KEYS if key in attrs}
        m_cal = np.array(attrs["affine_matrix"], dtype=float)
    constants["source_calibration_file"] = path.name
    # The calibration raster is centered on the sun, so the calibration tilt is
    # the mean sun altitude during the scan.
    tilt_cal_deg = float(constants["mean_sun_altitude_deg"])
    return constants, m_cal, tilt_cal_deg


def geometry_correction_matrix(m_cal, tilt_cal_deg, tilt_deg):
    """C(T): true (az pan-deg, alt) offsets = C(T) @ assumed offsets at tilt T."""
    c_cal = -np.linalg.inv(m_cal) @ G_SIGN
    cos_ratio = np.cos(np.radians(tilt_cal_deg)) / max(np.cos(np.radians(tilt_deg)), 1e-3)
    return np.diag([cos_ratio, 1.0]) @ c_cal


def polar_rotation_deg(matrix):
    u_svd, _, vt_svd = np.linalg.svd(matrix)
    rotation = u_svd @ vt_svd
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def overwrite_dataset(group, name, data, **kwargs):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def frame_group_names(handle):
    return sorted(
        name for name in handle.keys()
        if name.startswith("Aquistion_") or name.startswith("calibration_acqui_")
    )


def corrected_geometry(uv_group, pan_raw, tilt_raw, roll_deg, delta_pan, delta_pitch):
    view_az = uv_group["view_az"][:].astype(np.float32)
    view_zen = uv_group["view_zen"][:].astype(np.float32)
    az_c = np.float32(pan_raw)
    zen_c = np.float32(90.0 - tilt_raw)
    theta = np.radians(roll_deg)
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    az_off = view_az - az_c
    up_off = zen_c - view_zen
    view_az_v4 = az_c + cos_t * az_off - sin_t * up_off + np.float32(delta_pan)
    view_zen_v4 = zen_c - (sin_t * az_off + cos_t * up_off) - np.float32(delta_pitch)
    return view_az_v4, view_zen_v4


def apply_to_file(path, constants, m_cal, tilt_cal_deg, dry_run=False, attrs_only=False):
    delta_pitch = float(constants["delta_pitch_deg"])
    delta_pan = float(constants["delta_pan_deg"])
    mode = "r" if dry_run else "r+"
    written = 0
    skipped = 0
    rolls = []
    with h5py.File(path, mode) as handle:
        names = frame_group_names(handle)
        for name in names:
            grp = handle[name]
            attrs = grp.attrs
            pan_raw = attrs.get("Geometry Pan Used [deg]")
            tilt_raw = attrs.get("Geometry Tilt Used [deg]")
            if pan_raw is None or tilt_raw is None:
                skipped += 1
                continue
            pan_raw = float(pan_raw)
            tilt_raw = float(tilt_raw)
            tilt_v4 = tilt_raw + delta_pitch
            pan_v4 = pan_raw + delta_pan
            c_matrix = geometry_correction_matrix(m_cal, tilt_cal_deg, tilt_raw)
            roll_deg = polar_rotation_deg(c_matrix)
            rolls.append(roll_deg)
            if not dry_run:
                attrs["Tilt Error v4 Pred [deg]"] = delta_pitch
                attrs["Tilt_v4 [deg]"] = tilt_v4
                attrs["Zenith_v4 [deg]"] = 90.0 - tilt_v4
                attrs["Pan_v4 [deg]"] = pan_v4
                attrs["Azimuth_v4 [deg]"] = (pan_v4 + 180.0) % 360.0
                attrs["Camera Roll v4 [deg]"] = roll_deg
                attrs["Geometry Correction Matrix v4"] = c_matrix
                if not attrs_only and "UV Image Data" in grp and "view_az" in grp["UV Image Data"]:
                    uv = grp["UV Image Data"]
                    view_az_v4, view_zen_v4 = corrected_geometry(
                        uv, pan_raw, tilt_raw, roll_deg, delta_pan, delta_pitch)
                    overwrite_dataset(uv, "view_az_v4", view_az_v4, compression="gzip")
                    overwrite_dataset(uv, "view_zen_v4", view_zen_v4, compression="gzip")
            written += 1
        if not dry_run:
            if "Pointing_Calibration_v4" in handle:
                del handle["Pointing_Calibration_v4"]
            grp = handle.create_group("Pointing_Calibration_v4")
            grp.attrs["model"] = (
                "Per-frame corrections from a pitch/roll raster calibration scan "
                "(process_calibration_v4.py): tilt_v4 = tilt_raw + delta_pitch, "
                "pan_v4 = pan_raw + delta_pan; per-frame camera roll = polar rotation "
                "of C(T) = diag(cos(T_cal)/cos(T), 1) @ (-inv(M) @ diag(1, -1)). "
                "view_az_v4/view_zen_v4 = grid rotated by the per-frame roll about the "
                "geometry anchor, then shifted by delta_pan / delta_pitch. "
                "Apply on the fly as view_zen - Tilt Error v4 Pred, view_az + "
                "(Pan_v4 - pan_raw), grid and Q/U rotated by Camera Roll v4."
            )
            grp.attrs["applied_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            grp.attrs["calibration_tilt_deg"] = tilt_cal_deg
            grp.attrs["affine_matrix"] = m_cal
            for key, value in constants.items():
                grp.attrs[key] = value
    return written, skipped, rolls


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder of measurement H5 files")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Processed *_calibration.h5 to read constants from (default: search the folder)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not modify files")
    parser.add_argument(
        "--attrs-only",
        action="store_true",
        help="Write attributes only; skip the corrected view_az_v4/view_zen_v4 datasets",
    )
    parser.add_argument(
        "--only-file",
        type=Path,
        default=None,
        help="Write v4 attributes/datasets only to this H5 file.",
    )
    args = parser.parse_args()

    directory = args.folder.expanduser()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    cal_path = args.calibration if args.calibration else find_calibration_file(directory)
    constants, m_cal, tilt_cal_deg = read_calibration(cal_path)
    print(
        f"Calibration {cal_path.name}: delta_pitch={float(constants['delta_pitch_deg']):+.4f} deg, "
        f"delta_pan={float(constants['delta_pan_deg']):+.4f} deg, "
        f"roll at calibration tilt {tilt_cal_deg:.2f} = {float(constants['delta_roll_deg']):+.4f} deg "
        f"(n={int(constants.get('fit_n_points_used', 0))}, "
        f"rms=({float(constants.get('residual_rms_x_deg', np.nan)):.4f}, "
        f"{float(constants.get('residual_rms_y_deg', np.nan)):.4f}) deg)",
        flush=True,
    )

    targets = [
        p for p in sorted(directory.iterdir())
        if is_real_h5(p) and not p.name.endswith("_calibration.h5") and p.resolve() != cal_path.resolve()
    ]
    if args.only_file is not None:
        only = args.only_file.expanduser()
        if not only.is_absolute():
            only = directory / only
        only = only.resolve()
        targets = [p for p in targets if p.resolve() == only]
        if not targets:
            raise SystemExit(f"--only-file is not a measurement H5 in {directory}: {only}")
    if not targets:
        raise SystemExit(f"No measurement H5 files in {directory}")

    locked = []
    for path in targets:
        try:
            written, skipped, rolls = apply_to_file(
                path, constants, m_cal, tilt_cal_deg,
                dry_run=args.dry_run, attrs_only=args.attrs_only,
            )
        except BlockingIOError:
            locked.append(path)
            print(f"{path.name}: SKIPPED - file is open in another program (close it and rerun)", flush=True)
            continue
        suffix = " (dry run)" if args.dry_run else " (attrs only)" if args.attrs_only else ""
        roll_info = f", roll {min(rolls):+.3f}..{max(rolls):+.3f} deg" if rolls else ""
        print(f"{path.name}: v4 on {written} groups, {skipped} skipped{roll_info}{suffix}", flush=True)
    if locked:
        print(f"\n{len(locked)} file(s) were locked; rerun this command after closing them.", flush=True)


if __name__ == "__main__":
    main()

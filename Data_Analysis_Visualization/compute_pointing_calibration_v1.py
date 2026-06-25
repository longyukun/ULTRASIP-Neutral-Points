#!/usr/bin/env python3
"""
compute_pointing_calibration_v1.py
====================================
Per-acquisition sun-centroid pointing correction (v1).

Background
----------
After the fix to process_single_level0_1_2.py, Level 1 uses the IMAGE CENTRE
(IMG_X/2, IMG_Y/2) as the geometry anchor for every frame.  At the image-
centre pixel:

    view_zen  =  90 - Tilt_moog            (Moog zenith coordinate)
    view_az   =  Pan_moog                  (Moog azimuth coordinate, 0 = South)

This means any Moog pointing error enters Level 1 as a uniform shift across
the whole view_zen / view_az grid.  v1 measures that shift per-frame from the
detected sun centroid and writes attributes in the same naming convention used
by v2 / v3, so the GUI can apply them identically.

Formula derivation
------------------
With image-centre reference, Level 1 assigns to the sun pixel (cx_sun, cy_sun):

    view_zen_at_sun  =  90 - Tilt + (cy_sun - IMG_Y/2) * VFOV
    view_az_at_sun   =  Pan        + (cx_sun - IMG_X/2) * HFOV

True sky position of the sun (from ephemeris attributes):
    true_sun_zenith  =  90 - Sun_Position_Altitude
    true_sun_azimuth =  Sun_Position_Azimuth

Pointing error (= amount to subtract from view_zen / add to view_az):
    zen_shift  =  view_zen_at_sun - true_sun_zenith
               =  (Sun_Alt - Tilt) + (cy_sun - IMG_Y/2) * VFOV
                  ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  Moog tilt bias       pixel displacement from centre

    pan_err    =  view_az_at_sun - true_sun_azimuth
               =  (Pan - Sun_Az) + (cx_sun - IMG_X/2) * HFOV

The GUI reads:
    zen_shift  ← attrs["Tilt Error v1 Pred [deg]"]
    az_shift   ← attrs["Pan_v1 [deg]"] - attrs["Geometry Pan Used [deg]"]

and applies:
    view_zen -= zen_shift
    view_az  += az_shift  (= view_az += Pan_v1 - Pan_raw = view_az - pan_err)

So we store:
    Tilt Error v1 Pred [deg]  =  zen_shift
    Pan_v1 [deg]              =  Pan_raw - pan_err  =  Sun_Az - (cx_sun - w/2)*HFOV
    Zenith_v1 [deg]           =  90 - (Tilt + zen_shift)
    Camera Roll v1 [deg]      =  0   (v1 does not estimate roll)

Sign convention
---------------
Positive zen_shift means the instrument pointed TOO HIGH (view_zen too large);
subtract zen_shift from view_zen to lower it toward the true sun zenith.

Requirements
------------
Level 0 must have been run first (I_corrected is needed).

Usage
-----
    python compute_pointing_calibration_v1.py /path/to/file.h5
    python compute_pointing_calibration_v1.py /path/to/file.h5 --dry-run
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ── camera constants (must match process_single_level0_1_2.py) ───────────────
IMG_X = 2848
IMG_Y = 2848
HFOV = 0.0020   # degrees per pixel (horizontal)
VFOV = 0.0020   # degrees per pixel (vertical)

# Sun centroid detection
SUN_THRESHOLD_FRAC = 0.95   # fraction of max(I_corrected)
SUN_MARGIN_PX = 220          # min pixel distance from image edge for valid sun


# ── helpers ───────────────────────────────────────────────────────────────────

def find_sun_center(intensity):
    """Centroid of pixels >= SUN_THRESHOLD_FRAC * max(intensity).

    Returns (cx, cy) in pixel coordinates, or None if not found.
    """
    arr = np.asarray(intensity, dtype=np.float64)
    threshold = SUN_THRESHOLD_FRAC * np.nanmax(arr)
    mask = arr >= threshold
    if not np.any(mask):
        return None
    rows, cols = np.nonzero(mask)
    return float(np.mean(cols)), float(np.mean(rows))   # (cx, cy)


def sun_is_complete(cx, cy, w, h, margin=SUN_MARGIN_PX):
    """Return True if sun centroid is at least `margin` pixels from every edge."""
    return margin <= cx <= w - margin and margin <= cy <= h - margin


def finite_attr(attrs, key, default=np.nan):
    try:
        v = float(attrs.get(key, default))
    except Exception:
        v = np.nan
    return v if np.isfinite(v) else np.nan


def select_tilt(attrs):
    for key in (
        "Moog Post Capture Actual Tilt [deg]",
        "Moog Actual Tilt [deg]",
        "Tilt",
    ):
        v = finite_attr(attrs, key)
        if np.isfinite(v):
            return v, key
    raise KeyError("No usable tilt attribute (tried Moog Post Capture / Actual / Tilt)")


def select_pan(attrs):
    for key in (
        "Moog Post Capture Actual Pan [deg]",
        "Moog Actual Pan [deg]",
        "Pan",
    ):
        v = finite_attr(attrs, key)
        if np.isfinite(v):
            return v, key
    raise KeyError("No usable pan attribute (tried Moog Post Capture / Actual / Pan)")


def frame_names(handle):
    """Return all Aquistion_* and calibration_acqui_* group names, sorted."""
    names = [
        n for n in handle.keys()
        if (n.startswith("Aquistion_") or n.startswith("calibration_acqui_"))
        and "UV Image Data" in handle[n]
    ]

    def sort_key(n):
        if n.startswith("Aquistion_"):
            return (0, int(n.split("_")[-1]))
        parts = n.split("_")
        direction = parts[2] if len(parts) > 2 else ""
        amount_str = parts[3].replace("p", ".") if len(parts) > 3 else "0"
        try:
            amount_val = float(amount_str)
        except ValueError:
            amount_val = 0.0
        return (1, 0 if direction == "down" else 1, amount_val)

    return sorted(names, key=sort_key)


# ── per-frame correction ──────────────────────────────────────────────────────

def compute_v1_attrs(grp):
    """Compute v1 pointing correction attributes for one frame group.

    Returns a dict of attributes to write, or raises ValueError with a reason.
    """
    uv = grp.get("UV Image Data")
    if uv is None:
        raise ValueError("no UV Image Data")
    if "I_corrected" not in uv:
        raise ValueError("no I_corrected — run Level 0 first")

    attrs = grp.attrs
    tilt, tilt_src = select_tilt(attrs)
    pan_raw, pan_src = select_pan(attrs)

    sun_alt = finite_attr(attrs, "Sun Position Altitude")
    sun_az = finite_attr(attrs, "Sun Position Azimuth")
    if not np.isfinite(sun_alt):
        raise ValueError("missing 'Sun Position Altitude' attribute")
    if not np.isfinite(sun_az):
        raise ValueError("missing 'Sun Position Azimuth' attribute")

    intensity = uv["I_corrected"][:]
    h, w = intensity.shape

    result = find_sun_center(intensity)
    if result is None:
        raise ValueError("sun not detected in I_corrected")
    cx, cy = result
    if not sun_is_complete(cx, cy, w, h):
        raise ValueError(
            f"sun centroid ({cx:.0f},{cy:.0f}) too close to image edge "
            f"(margin={SUN_MARGIN_PX}px)"
        )

    # ── pointing error formulae ───────────────────────────────────────────────
    #
    # zen_shift:
    #   Level 1 assigned to sun pixel: view_zen_at_sun = 90-Tilt+(cy-h/2)*VFOV
    #   True sun zenith: 90 - sun_alt
    #   Error (amount to subtract from view_zen):
    #     zen_shift = (90-Tilt+(cy-h/2)*VFOV) - (90-sun_alt)
    #               = (sun_alt - Tilt) + (cy - h/2) * VFOV
    #
    # pan_err  (amount to subtract from view_az):
    #   Level 1 assigned: view_az_at_sun = Pan + (cx - w/2)*HFOV
    #   True sun azimuth: sun_az
    #   Error: (Pan - sun_az) + (cx - w/2)*HFOV
    #
    # Pan_v1 = Pan_raw - pan_err  (so az_shift = Pan_v1 - Pan_raw = -pan_err)
    # GUI applies: view_az += (Pan_v1 - Pan_raw) = view_az - pan_err  ✓
    zen_shift = (sun_alt - tilt) + (cy - h / 2.0) * VFOV
    pan_err   = (pan_raw - sun_az) + (cx - w / 2.0) * HFOV
    pan_v1    = pan_raw - pan_err          # = sun_az - (cx - w/2)*HFOV
    zenith_v1 = 90.0 - (tilt + zen_shift)  # corrected zenith at image centre

    return {
        "Tilt Error v1 Pred [deg]": float(zen_shift),
        "Zenith_v1 [deg]":          float(zenith_v1),
        "Az_v1 [deg]":              float(pan_v1),    # corrected az at image centre
        "Pan_v1 [deg]":             float(pan_v1),
        "Camera Roll v1 [deg]":     0.0,
        "Sun Center v1 cx [px]":    float(cx),
        "Sun Center v1 cy [px]":    float(cy),
        "_tilt_src":  tilt_src,
        "_pan_src":   pan_src,
        "_pan_err":   float(pan_err),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def process_file(h5_path, dry_run=False):
    path = Path(h5_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    print(f"{'[DRY RUN] ' if dry_run else ''}Processing {path.name}", flush=True)
    n_ok = n_fail = 0

    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as handle:
        names = frame_names(handle)
        if not names:
            raise SystemExit("No Aquistion_* or calibration_acqui_* groups found")

        for name in names:
            grp = handle[name]
            try:
                result = compute_v1_attrs(grp)
            except (KeyError, ValueError) as exc:
                print(f"  SKIP {name}: {exc}", flush=True)
                n_fail += 1
                continue

            zen_shift = result["Tilt Error v1 Pred [deg]"]
            pan_v1    = result["Pan_v1 [deg]"]
            pan_err   = result["_pan_err"]
            cx, cy    = result["Sun Center v1 cx [px]"], result["Sun Center v1 cy [px]"]

            print(
                f"  {name}: zen_shift={zen_shift:+.4f}° "
                f"pan_err={pan_err:+.4f}° "
                f"pan_v1={pan_v1:.4f}° "
                f"sun=({cx:.0f},{cy:.0f})",
                flush=True,
            )

            if not dry_run:
                write_keys = {k: v for k, v in result.items() if not k.startswith("_")}
                for attr_name, attr_val in write_keys.items():
                    grp.attrs[attr_name] = attr_val

            n_ok += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done: {n_ok} OK, {n_fail} skipped", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write per-frame v1 (sun-centroid) pointing correction attributes "
            "into an H5 file.  Level 0 (I_corrected) must exist.  "
            "Level 1 must have been processed with the image-centre reference "
            "(process_single_level0_1_2.py ≥ 2026-06 version)."
        )
    )
    parser.add_argument("h5_path", help="Path to the H5 file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print corrections but do not write to the H5 file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_file(args.h5_path, dry_run=args.dry_run)

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_H5_DIR = Path("/Volumes/LaCie/Level2 data/2026_06_05")
DEFAULT_GRASP_CSV = Path(
    "/Users/dustin/Library/CloudStorage/OneDrive-UniversityofArizona/GRASP/Tucson_20260605.grasp.csv")
DEFAULT_OUT_DIR = SCRIPT_DIR / "batch_outputs"
DEFAULT_NUC = ROOT / "Data_Analysis_Visualization" / "NUC_0813.npz"
DEFAULT_WMATRIX = ROOT / "Data_Analysis_Visualization" / "ULTRASIP_AvgWmatrix_15.npy"

H5_FILE_COLUMNS = ("ultrasip.h5_file", "h5_file", "file", "filename", "h5")
GRASP_NP_COLUMNS = ("grasp_np_za_355nm", "grasp_np_za", "grasp_np_zen_deg")
GRASP_SZA_COLUMNS = ("grasp_sza_deg", "sun_zenith_deg", "sza_deg", "sza")
CORRECTION_COLUMNS = (
    "calib_tilt_correction_deg",
    "correction_deg",
    "tilt_correction_deg",
    "ultrasip_correction_deg",
    "corr_deg",
    "corr",
)


def acquisition_names(handle):
    names = [name for name in handle.keys() if name.startswith("Aquistion_")]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def evenly_spaced(items, count):
    if count is None or count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count)
    indices = sorted({int(round(idx)) for idx in indices})
    while len(indices) < count:
        for idx in range(len(items)):
            if idx not in indices:
                indices.append(idx)
                break
    return [items[idx] for idx in sorted(indices[:count])]


def first_existing_column(frame, candidates, required=True):
    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")
    return None


def numeric_value(row, columns, default=np.nan):
    for column in columns:
        if column in row and pd.notna(row[column]):
            try:
                return float(row[column])
            except Exception:
                pass
    return default


def first_numeric_value(row, columns):
    for column in columns:
        if column in row:
            try:
                value = float(row[column])
            except Exception:
                continue
            if np.isfinite(value):
                return value, column
    return np.nan, ""


def robust_csv_candidates(h5_path):
    name = h5_path.stem + "_robust_np_methods.csv"
    return [
        h5_path.with_name(name),
        SCRIPT_DIR / "robust_np_outputs" / name,
    ]


def robust_csv_path(h5_path):
    for candidate in robust_csv_candidates(h5_path):
        if candidate.exists():
            return candidate
    return h5_path.with_name(h5_path.stem + "_robust_np_methods.csv")


def h5_has_level2_products(h5_path):
    try:
        with h5py.File(h5_path, "r") as handle:
            names = acquisition_names(handle)
            if not names:
                return False
            for aq_name in names:
                uv = handle[aq_name].get("UV Image Data")
                if uv is None:
                    return False
                has_stokes = all(name in uv for name in ["I_corrected", "Q_corrected", "U_corrected"])
                has_angles = (
                    ("view_zen" in uv and "view_az" in uv)
                    or ("view_zen_corrected" in uv and "view_az_corrected" in uv)
                )
                if not (has_stokes and has_angles):
                    return False
            return True
    except OSError:
        return False


def run_command(command, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, env=env)


def ensure_robust_csv(h5_path, args, env):
    csv_path = robust_csv_path(h5_path)
    if csv_path.exists() and not args.force_robust:
        return csv_path, "csv"

    if not h5_has_level2_products(h5_path):
        if args.no_process_level2:
            raise RuntimeError(f"{h5_path.name} has no Level2 products and --no-process-level2 was set")
        run_command([
            sys.executable,
            str(SCRIPT_DIR / "process_single_level0_1_2.py"),
            str(h5_path),
            str(args.nuc),
            str(args.wmatrix),
        ], env)
    else:
        print(f"Using existing Level2 products: {h5_path.name}", flush=True)

    run_command([
        sys.executable,
        str(SCRIPT_DIR / "robust_np_all_acquisitions.py"),
        str(h5_path),
    ], env)
    csv_path = robust_csv_path(h5_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Robust summary was not created for {h5_path}")
    return csv_path, "computed"


def read_robust_rows(csv_path):
    with open(csv_path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except Exception:
                pass
    return rows


def best_robust_row(rows):
    best_labeled = [row for row in rows if str(row.get("selection_label", "")).upper() == "BEST"]
    if best_labeled:
        return best_labeled[0]
    return max(rows, key=lambda row: row.get("total_confidence", 0), default={})


def fit_window(y, x, start, stop):
    xw = x[start:stop]
    yw = y[start:stop]
    finite = np.isfinite(xw) & np.isfinite(yw)
    if np.count_nonzero(finite) < 20:
        return None
    xw = xw[finite]
    yw = yw[finite]
    results = sm.WLS(yw, sm.add_constant(xw), weights=np.ones_like(yw)).fit()
    intercept = float(results.params[0])
    stderr_arcsec = float(results.bse[0] * 3600)
    r2 = float(results.rsquared)
    outside = max(float(np.nanmin(yw)) - intercept, 0, intercept - float(np.nanmax(yw)))
    zero_gap = 0.0
    if np.nanmin(xw) > 0:
        zero_gap = float(np.nanmin(xw))
    elif np.nanmax(xw) < 0:
        zero_gap = float(abs(np.nanmax(xw)))
    score = r2 - stderr_arcsec / 1000 - outside * 2 - zero_gap * 10
    return {
        "start": start,
        "stop": stop,
        "intercept": intercept,
        "stderr_arcsec": stderr_arcsec,
        "r2": r2,
        "slope": float(results.params[1]),
        "score": float(score),
    }


def best_crop(y, x):
    best = None
    edge_margin = 100
    widths = [200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200]
    for width in widths:
        if len(x) < width + 2 * edge_margin:
            continue
        step = 5 if width <= 400 else 10
        for start in range(edge_margin, len(x) - width - edge_margin + 1, step):
            candidate = fit_window(y, x, start, start + width)
            if candidate is None:
                continue
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def load_view_geometry(uv):
    if "view_zen" in uv and "view_az" in uv:
        return uv["view_zen"][:], uv["view_az"][:]
    return uv["view_zen_corrected"][:], uv["view_az_corrected"][:]


def rotate_qu(q, u, theta_deg):
    """Rotate Q/I, U/I in Stokes space by theta using R(-theta)."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    q_rot = c * q + s * u
    u_rot = -s * q + c * u
    return q_rot, u_rot


def apply_pointing_correction(view_zen, view_az, q, u, attrs, version):
    """Apply the per-frame vN pointing calibration (Pointing_Calibration_v2) to
    view_zen/view_az (constant shift) and q/u (Stokes rotation by camera roll)."""
    if version is None:
        return view_zen, view_az, q, u
    try:
        zen_shift = float(attrs.get(f"Tilt Error {version} Pred [deg]", np.nan))
        pan_vn = float(attrs.get(f"Pan_{version} [deg]", np.nan))
        pan_raw = float(attrs.get("Geometry Pan Used [deg]", np.nan))
        roll = float(attrs.get(f"Camera Roll {version} [deg]", np.nan))
    except Exception:
        return view_zen, view_az, q, u
    if np.isfinite(zen_shift):
        view_zen = view_zen - zen_shift
    if np.isfinite(pan_vn) and np.isfinite(pan_raw):
        view_az = view_az + (pan_vn - pan_raw)
    if np.isfinite(roll) and roll != 0.0:
        q, u = rotate_qu(q, u, roll)
    return view_zen, view_az, q, u


def average_linearfit_for_acquisition(h5_path, aq_name, version=None):
    if not aq_name:
        return {}
    if not h5_has_level2_products(h5_path):
        return {}
    with h5py.File(h5_path, "r") as handle:
        if aq_name not in handle:
            return {}
        aq = handle[aq_name]
        uv = aq["UV Image Data"]
        required = ["I_corrected", "Q_corrected", "U_corrected"]
        if any(name not in uv for name in required):
            return {}
        view_zen, view_az = load_view_geometry(uv)
        intensity = uv["I_corrected"][:]
        q = uv["Q_corrected"][:] / np.maximum(intensity, 1e-6)
        u = uv["U_corrected"][:] / np.maximum(intensity, 1e-6)
        view_zen, view_az, q, u = apply_pointing_correction(view_zen, view_az, q, u, aq.attrs, version)
        avgq = np.nanmean(q, axis=1)
        avgu = np.nanmean(u, axis=0)
        vza = view_zen[:, 0]
        vaz = view_az[0, :]
        q_best = best_crop(vza, avgq)
        u_best = best_crop(vaz, avgu)
        if q_best is None or u_best is None:
            return {}
        return {
            "avg_zen_deg": q_best["intercept"],
            "avg_az_deg": u_best["intercept"],
            "avg_q_score": q_best["score"],
            "avg_u_score": u_best["score"],
            "avg_combined_score": q_best["score"] + u_best["score"],
            "avg_q_r2": q_best["r2"],
            "avg_u_r2": u_best["r2"],
            "avg_q_crop": f"{q_best['start']}:{q_best['stop']}",
            "avg_u_crop": f"{u_best['start']}:{u_best['stop']}",
        }


def parse_h5_time(h5_name):
    match = re.search(r"(\d{8})_(\d{2})_(\d{2})_(\d{2})", h5_name)
    if not match:
        return pd.NaT
    date_part, hour, minute, second = match.groups()
    try:
        return datetime.strptime(f"{date_part}_{hour}_{minute}_{second}", "%Y%m%d_%H_%M_%S")
    except ValueError:
        return pd.NaT


def build_grasp_lookup(grasp):
    file_column = first_existing_column(grasp, H5_FILE_COLUMNS)
    by_name = {}
    by_stem = {}
    for _, row in grasp.iterrows():
        name = Path(str(row[file_column])).name
        by_name[name] = row
        by_stem[Path(name).stem] = row
    return by_name, by_stem


def matching_grasp_row(h5_path, by_name, by_stem):
    row = by_name.get(h5_path.name)
    if row is not None:
        return row
    return by_stem.get(h5_path.stem)


def h5_files_for_grasp(h5_dir, by_name, by_stem, recursive=False):
    pattern = "**/*.h5" if recursive else "*.h5"
    files = []
    for path in sorted(h5_dir.glob(pattern)):
        if path.name.startswith("._"):
            continue
        if matching_grasp_row(path, by_name, by_stem) is not None:
            files.append(path)
    return files


def delta_rows_for_method(base, method, np_zen, sun_zen, grasp_np_zen, grasp_sza, correction_deg, has_correction):
    ultrasip_delta = sun_zen - np_zen
    grasp_delta = grasp_sza - grasp_np_zen
    base[f"{method}_np_zen_deg"] = np_zen
    base[f"{method}_ultrasip_sun_minus_npza_deg"] = ultrasip_delta
    base[f"{method}_grasp_sun_minus_npza_deg"] = grasp_delta
    base[f"{method}_diff_deg"] = ultrasip_delta - grasp_delta
    if has_correction:
        np_zen_corr = np_zen - correction_deg
        ultrasip_delta_corr = sun_zen - np_zen_corr
        base[f"{method}_np_zen_corr_deg"] = np_zen_corr
        base[f"{method}_ultrasip_sun_minus_npza_corr_deg"] = ultrasip_delta_corr
        base[f"{method}_diff_corr_deg"] = ultrasip_delta_corr - grasp_delta


def plot_diff_time(rows, out_png):
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return
    frame["plot_time"] = pd.to_datetime(frame["time_local"], errors="coerce")
    if frame["plot_time"].isna().all():
        x = np.arange(len(frame))
        xlabel = "H5 file"
    else:
        x = frame["plot_time"]
        xlabel = "Time"

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axhline(0, color="black", linewidth=1)
    styles = {
        "dolp": ("o", "tab:blue", "DoLP low-region"),
        "zero": ("s", "tab:orange", "Q/U zero-line"),
        "avg": ("D", "tab:green", "Average Q/U linearfit"),
        "avg_v2": ("^", "tab:purple", "Average Q/U v2"),
        "avg_v3": ("v", "tab:red", "Average Q/U v3"),
    }
    has_any_correction = bool(frame.get("has_correction", pd.Series(dtype=bool)).any())
    for method, (marker, color, label) in styles.items():
        if f"{method}_diff_deg" not in frame.columns:
            continue
        ax.plot(
            x,
            frame[f"{method}_diff_deg"],
            marker=marker,
            color=color,
            linewidth=1.5,
            label=f"{label} raw",
        )
        if has_any_correction and f"{method}_diff_corr_deg" in frame.columns:
            ax.plot(
                x,
                frame[f"{method}_diff_corr_deg"],
                marker=marker,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"{label} corr",
            )
    if frame["plot_time"].isna().all():
        labels = [Path(name).stem for name in frame["h5_file"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        fig.autofmt_xdate(rotation=35)
    ax.set_ylabel("(ULTRASIP sun - NPZA) - (GRASP sun - NPZA) [deg]")
    ax.set_xlabel(xlabel)
    ax.set_title("Best-acquisition ULTRASIP/GRASP neutral point delta difference")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2 if has_any_correction else 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def finite_limits(*arrays):
    values = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = max((hi - lo) * 0.08, 0.5)
    return lo - pad, hi + pad


def plot_ultrasip_vs_grasp(rows, out_png):
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return
    has_any_correction = bool(frame.get("has_correction", pd.Series(dtype=bool)).any())
    if not has_any_correction:
        return

    styles = {
        "dolp": ("o", "tab:blue", "DoLP low-region"),
        "zero": ("s", "tab:orange", "Q/U zero-line"),
        "avg": ("D", "tab:green", "Average Q/U linearfit"),
        "avg_v2": ("^", "tab:purple", "Average Q/U v2"),
        "avg_v3": ("v", "tab:red", "Average Q/U v3"),
    }
    # avg_v2/avg_v3 already use vN-corrected geometry and have no separate
    # "_corr" variant, so plot the same values in both panels.
    no_corr_variant = {"avg_v2", "avg_v3"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, suffix, title in (
            (axes[0], "", "Before correction"),
            (axes[1], "_corr", "After correction")):
        grasp_values = []
        ultrasip_values = []
        for method, (marker, color, label) in styles.items():
            grasp_col = f"{method}_grasp_sun_minus_npza_deg"
            col_suffix = "" if method in no_corr_variant else suffix
            ultra_col = f"{method}_ultrasip_sun_minus_npza{col_suffix}_deg"
            if grasp_col not in frame.columns or ultra_col not in frame.columns:
                continue
            ax.scatter(
                frame[grasp_col],
                frame[ultra_col],
                marker=marker,
                color=color,
                s=42,
                alpha=0.85,
                label=label,
            )
            grasp_values.append(frame[grasp_col])
            ultrasip_values.append(frame[ultra_col])
        lo, hi = finite_limits(*(grasp_values + ultrasip_values))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle=":")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("GRASP sun - NPZA [deg]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("ULTRASIP sun - NPZA [deg]")
    axes[1].legend(loc="best")
    fig.suptitle("ULTRASIP vs GRASP neutral point delta")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare best-acquisition ULTRASIP neutral point results against GRASP. "
            "Existing robust CSV summaries are reused; Level2 H5 products are reused when present."
        ))
    parser.add_argument("h5_dir", nargs="?", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("grasp_csv", nargs="?", type=Path, default=DEFAULT_GRASP_CSV)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--recursive", action="store_true", help="Search h5_dir recursively for *.h5")
    parser.add_argument("--max-files", type=int, default=None, help="Evenly sample this many matched files")
    parser.add_argument("--force-robust", action="store_true", help="Recompute robust CSV even when it exists")
    parser.add_argument("--no-process-level2", action="store_true", help="Do not run Level0/1/2 if Level2 is missing")
    parser.add_argument("--correction-deg", type=float, default=None, help="Correction to subtract from ULTRASIP NP zenith")
    parser.add_argument("--nuc", type=Path, default=DEFAULT_NUC)
    parser.add_argument("--wmatrix", type=Path, default=DEFAULT_WMATRIX)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grasp = pd.read_csv(args.grasp_csv)
    first_existing_column(grasp, GRASP_NP_COLUMNS)
    first_existing_column(grasp, GRASP_SZA_COLUMNS)
    by_name, by_stem = build_grasp_lookup(grasp)
    h5_files = h5_files_for_grasp(args.h5_dir, by_name, by_stem, recursive=args.recursive)
    selected = evenly_spaced(h5_files, args.max_files)
    if not selected:
        raise SystemExit(f"No H5 files in {args.h5_dir} matched rows in {args.grasp_csv}")

    print("Selected files:")
    for path in selected:
        print(" ", path.name)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/ultrasip_matplotlib")
    env.setdefault("NP_SAVE_DIAGNOSTICS", "0")
    env.setdefault("NP_WORKERS", "4")
    env.setdefault("NP_MIN_SUN_ZEN_SEPARATION_DEG", "1.0")

    rows = []
    for h5_path in selected:
        summary_csv, source = ensure_robust_csv(h5_path, args, env)
        robust_rows = read_robust_rows(summary_csv)
        best = best_robust_row(robust_rows)
        if not best:
            print(f"Skipping {h5_path.name}: no robust rows", flush=True)
            continue

        aq_name = best.get("acquisition", "")
        avg_best = average_linearfit_for_acquisition(h5_path, aq_name)
        avg_v2_best = average_linearfit_for_acquisition(h5_path, aq_name, version="v2")
        avg_v3_best = average_linearfit_for_acquisition(h5_path, aq_name, version="v3")
        grasp_row = matching_grasp_row(h5_path, by_name, by_stem)
        grasp_np_zen = numeric_value(grasp_row, GRASP_NP_COLUMNS)
        grasp_sza = numeric_value(grasp_row, GRASP_SZA_COLUMNS)
        if args.correction_deg is not None:
            correction_deg = float(args.correction_deg)
            correction_source = "cli"
            has_correction = True
        else:
            correction_deg, correction_source = first_numeric_value(best, CORRECTION_COLUMNS)
            has_correction = bool(correction_source)
        sun_zen = float(best.get("sun_zen_for_filter_deg", np.nan))

        row = {
            "h5_file": h5_path.name,
            "time_local": parse_h5_time(h5_path.name),
            "robust_summary_csv": str(summary_csv),
            "robust_source": source,
            "best_acquisition": aq_name,
            "best_total_confidence": best.get("total_confidence", np.nan),
            "sun_zen_deg": sun_zen,
            "grasp_sza_deg": grasp_sza,
            "grasp_np_za_deg": grasp_np_zen,
            "correction_deg": correction_deg,
            "correction_source": correction_source,
            "has_correction": has_correction,
            "dolp_confidence": best.get("dolp_confidence", np.nan),
            "zero_confidence": best.get("zero_confidence", np.nan),
            "avg_combined_score": avg_best.get("avg_combined_score", np.nan),
        }

        delta_rows_for_method(
            row, "dolp", float(best.get("dolp_zen_deg", np.nan)),
            sun_zen, grasp_np_zen, grasp_sza, correction_deg, has_correction)
        delta_rows_for_method(
            row, "zero", float(best.get("zero_zen_deg", np.nan)),
            sun_zen, grasp_np_zen, grasp_sza, correction_deg, has_correction)
        delta_rows_for_method(
            row, "avg", float(avg_best.get("avg_zen_deg", np.nan)),
            sun_zen, grasp_np_zen, grasp_sza, correction_deg, has_correction)
        delta_rows_for_method(
            row, "avg_v2", float(avg_v2_best.get("avg_zen_deg", np.nan)),
            sun_zen, grasp_np_zen, grasp_sza, np.nan, False)
        delta_rows_for_method(
            row, "avg_v3", float(avg_v3_best.get("avg_zen_deg", np.nan)),
            sun_zen, grasp_np_zen, grasp_sza, np.nan, False)
        rows.append(row)

    prefix = args.output_prefix
    if prefix is None:
        prefix = f"np_vs_grasp_{args.grasp_csv.stem}"
    out_csv = args.out_dir / f"{prefix}_best_acquisition_diff.csv"
    out_png = args.out_dir / f"{prefix}_best_acquisition_diff.png"
    out_compare_png = args.out_dir / f"{prefix}_ultrasip_vs_grasp.png"
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    plot_diff_time(rows, out_png)
    plot_ultrasip_vs_grasp(rows, out_compare_png)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    if out_compare_png.exists():
        print(f"Wrote {out_compare_png}")
    if rows:
        columns = [
            "h5_file", "best_acquisition", "correction_deg", "correction_source",
            "dolp_diff_deg", "zero_diff_deg", "avg_diff_deg",
            "avg_v2_diff_deg", "avg_v3_diff_deg",
        ]
        if frame["has_correction"].any():
            columns.extend([
                "dolp_diff_corr_deg",
                "zero_diff_corr_deg",
                "avg_diff_corr_deg",
            ])
        print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()

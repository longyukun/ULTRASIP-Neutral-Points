#!/usr/bin/env python3
"""
compare_flip_noflip.py
======================
Strip existing NP-estimation groups from an H5 file, then estimate the
neutral-point zenith angle for selected acquisitions using three methods:

  1. FLIP   – replicates the np.flip bug in Process_Level2.py
  2. NOFLIP – corrected version (no flip)
  3. AUTO   – automated best-crop pipeline (robust_np_qt_app.py logic)

Both FLIP and NOFLIP use the *same* crop window (determined by NOFLIP)
so the only variable is the flip itself.

Output:
  • terminal summary table
  • <h5_stem>_flip_comparison.png  (same folder as the H5)

Usage:
    python compare_flip_noflip.py [path_to_h5] [aq_idx1 aq_idx2 ...]

Examples:
    python compare_flip_noflip.py file.h5              # uses default TARGET_AQS
    python compare_flip_noflip.py file.h5 0 1 10 20   # explicit indices
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# ── configuration ──────────────────────────────────────────────────────────────
H5_PATH   = "/Volumes/Crucial/NormRoof_20250717_19_06_39.h5"
TARGET_AQS = [0, 1, 10]
SIGMA      = 2.0   # Gaussian smoothing (pixels)
# ───────────────────────────────────────────────────────────────────────────────


# ── helpers ────────────────────────────────────────────────────────────────────

def _fit_window(coord, stokes, s, e):
    """Linear fit  coord = a + b*stokes  in index window [s:e].
    Returns dict with intercept, r2, stderr_arcsec, score – or None."""
    x = np.asarray(stokes[s:e], dtype=np.float64)
    y = np.asarray(coord[s:e],  dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return None
    x, y = x[ok], y[ok]
    A = np.column_stack([np.ones_like(x), x])
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return None
    fitted   = A @ beta
    resid    = y - fitted
    ss_res   = float((resid**2).sum())
    ss_tot   = float(((y - y.mean())**2).sum())
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof      = max(len(y) - 2, 1)
    try:
        cov    = ss_res / dof * np.linalg.inv(A.T @ A)
        stderr = float(np.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        stderr = np.inf

    intercept = float(beta[0])
    outside   = max(y.min() - intercept, 0.0, intercept - y.max())
    if   x.min() > 0:   zero_gap = float(x.min())
    elif x.max() < 0:   zero_gap = float(abs(x.max()))
    else:                zero_gap = 0.0

    score = r2 - stderr * 3600 / 1000.0 - outside * 2.0 - zero_gap * 10.0
    return dict(start=s, stop=e, intercept=intercept, slope=float(beta[1]),
                r2=r2, stderr_deg=stderr, stderr_arcsec=stderr*3600,
                score=score, outside=outside, n=len(y))


def best_crop(coord, stokes):
    """Search crop windows, return the fit with the highest score."""
    n    = len(stokes)
    best = None
    for width in [200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200]:
        if n < width + 200:
            continue
        step = 5 if width <= 400 else 10
        for s in range(100, n - width - 100, step):
            c = _fit_window(coord, stokes, s, s + width)
            if c and (best is None or c["score"] > best["score"]):
                best = c
    return best


def wls_in_window(coord, stokes, s, e):
    """Return (intercept_deg, stderr_deg, r2) for the given window."""
    c = _fit_window(coord, stokes, s, e)
    if c:
        return c["intercept"], c["stderr_deg"], c["r2"]
    return np.nan, np.nan, np.nan


# ── strip existing NP estimation groups ────────────────────────────────────────

def strip_np_groups(h5_path):
    NP_KEYS = ["Neutral Point Estimation", "Corrected Neutral Point Localization"]
    removed = []
    with h5py.File(h5_path, "r+") as f:
        for key in NP_KEYS:
            if key in f:
                del f[key]
                removed.append(f"/{key}")
        for aq_name in list(f.keys()):
            if not (aq_name.startswith("Aquistion_") or
                    aq_name.startswith("calibration_")):
                continue
            aq = f[aq_name]
            for key in NP_KEYS:
                if key in aq:
                    del aq[key]
                    removed.append(f"/{aq_name}/{key}")
    return removed


# ── per-acquisition analysis ───────────────────────────────────────────────────

def analyse_acquisition(f, aq_idx):
    aq_name = f"Aquistion_{aq_idx}"
    if aq_name not in f:
        return None, f"{aq_name} not found"

    aq = f[aq_name]
    uv = aq["UV Image Data"]

    for k in ["I_corrected", "Q_corrected", "U_corrected"]:
        if k not in uv:
            return None, f"missing {k}"

    if "view_zen" in uv:
        view_zen = uv["view_zen"][:]
        view_az  = uv["view_az"][:]
    elif "view_zen_corrected" in uv:
        view_zen = uv["view_zen_corrected"][:]
        view_az  = uv["view_az_corrected"][:]
    else:
        return None, "no geometry (view_zen / view_zen_corrected)"

    I = uv["I_corrected"][:].astype(np.float64)
    Q = uv["Q_corrected"][:].astype(np.float64)
    U = uv["U_corrected"][:].astype(np.float64)

    I_s = gaussian_filter(I, sigma=SIGMA)
    Q_s = gaussian_filter(Q, sigma=SIGMA)
    U_s = gaussian_filter(U, sigma=SIGMA)
    q   = Q_s / np.maximum(I_s, 1e-6)
    u   = U_s / np.maximum(I_s, 1e-6)

    sun_alt = float(aq.attrs.get("Sun Position Altitude", np.nan))
    sun_az  = float(aq.attrs.get("Sun Position Azimuth",  np.nan))

    # Coordinate vectors
    vza      = view_zen[:, 0]               # zenith per row (↑ row → ↑ zenith)
    vaz      = view_az[0, :]                # azimuth per col
    avg_vzen = np.nanmean(view_zen, axis=1) # averaged zenith (for AUTO)
    avg_vaz  = np.nanmean(view_az,  axis=0) # averaged azimuth (for AUTO)

    # Row/col averages of q, u
    avgq = np.nanmean(q, axis=1)   # shape (nrows,); avgq[k] ↔ row k ↔ vza[k]
    avgu = np.nanmean(u, axis=0)   # shape (ncols,); avgu[k] ↔ col k ↔ vaz[k]

    # Flipped version (the Process_Level2 bug):
    #   avgq_flip[k] = avgq[N-1-k]  ↔ row (N-1-k), but paired with vza[k] (row k)
    avgq_flip = np.flip(avgq)

    # ── best crop determined by NO-FLIP (correct alignment) ──
    crop_q = best_crop(vza, avgq)
    crop_u = best_crop(vaz, avgu)

    # ── best crop for AUTO (uses avg_vzen as coord) ──
    crop_q_auto = best_crop(avg_vzen, avgq)
    crop_u_auto = best_crop(avg_vaz,  avgu)

    # Method 1 – NO FLIP (correct)
    if crop_q:
        s, e = crop_q["start"], crop_q["stop"]
        zen_noflip, zen_noflip_err, r2_nf = wls_in_window(vza, avgq, s, e)
    else:
        zen_noflip = zen_noflip_err = r2_nf = np.nan

    if crop_u:
        s, e = crop_u["start"], crop_u["stop"]
        az_noflip, _, _ = wls_in_window(vaz, avgu, s, e)
    else:
        az_noflip = np.nan

    # Method 2 – FLIP (bug); SAME crop window as no-flip to isolate the effect
    if crop_q:
        s, e = crop_q["start"], crop_q["stop"]
        zen_flip, zen_flip_err, r2_fl = wls_in_window(vza, avgq_flip, s, e)
    else:
        zen_flip = zen_flip_err = r2_fl = np.nan

    # Method 3 – AUTO (best crop on avg_vzen)
    if crop_q_auto:
        zen_auto    = crop_q_auto["intercept"]
        zen_auto_err = crop_q_auto["stderr_deg"]
        r2_auto     = crop_q_auto["r2"]
    else:
        zen_auto = zen_auto_err = r2_auto = np.nan

    if crop_u_auto:
        az_auto = crop_u_auto["intercept"]
    else:
        az_auto = np.nan

    return dict(
        aq_idx      = aq_idx,
        sun_alt     = sun_alt,
        sun_zen     = 90.0 - sun_alt,
        sun_az      = sun_az,
        q           = q,           # 2-D Q/I image (smoothed)
        view_zen    = view_zen,    # 2-D zenith grid
        view_az     = view_az,     # 2-D azimuth grid
        vza         = vza,
        vaz         = vaz,
        avg_vzen    = avg_vzen,
        avg_vaz     = avg_vaz,
        avgq        = avgq,
        avgq_flip   = avgq_flip,
        avgu        = avgu,
        crop_q      = crop_q,
        crop_u      = crop_u,
        crop_q_auto = crop_q_auto,
        zen_noflip      = zen_noflip,
        zen_noflip_err  = zen_noflip_err,
        r2_noflip       = r2_nf,
        zen_flip        = zen_flip,
        zen_flip_err    = zen_flip_err,
        r2_flip         = r2_fl,
        zen_auto        = zen_auto,
        zen_auto_err    = zen_auto_err,
        r2_auto         = r2_auto,
        az_noflip       = az_noflip,
        az_auto         = az_auto,
    ), None


# ── plotting ───────────────────────────────────────────────────────────────────

def _symlog_norm_limits(data, pct=99.5):
    """Return (vmin, vmax) for a symlog colour scale capped at ±pct percentile."""
    flat = data[np.isfinite(data)]
    if flat.size == 0:
        return -0.1, 0.1
    v = float(np.percentile(np.abs(flat), pct))
    v = max(v, 1e-4)
    return -v, v


def plot_results(results, h5_path):
    n   = len(results)
    # 5 panels per row: [Q/I image | full profile | zoomed crop | U profile | bar]
    fig = plt.figure(figsize=(26, 5.8 * n), constrained_layout=True)
    outer_gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.06)

    C_NF   = "#2196F3"   # blue  – no flip
    C_FL   = "#F44336"   # red   – flip
    C_AUTO = "#4CAF50"   # green – auto

    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import FuncFormatter

    for row_i, res in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer_gs[row_i],
            wspace=0.42, width_ratios=[2.5, 2.2, 2.2, 1.6, 1.0])

        ax_img  = fig.add_subplot(inner[0])   # NEW: 2-D Q/I
        ax_full = fig.add_subplot(inner[1])
        ax_zoom = fig.add_subplot(inner[2])
        ax_u    = fig.add_subplot(inner[3])
        ax_bar  = fig.add_subplot(inner[4])

        # ── panel 0: 2-D Q/I image (zenith × azimuth) ────────────────────────
        q2d      = res["q"]
        vzen2d   = res["view_zen"]
        vaz2d    = res["view_az"]
        az_min   = float(np.nanmin(vaz2d))
        az_max   = float(np.nanmax(vaz2d))
        zen_min  = float(np.nanmin(vzen2d))
        zen_max  = float(np.nanmax(vzen2d))
        extent   = [az_min, az_max, zen_max, zen_min]   # imshow: origin=upper

        vmin, vmax = _symlog_norm_limits(q2d)
        norm = SymLogNorm(linthresh=max(vmax * 0.01, 1e-4),
                          linscale=1, vmin=vmin, vmax=vmax, base=10)

        im = ax_img.imshow(q2d, cmap="coolwarm", norm=norm,
                           extent=extent, aspect="auto", origin="upper")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Q/I")

        # contour Q/I = 0
        try:
            cx = np.linspace(az_min, az_max, q2d.shape[1])
            cy = np.linspace(zen_min, zen_max, q2d.shape[0])
            ax_img.contour(cx, cy, q2d, levels=[0],
                           colors="white", linewidths=1.2)
        except Exception:
            pass

        # mark NP zenith lines at azimuth mid-point
        az_mid = (az_min + az_max) / 2
        for zen, c, lbl in [
            (res["zen_noflip"], C_NF,  f"NoFlip {res['zen_noflip']:.3f}°"),
            (res["zen_flip"],   C_FL,  f"Flip   {res['zen_flip']:.3f}°"),
            (res["zen_auto"],   C_AUTO,f"Auto   {res['zen_auto']:.3f}°"),
        ]:
            if np.isfinite(zen):
                ax_img.axhline(zen, color=c, lw=1.4, linestyle="--",
                               alpha=0.9, label=lbl)

        # mark sun position
        sun_zen = res["sun_zen"]
        sun_az  = res["sun_az"]
        if np.isfinite(sun_zen) and np.isfinite(sun_az):
            ax_img.scatter(sun_az, sun_zen, marker="*", s=140,
                           c="yellow", edgecolors="k", linewidths=0.6,
                           zorder=5, label=f"Sun {sun_zen:.2f}°")

        ax_img.set_xlabel("Azimuth [deg]", fontsize=8)
        ax_img.set_ylabel("Zenith [deg]", fontsize=8)
        ax_img.tick_params(labelsize=7)
        ax_img.set_title(
            f"Aq {res['aq_idx']}  Q/I\nSun zen={res['sun_zen']:.2f}° az={res['sun_az']:.2f}°",
            fontsize=9)
        ax_img.legend(fontsize=6.5, loc="lower right",
                      framealpha=0.75, handlelength=1.2)

        vza  = res["vza"]
        vaz  = res["vaz"]
        avgq = res["avgq"]
        aflip = res["avgq_flip"]
        avgu = res["avgu"]
        crop = res["crop_q"]

        tag  = f"Aq {res['aq_idx']}  |  Sun zen={res['sun_zen']:.2f}°, az={res['sun_az']:.2f}°"

        # ── panel A: full Q profile ──────────────────────────────────────────
        ax_full.plot(avgq,  vza, color=C_NF,   lw=1.5, label="no flip")
        ax_full.plot(aflip, vza, color=C_FL,   lw=1.5, linestyle="--",
                     alpha=0.85, label="flip")
        ax_full.axvline(0, color="k", lw=0.7)
        for zen, c in [(res["zen_noflip"], C_NF),
                       (res["zen_flip"],   C_FL),
                       (res["zen_auto"],   C_AUTO)]:
            if np.isfinite(zen):
                ax_full.axhline(zen, color=c, lw=1.3, linestyle=":",
                                alpha=0.9)
        if crop:
            s, e = crop["start"], crop["stop"]
            ax_full.axhspan(vza[s], vza[e-1],
                            alpha=0.07, color="gray", label="crop window")
        ax_full.invert_yaxis()
        ax_full.set_xscale("symlog", linthresh=0.005)
        ax_full.set_xlabel("avg Q/I")
        ax_full.set_ylabel("Zenith [deg]")
        ax_full.set_title(f"{tag}\nFull profile", fontsize=9)
        ax_full.legend(fontsize=8, loc="lower right")
        ax_full.grid(True, alpha=0.25)

        # ── panel B: zoomed crop region ──────────────────────────────────────
        if crop:
            s, e = crop["start"], crop["stop"]
            zs, ze = vza[s], vza[e-1]

            ax_zoom.plot(avgq[s:e],  vza[s:e], color=C_NF, lw=2.0,
                         label=f"no flip  →  NP = {res['zen_noflip']:.3f}°\n"
                               f"  R²={res['r2_noflip']:.4f}  "
                               f"err={res['zen_noflip_err']*3600:.0f}\"")
            ax_zoom.plot(aflip[s:e], vza[s:e], color=C_FL, lw=2.0,
                         linestyle="--",
                         label=f"flip     →  NP = {res['zen_flip']:.3f}°\n"
                               f"  R²={res['r2_flip']:.4f}  "
                               f"err={res['zen_flip_err']*3600:.0f}\"")

            for zen, c, lbl in [
                (res["zen_noflip"], C_NF,  None),
                (res["zen_flip"],   C_FL,  None),
                (res["zen_auto"],   C_AUTO,
                 f"auto  →  NP = {res['zen_auto']:.3f}°"),
            ]:
                if np.isfinite(zen):
                    ax_zoom.axhline(zen, color=c, lw=1.6, linestyle=":",
                                    alpha=0.95,
                                    label=lbl if lbl else "_nolegend_")

            ax_zoom.axvline(0, color="k", lw=0.7)
            diff = res["zen_flip"] - res["zen_noflip"]
            ax_zoom.set_title(
                f"Crop [{s}:{e}]  ({e-s} rows)\n"
                f"Flip − NoFlip = {diff:+.3f}°  ({diff*3600:+.0f}\")",
                fontsize=9)
        else:
            ax_zoom.text(0.5, 0.5, "No crop found",
                         ha="center", va="center", transform=ax_zoom.transAxes)
            ax_zoom.set_title("Crop region", fontsize=9)

        ax_zoom.invert_yaxis()
        ax_zoom.set_xscale("symlog", linthresh=0.005)
        ax_zoom.set_xlabel("avg Q/I")
        ax_zoom.set_ylabel("Zenith [deg]")
        ax_zoom.legend(fontsize=7.5, loc="lower right")
        ax_zoom.grid(True, alpha=0.25)

        # ── panel C: U profile / azimuth ─────────────────────────────────────
        ax_u.plot(avgu, vaz, color="mediumpurple", lw=1.5)
        ax_u.axvline(0, color="k", lw=0.7)
        for az, c, lbl in [
            (res["az_noflip"], C_NF,   f"no flip  az={res['az_noflip']:.3f}°"),
            (res["az_auto"],   C_AUTO, f"auto     az={res['az_auto']:.3f}°"),
        ]:
            if np.isfinite(az):
                ax_u.axhline(az, color=c, lw=1.6, linestyle=":",
                             alpha=0.95, label=lbl)
        ax_u.set_xscale("symlog", linthresh=0.005)
        ax_u.set_xlabel("avg U/I")
        ax_u.set_ylabel("Azimuth [deg]")
        ax_u.set_title("avg U/I vs azimuth", fontsize=9)
        ax_u.legend(fontsize=7.5)
        ax_u.grid(True, alpha=0.25)

        # ── panel D: bar chart ───────────────────────────────────────────────
        diffs  = [res["zen_flip"] - res["zen_noflip"],
                  res["zen_auto"] - res["zen_noflip"]]
        labels = ["Flip\n− NoFlip", "Auto\n− NoFlip"]
        bcolors = [C_FL, C_AUTO]
        bars = ax_bar.bar(labels, diffs, color=bcolors, width=0.5,
                          edgecolor="k", linewidth=0.8)
        ax_bar.axhline(0, color="k", lw=1)
        for bar, val in zip(bars, diffs):
            if not np.isfinite(val):
                continue
            yoff = 0.05 if val >= 0 else -0.05
            va   = "bottom" if val >= 0 else "top"
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        val + yoff,
                        f"{val:+.3f}°\n({val*3600:+.0f}\")",
                        ha="center", va=va, fontsize=10, fontweight="bold")
        ymax = max((abs(d) for d in diffs if np.isfinite(d)), default=1.0) + 0.6
        ax_bar.set_ylim(-ymax, ymax)
        ax_bar.set_ylabel("NP zenith diff [deg]")
        ax_bar.set_title(f"Aq {res['aq_idx']}\nnoflip = ref", fontsize=9)
        ax_bar.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax_bar.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"NP zenith: flip vs no-flip vs automated\n{os.path.basename(h5_path)}",
        fontsize=13, fontweight="bold")

    out = os.path.splitext(h5_path)[0] + "_flip_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # Parse arguments: first positional = h5 path, remaining = acquisition indices
    args = sys.argv[1:]
    if args and not args[0].lstrip("-").isdigit():
        h5_path = args.pop(0)
    else:
        h5_path = H5_PATH

    if args:
        try:
            target_aqs = [int(a) for a in args]
        except ValueError:
            print(f"Could not parse acquisition indices from: {args}", file=sys.stderr)
            sys.exit(1)
    else:
        target_aqs = TARGET_AQS

    print(f"File : {h5_path}")
    print(f"AQs  : {target_aqs}")
    print(f"sigma: {SIGMA} px\n")

    # 1. Strip existing NP estimation groups
    removed = strip_np_groups(h5_path)
    if removed:
        print("Stripped:", *removed, sep="\n  ")
    else:
        print("No NP groups to strip.")
    print()

    # 2. Analyse acquisitions
    results = []
    with h5py.File(h5_path, "r") as f:
        for idx in target_aqs:
            res, err = analyse_acquisition(f, idx)
            if res is None:
                print(f"[SKIP] Aquistion_{idx}: {err}")
                continue
            results.append(res)

    if not results:
        print("No acquisitions could be processed. Check that Level 0/1 "
              "products (I_corrected, view_zen, …) exist in the file.")
        sys.exit(1)

    # 3. Print summary table
    hdr = (f"{'Aq':>3}  {'SunZen':>7}  "
           f"{'NoFlip':>9}  {'Flip':>9}  {'Auto':>9}  "
           f"{'Flip-NF':>9}  {'Auto-NF':>9}")
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        diff_fl = res["zen_flip"]   - res["zen_noflip"]
        diff_au = res["zen_auto"]   - res["zen_noflip"]
        print(
            f"{res['aq_idx']:>3}  "
            f"{res['sun_zen']:>7.3f}  "
            f"{res['zen_noflip']:>9.4f}  "
            f"{res['zen_flip']:>9.4f}  "
            f"{res['zen_auto']:>9.4f}  "
            f"{diff_fl:>+9.4f}  "
            f"{diff_au:>+9.4f}"
        )
    print()
    print("Units: all zenith angles in degrees.  Flip-NF / Auto-NF = difference from NoFlip.")

    # Also print arcsec version
    print()
    print("Same differences in arcseconds:")
    for res in results:
        diff_fl = res["zen_flip"] - res["zen_noflip"]
        diff_au = res["zen_auto"] - res["zen_noflip"]
        print(f"  Aq {res['aq_idx']:>2}:  Flip−NoFlip = {diff_fl*3600:+7.1f}\"  "
              f"Auto−NoFlip = {diff_au*3600:+7.1f}\"")

    # 4. Plot
    plot_results(results, h5_path)
    plt.show()


if __name__ == "__main__":
    main()

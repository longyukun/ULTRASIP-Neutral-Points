import os
import sys
import time

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


def overwrite_dataset(group, name, data, **kwargs):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def pseudoinverse(matrix):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s_inv = np.where(s > 1e-12, 1 / s, 0)
    return np.matmul(
        vt.transpose(0, 1, 3, 2) * s_inv[..., None, :],
        u.transpose(0, 1, 3, 2))


def correct_img(pij, rij, bij):
    r_avg = np.mean(rij)
    b_avg = np.mean(bij)
    return (r_avg / rij) * (pij - bij) + b_avg


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


def process_level0(handle, nuc_path=None, wmatrix_path=None):
    use_calibration = nuc_path and wmatrix_path

    if use_calibration:
        print(f"Level 0 calibration NUC: {nuc_path}", flush=True)
        print(f"Level 0 calibration W: {wmatrix_path}", flush=True)
        nuc_files = np.load(nuc_path, allow_pickle=True)
        rij = nuc_files["arr1"].item()
        bij = nuc_files["arr2"].item()
        pinv_w = pseudoinverse(np.load(wmatrix_path))
    else:
        w_ideal = 0.5 * np.array([
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, 1],
            [1, 0, -1],
        ])
        pinv_w = np.linalg.pinv(w_ideal)

    for name in acquisition_names(handle):
        print(f"Level 0: {name}", flush=True)
        uv_data = handle[name]["UV Image Data"]
        uv_imgs = uv_data["UV Raw Images"][:].reshape(4, IMG_Y, IMG_X)
        p0, p45, p90, p135 = uv_imgs

        if use_calibration:
            c0 = correct_img(p0, rij[0], bij[0])
            c90 = correct_img(p90, rij[90], bij[90])
            c45 = correct_img(p45, rij[45], bij[45])
            c135 = correct_img(p135, rij[135], bij[135])
            p_stack = np.stack([c0, c90, c45, c135], axis=-1)
            stokes = np.matmul(pinv_w, p_stack[..., None])[..., 0]
        else:
            # Match the current Process_Level0 analyzer order: P0, P90, P45, P135.
            p_stack = np.stack([p0, p90, p45, p135], axis=0).reshape(4, -1)
            stokes = (pinv_w @ p_stack).reshape(3, IMG_Y, IMG_X)

        if use_calibration:
            overwrite_dataset(uv_data, "I_corrected", stokes[:, :, 0].astype(np.float32), compression="gzip")
            overwrite_dataset(uv_data, "Q_corrected", stokes[:, :, 1].astype(np.float32), compression="gzip")
            overwrite_dataset(uv_data, "U_corrected", stokes[:, :, 2].astype(np.float32), compression="gzip")
        else:
            overwrite_dataset(uv_data, "I_corrected", stokes[0].astype(np.float32), compression="gzip")
            overwrite_dataset(uv_data, "Q_corrected", stokes[1].astype(np.float32), compression="gzip")
            overwrite_dataset(uv_data, "U_corrected", stokes[2].astype(np.float32), compression="gzip")

    meta = handle["Measurement_Metadata"].attrs
    meta["Processed Level"] = "Level 0"
    if use_calibration:
        meta["Level 0 Calibration Note"] = (
            f"Processed with NUC file {nuc_path} and W matrix {wmatrix_path}.")
    else:
        meta["Level 0 Calibration Note"] = (
            "Processed with ideal analyzer W matrix only; NUC_0813.npz and "
            "ULTRASIP_AvgWmatrix_15.npy were not available in this workspace."
        )


def find_sun_center(intensity):
    threshold = 0.95 * np.nanmax(intensity)
    mask = intensity >= threshold
    if not np.any(mask):
        return IMG_X / 2, IMG_Y / 2

    rows, cols = np.nonzero(mask)
    return float(np.mean(cols)), float(np.mean(rows))


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


def process_level1(handle):
    names = acquisition_names(handle)
    sza0 = None
    x_center = None
    y_center = None

    for idx, name in enumerate(names):
        print(f"Level 1: {name}", flush=True)
        aq = handle[name]
        uv_data = aq["UV Image Data"]

        sun_az_attr = aq.attrs["Sun Position Azimuth"]
        sun_alt_attr = aq.attrs["Sun Position Altitude"]
        pan = aq.attrs.get("Moog Actual Pan [deg]", aq.attrs["Pan"])
        tilt, tilt_source = select_tilt(aq.attrs)
        aq.attrs["Geometry Pan Source"] = (
            "Moog Actual Pan [deg]" if "Moog Actual Pan [deg]" in aq.attrs else "Pan")
        aq.attrs["Geometry Tilt Source"] = tilt_source
        aq.attrs["Geometry Sun Altitude Drift Correction"] = "disabled"
        aq.attrs["Geometry Pan Used [deg]"] = float(pan)
        aq.attrs["Geometry Tilt Used [deg]"] = float(tilt)

        if idx == 0:
            intensity = uv_data["I_corrected"][:]
            x_center, y_center = find_sun_center(intensity)
            sza0 = sun_alt_attr
            handle["Measurement_Metadata"].attrs["Center Pixel (x,y)"] = np.array(
                [x_center, y_center])

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

    handle["Measurement_Metadata"].attrs["Processed Level"] = "Level 1"


def save_level2_figure(path, aq_name, timestamp, uv_data, out_dir):
    view_az = uv_data["view_az"][:]
    view_zen = uv_data["view_zen"][:]
    intensity = uv_data["I_corrected"][:]
    q = uv_data["Q_corrected"][:] / intensity
    u = uv_data["U_corrected"][:] / intensity
    dolp = np.sqrt(q ** 2 + u ** 2) * 100
    aolp = np.mod(np.degrees(0.5 * np.arctan2(uv_data["U_corrected"][:], uv_data["Q_corrected"][:])), 180)

    overwrite_dataset(uv_data, "q", q.astype(np.float32), compression="gzip")
    overwrite_dataset(uv_data, "u", u.astype(np.float32), compression="gzip")
    overwrite_dataset(uv_data, "DoLP", dolp.astype(np.float32), compression="gzip")
    overwrite_dataset(uv_data, "AoLP", aolp.astype(np.float32), compression="gzip")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    extent = [view_az.min(), view_az.max(), view_zen.max(), view_zen.min()]

    images = [
        (axes[0, 0], intensity, "I", "gray", None, None),
        (axes[0, 1], q, "Q/I", "coolwarm", -0.2, 0.2),
        (axes[0, 2], u, "U/I", "coolwarm", -0.2, 0.2),
        (axes[1, 0], dolp, "DoLP [%]", "hot", 0, 8),
        (axes[1, 1], np.log(np.maximum(dolp, 1e-6)), "log(DoLP [%])", "Blues_r", -1, 2),
        (axes[1, 2], aolp, "AoLP [deg]", "twilight", 0, 180),
    ]

    for ax, data, title, cmap, vmin, vmax in images:
        im = ax.imshow(
            data,
            cmap=cmap,
            interpolation="None",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel("Zenith [deg]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{os.path.basename(path)} {aq_name} {timestamp}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{timestamp}_{aq_name}_Level2.png"), dpi=200)
    plt.close(fig)


def process_level2(handle, path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for name in acquisition_names(handle):
        print(f"Level 2: {name}", flush=True)
        aq = handle[name]
        timestamp = aq.attrs.get("Timestamp UTC", name)
        save_level2_figure(path, name, timestamp, aq["UV Image Data"], out_dir)

    handle["Measurement_Metadata"].attrs["Processed Level"] = "Level 2"


def main():
    if len(sys.argv) not in (2, 4):
        raise SystemExit(
            "Usage: python process_single_level0_1_2.py /path/to/file.h5 "
            "[/path/to/NUC_0813.npz /path/to/ULTRASIP_AvgWmatrix_15.npy]")

    path = sys.argv[1]
    nuc_path = sys.argv[2] if len(sys.argv) == 4 else None
    wmatrix_path = sys.argv[3] if len(sys.argv) == 4 else None
    out_dir = os.path.join(os.path.dirname(path), "level2_figures")
    start = time.time()

    with h5py.File(path, "r+") as handle:
        process_level0(handle, nuc_path, wmatrix_path)
        process_level1(handle)
        process_level2(handle, path, out_dir)

    print(f"Done in {time.time() - start:.1f} seconds")
    print(f"Level 2 figures: {out_dir}")


if __name__ == "__main__":
    main()

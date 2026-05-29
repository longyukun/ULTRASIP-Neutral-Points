import csv
import os
import queue
import tempfile
import threading
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-ultrasip"))

import h5py
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import circle_fit_sun_center as cfit


DEFAULT_LOG = Path(__file__).with_name("sun_center_diff_log.csv")
FOV_CENTER_Y = cfit.IMG_Y / 2.0


def sun_candidate_label(fit):
    return bool(fit and fit.get("sun_disk_candidate", False))


def get_tilt_pair(aq0, aq1):
    actual_key = "Moog Actual Tilt [deg]"
    if actual_key in aq0.attrs and actual_key in aq1.attrs:
        return float(aq0.attrs[actual_key]), float(aq1.attrs[actual_key]), actual_key
    requested_key = "Moog Requested Tilt [deg]"
    if requested_key in aq0.attrs and requested_key in aq1.attrs:
        return float(aq0.attrs[requested_key]), float(aq1.attrs[requested_key]), requested_key
    return float(aq0.attrs.get("Tilt", np.nan)), float(aq1.attrs.get("Tilt", np.nan)), "Tilt"


def get_moog_label_values(aq):
    has_requested = "Moog Requested Pan [deg]" in aq.attrs and "Moog Requested Tilt [deg]" in aq.attrs
    has_actual = "Moog Actual Pan [deg]" in aq.attrs and "Moog Actual Tilt [deg]" in aq.attrs
    return {
        "requested_pan": float(aq.attrs["Moog Requested Pan [deg]"] if has_requested else aq.attrs.get("Pan", np.nan)),
        "requested_tilt": float(aq.attrs["Moog Requested Tilt [deg]"] if has_requested else aq.attrs.get("Tilt", np.nan)),
        "actual_pan": float(aq.attrs.get("Moog Actual Pan [deg]", np.nan)),
        "actual_tilt": float(aq.attrs.get("Moog Actual Tilt [deg]", np.nan)),
        "has_requested": has_requested,
        "has_actual": has_actual,
    }


def center_zenith_from_sun_pixel(aq, fit, fov_center_y=FOV_CENTER_Y):
    sun_zenith = 90.0 - float(aq.attrs.get("Sun Position Altitude", np.nan))
    row_offset = (fit["y"] - fov_center_y) * cfit.VFOV
    return sun_zenith - row_offset


def analyze_h5(path, intensity_mode, aq0_radius_scale=1.0, aq1_radius_scale=1.0):
    cfit.INTENSITY_MODE = intensity_mode
    h5 = h5py.File(path, "r")
    try:
        if "Aquistion_0" not in h5 or "Aquistion_1" not in h5:
            raise ValueError("H5 must contain Aquisition_0 and Aquisition_1")

        aq0 = h5["Aquistion_0"]
        aq1 = h5["Aquistion_1"]
        fit0 = cfit.fit_sun_circle(aq0)
        fit1_free = cfit.fit_sun_circle(aq1)
        if fit0 is None or fit1_free is None:
            raise ValueError("Could not fit sun circle for both Aquisition_0 and Aquisition_1")
        if aq0_radius_scale != 1.0:
            scaled_fit0 = dict(fit0)
            scaled_fit0["radius"] = fit0["radius"] * aq0_radius_scale
            scaled_fit0["fit_method"] = f"{fit0.get('fit_method', '')}+manual_radius_scale"
            fit0 = scaled_fit0
        completed_radius = fit0["radius"] * aq1_radius_scale
        completed_fit1 = cfit.fit_completed_disk_centroid_from_arc(aq1, completed_radius, initial_fit=fit1_free)
        fit1 = completed_fit1 if completed_fit1 is not None else fit1_free

        aq0_tilt, aq1_tilt, tilt_source = get_tilt_pair(aq0, aq1)
        aq0_sun_alt = float(aq0.attrs.get("Sun Position Altitude", np.nan))
        aq1_sun_alt = float(aq1.attrs.get("Sun Position Altitude", np.nan))
        pixel_term = (fit1["y"] - fit0["y"]) * cfit.VFOV
        tilt_term = aq0_tilt - aq1_tilt
        sun_alt_term = aq1_sun_alt - aq0_sun_alt
        z0 = 90.0 - aq0_tilt
        z1 = z0 + pixel_term + tilt_term + sun_alt_term
        aq0_center_zen_from_sun = center_zenith_from_sun_pixel(aq0, fit0)
        aq1_center_zen_from_sun = center_zenith_from_sun_pixel(aq1, fit1)
        center_zen_diff_from_sun = aq1_center_zen_from_sun - aq0_center_zen_from_sun
        aq0_center_zen_from_moog = 90.0 - aq0_tilt
        aq1_center_zen_from_moog = 90.0 - aq1_tilt
        center_zen_diff_from_moog = aq1_center_zen_from_moog - aq0_center_zen_from_moog

        display0 = cfit.downsample_intensity(aq0)
        display1 = cfit.downsample_intensity(aq1)

        return {
            "path": path,
            "file": os.path.basename(path),
            "mode": intensity_mode,
            "aq0_radius_scale": aq0_radius_scale,
            "aq1_radius_scale": aq1_radius_scale,
            "fit0": fit0,
            "fit1": fit1,
            "fit1_free": fit1_free,
            "fit1_completed": completed_fit1,
            "display0": display0,
            "display1": display1,
            "aq0_moog": get_moog_label_values(aq0),
            "aq1_moog": get_moog_label_values(aq1),
            "zenith0": z0,
            "zenith1": z1,
            "diff": z1 - z0,
            "fov_center_y": FOV_CENTER_Y,
            "aq0_center_zen_from_sun": aq0_center_zen_from_sun,
            "aq1_center_zen_from_sun": aq1_center_zen_from_sun,
            "center_zen_diff_from_sun": center_zen_diff_from_sun,
            "aq0_center_zen_from_moog": aq0_center_zen_from_moog,
            "aq1_center_zen_from_moog": aq1_center_zen_from_moog,
            "center_zen_diff_from_moog": center_zen_diff_from_moog,
            "pixel_term": pixel_term,
            "tilt_term": tilt_term,
            "tilt_source": tilt_source,
            "sun_alt_term": sun_alt_term,
            "aq0_tilt": aq0_tilt,
            "aq1_tilt": aq1_tilt,
            "aq0_sun_alt": aq0_sun_alt,
            "aq1_sun_alt": aq1_sun_alt,
            "aq0_ok": sun_candidate_label(fit0),
            "aq1_ok": sun_candidate_label(fit1),
        }
    finally:
        cfit.close_h5_quietly(h5)


class SunDiffGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ULTRASIP Sun Center Diff")
        self.result = None
        self.worker_queue = queue.Queue()

        self.intensity_mode = tk.StringVar(value="p0")
        self.aq0_radius_scale = tk.DoubleVar(value=1.0)
        self.aq1_radius_scale = tk.DoubleVar(value=1.0)
        self.log_path = tk.StringVar(value=str(DEFAULT_LOG))
        self.status = tk.StringVar(value="Open or drop an H5 file.")

        self._build_ui()
        self.root.after(100, self._poll_worker)

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.open_button = ttk.Button(toolbar, text="Open H5", command=self.open_file)
        self.open_button.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Intensity").pack(side=tk.LEFT, padx=(12, 4))
        mode_menu = ttk.OptionMenu(toolbar, self.intensity_mode, self.intensity_mode.get(), "p0", "p0_p90")
        mode_menu.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Aq0 radius x").pack(side=tk.LEFT, padx=(12, 4))
        self.aq0_radius_scale_label = ttk.Label(toolbar, text="1.00", width=4)
        self.aq0_radius_scale_label.pack(side=tk.LEFT)
        aq0_radius_scale = ttk.Scale(
            toolbar,
            from_=0.60,
            to=1.80,
            orient=tk.HORIZONTAL,
            variable=self.aq0_radius_scale,
            command=self.on_aq0_radius_scale_change,
            length=120,
        )
        aq0_radius_scale.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(toolbar, text="Aq1 radius x").pack(side=tk.LEFT, padx=(12, 4))
        self.radius_scale_label = ttk.Label(toolbar, text="1.00", width=4)
        self.radius_scale_label.pack(side=tk.LEFT)
        radius_scale = ttk.Scale(
            toolbar,
            from_=0.60,
            to=1.80,
            orient=tk.HORIZONTAL,
            variable=self.aq1_radius_scale,
            command=self.on_radius_scale_change,
            length=160,
        )
        radius_scale.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Refit", command=self.refit_current).pack(side=tk.LEFT, padx=(8, 0))

        self.append_button = ttk.Button(toolbar, text="Append Diff To CSV", command=self.append_current, state=tk.DISABLED)
        self.append_button.pack(side=tk.LEFT, padx=(12, 0))

        ttk.Button(toolbar, text="Choose CSV", command=self.choose_log).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(toolbar, textvariable=self.log_path).pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)

        self.drop_frame = ttk.Frame(self.root, padding=8)
        self.drop_frame.pack(side=tk.TOP, fill=tk.X)
        self.drop_label = ttk.Label(
            self.drop_frame,
            text="Drop H5 here" if DND_AVAILABLE else "Drag/drop needs tkinterdnd2; use Open H5 on this machine.",
            anchor="center",
            relief=tk.GROOVE,
            padding=10,
        )
        self.drop_label.pack(fill=tk.X)
        if DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

        self.info = ttk.Label(self.root, textvariable=self.status, padding=(8, 0, 8, 8), justify=tk.LEFT)
        self.info.pack(side=tk.TOP, fill=tk.X)

        self.figure = Figure(figsize=(12, 5.8), dpi=100)
        self.ax0 = self.figure.add_subplot(1, 2, 1)
        self.ax1 = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._clear_axes()

    def _clear_axes(self):
        for ax, title in ((self.ax0, "Aquisition_0"), (self.ax1, "Aquisition_1")):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel(f"column / {cfit.DISPLAY_STEP}")
            ax.set_ylabel(f"row / {cfit.DISPLAY_STEP}")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*")])
        if path:
            self.load_file(path)

    def choose_log(self):
        path = filedialog.asksaveasfilename(
            initialfile=DEFAULT_LOG.name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if path:
            self.log_path.set(path)

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.load_file(paths[0])

    def load_file(self, path):
        self.result = None
        self.append_button.config(state=tk.DISABLED)
        self.open_button.config(state=tk.DISABLED)
        self.status.set(f"Loading {path} ...")
        self._clear_axes()

        thread = threading.Thread(
            target=self._worker,
            args=(path, self.intensity_mode.get(), self.aq0_radius_scale.get(), self.aq1_radius_scale.get()),
            daemon=True,
        )
        thread.start()

    def _worker(self, path, intensity_mode, aq0_radius_scale, aq1_radius_scale):
        try:
            self.worker_queue.put(("result", analyze_h5(path, intensity_mode, aq0_radius_scale, aq1_radius_scale)))
        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

    def on_aq0_radius_scale_change(self, value):
        self.aq0_radius_scale_label.config(text=f"{float(value):.2f}")

    def on_radius_scale_change(self, value):
        self.radius_scale_label.config(text=f"{float(value):.2f}")

    def refit_current(self):
        if self.result is None:
            return
        self.load_file(self.result["path"])

    def _poll_worker(self):
        try:
            kind, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker)
            return

        self.open_button.config(state=tk.NORMAL)
        if kind == "error":
            self.status.set(f"Error: {payload}")
            self.append_button.config(state=tk.DISABLED)
        else:
            self.result = payload
            self._plot_result(payload)
            self.append_button.config(state=tk.NORMAL)
        self.root.after(100, self._poll_worker)

    def _plot_result(self, result):
        self._plot_one(self.ax0, result["display0"], result["fit0"], "Aquisition_0", moog_values=result["aq0_moog"])
        self._plot_one(
            self.ax1,
            result["display1"],
            result["fit1"],
            "Aquisition_1",
            result["fit0"],
            moog_values=result["aq1_moog"],
        )
        self.figure.suptitle(result["file"])
        self.figure.tight_layout()
        self.canvas.draw_idle()

        f0 = result["fit0"]
        f1 = result["fit1"]
        self.status.set(
            f"{result['file']} | mode={result['mode']} | aq0 radius scale={result['aq0_radius_scale']:.2f} | "
            f"aq1 radius scale={result['aq1_radius_scale']:.2f} | "
            f"sun residual diff={result['diff']:.6f} deg\n"
            f"components: pixel-only={(result['pixel_term']):.6f} deg, "
            f"tilt={result['tilt_term']:.6f} deg ({result['tilt_source']}), "
            f"sun-alt={result['sun_alt_term']:.6f} deg\n"
            f"FOV center zen from sun: aq0={result['aq0_center_zen_from_sun']:.6f}, "
            f"aq1={result['aq1_center_zen_from_sun']:.6f}, "
            f"diff={result['center_zen_diff_from_sun']:.6f} deg\n"
            f"FOV center zen from Moog ({result['tilt_source']}): "
            f"aq0={result['aq0_center_zen_from_moog']:.6f}, "
            f"aq1={result['aq1_center_zen_from_moog']:.6f}, "
            f"diff={result['center_zen_diff_from_moog']:.6f} deg\n"
            f"aq0 pixel=({f0['x']:.1f}, {f0['y']:.1f}), zenith={result['zenith0']:.6f}, "
            f"r={f0['radius']:.1f}, rmse={f0['rmse']:.1f}, method={f0.get('fit_method', '')}, "
            f"sun_candidate={result['aq0_ok']}\n"
            f"aq1 pixel=({f1['x']:.1f}, {f1['y']:.1f}), zenith={result['zenith1']:.6f}, "
            f"r={f1['radius']:.1f}, rmse={f1['rmse']:.1f}, method={f1.get('fit_method', '')}, "
            f"sun_candidate={result['aq1_ok']}"
        )

    def _format_plot_title(self, title, moog_values):
        if moog_values is None:
            return title
        requested = f"Req P/T=({moog_values['requested_pan']:.2f}, {moog_values['requested_tilt']:.2f})"
        if not moog_values.get("has_actual", False):
            return f"{title}\n{requested}"
        return (
            f"{title}\n"
            f"{requested} "
            f"Act P/T=({moog_values['actual_pan']:.2f}, {moog_values['actual_tilt']:.2f})"
        )

    def _plot_one(self, ax, display, fit, title, reference_fit=None, moog_values=None):
        ax.clear()
        ax.imshow(display, cmap="gray", origin="upper", interpolation="none")
        ax.set_title(self._format_plot_title(title, moog_values))
        ax.set_xlabel(f"column / {cfit.DISPLAY_STEP}")
        ax.set_ylabel(f"row / {cfit.DISPLAY_STEP}")

        theta = np.linspace(0, 2 * np.pi, 400)
        cx = fit["x"] / cfit.DISPLAY_STEP
        cy = fit["y"] / cfit.DISPLAY_STEP
        radius = fit["radius"] / cfit.DISPLAY_STEP
        ax.fill(cx + radius * np.cos(theta), cy + radius * np.sin(theta), color="red", alpha=0.12)
        ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), "r-", lw=2)
        ax.plot(cx, cy, "r+", ms=14, mew=2)
        if reference_fit is not None:
            ax.plot(reference_fit["x"] / cfit.DISPLAY_STEP, reference_fit["y"] / cfit.DISPLAY_STEP, "c+", ms=12, mew=2)

        ax.text(
            0.02,
            0.98,
            f"pixel=({fit['x']:.1f}, {fit['y']:.1f})\n"
            f"r={fit['radius']:.1f}, rmse={fit['rmse']:.1f}\n"
            f"n={fit['n_points']}, thr={fit['threshold_fraction']:.2f}\n"
            f"{fit.get('fit_method', '')}\n"
            f"sun_candidate={fit.get('sun_disk_candidate', False)}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.82},
        )

        h, w = display.shape
        x0 = min(0, cx - radius - 20)
        x1 = max(w, cx + radius + 20)
        y0 = min(0, cy - radius - 20)
        y1 = max(h, cy + radius + 20)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    def append_current(self):
        if self.result is None:
            return
        path = Path(self.log_path.get())
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = [
            "timestamp",
            "file",
            "path",
            "mode",
            "aq0_radius_scale",
            "aq1_radius_scale",
            "diff_deg",
            "center_zenith_diff_from_sun_deg",
            "center_zenith_diff_from_moog_deg",
            "aq0_center_zenith_from_moog_deg",
            "aq1_center_zenith_from_moog_deg",
            "aq0_center_zenith_from_sun_deg",
            "aq1_center_zenith_from_sun_deg",
            "fov_center_y_px",
            "pixel_only_diff_deg",
            "tilt_term_deg",
            "tilt_source",
            "sun_alt_term_deg",
            "aq0_zenith_deg",
            "aq1_zenith_deg",
            "aq0_x_px",
            "aq0_y_px",
            "aq0_radius_px",
            "aq0_rmse_px",
            "aq0_sun_candidate",
            "aq0_fit_method",
            "aq1_x_px",
            "aq1_y_px",
            "aq1_radius_px",
            "aq1_rmse_px",
            "aq1_sun_candidate",
            "aq1_fit_method",
            "aq0_tilt_deg",
            "aq1_tilt_deg",
            "aq0_sun_alt_deg",
            "aq1_sun_alt_deg",
        ]
        r = self.result
        f0 = r["fit0"]
        f1 = r["fit1"]
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "file": r["file"],
            "path": r["path"],
            "mode": r["mode"],
            "aq0_radius_scale": f"{r['aq0_radius_scale']:.6f}",
            "aq1_radius_scale": f"{r['aq1_radius_scale']:.6f}",
            "diff_deg": f"{r['diff']:.9f}",
            "center_zenith_diff_from_sun_deg": f"{r['center_zen_diff_from_sun']:.9f}",
            "center_zenith_diff_from_moog_deg": f"{r['center_zen_diff_from_moog']:.9f}",
            "aq0_center_zenith_from_moog_deg": f"{r['aq0_center_zen_from_moog']:.9f}",
            "aq1_center_zenith_from_moog_deg": f"{r['aq1_center_zen_from_moog']:.9f}",
            "aq0_center_zenith_from_sun_deg": f"{r['aq0_center_zen_from_sun']:.9f}",
            "aq1_center_zenith_from_sun_deg": f"{r['aq1_center_zen_from_sun']:.9f}",
            "fov_center_y_px": f"{r['fov_center_y']:.3f}",
            "pixel_only_diff_deg": f"{r['pixel_term']:.9f}",
            "tilt_term_deg": f"{r['tilt_term']:.9f}",
            "tilt_source": r["tilt_source"],
            "sun_alt_term_deg": f"{r['sun_alt_term']:.9f}",
            "aq0_zenith_deg": f"{r['zenith0']:.9f}",
            "aq1_zenith_deg": f"{r['zenith1']:.9f}",
            "aq0_x_px": f"{f0['x']:.3f}",
            "aq0_y_px": f"{f0['y']:.3f}",
            "aq0_radius_px": f"{f0['radius']:.3f}",
            "aq0_rmse_px": f"{f0['rmse']:.3f}",
            "aq0_sun_candidate": r["aq0_ok"],
            "aq0_fit_method": f0.get("fit_method", ""),
            "aq1_x_px": f"{f1['x']:.3f}",
            "aq1_y_px": f"{f1['y']:.3f}",
            "aq1_radius_px": f"{f1['radius']:.3f}",
            "aq1_rmse_px": f"{f1['rmse']:.3f}",
            "aq1_sun_candidate": r["aq1_ok"],
            "aq1_fit_method": f1.get("fit_method", ""),
            "aq0_tilt_deg": f"{r['aq0_tilt']:.9f}",
            "aq1_tilt_deg": f"{r['aq1_tilt']:.9f}",
            "aq0_sun_alt_deg": f"{r['aq0_sun_alt']:.9f}",
            "aq1_sun_alt_deg": f"{r['aq1_sun_alt']:.9f}",
        }
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        messagebox.showinfo("Appended", f"Saved current diff to:\n{path}")


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    root.geometry("1300x820")
    app = SunDiffGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

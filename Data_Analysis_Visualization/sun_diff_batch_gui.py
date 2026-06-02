import csv
import os
import queue
import tempfile
import threading
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-ultrasip"))

import numpy as np

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from sun_diff_gui import DEFAULT_LOG, analyze_h5, list_acquisition_groups


DEFAULT_BATCH_LOG = DEFAULT_LOG.with_name("sun_center_diff_batch_log.csv")


def finite_values(rows, key):
    values = [row[key] for row in rows if np.isfinite(row.get(key, np.nan))]
    return np.array(values, dtype=float)


def summarize_accuracy(rows):
    errors = finite_values(rows, "moog_minus_sun_diff_deg")
    abs_errors = np.abs(errors)
    return {
        "count": int(errors.size),
        "bias_deg": float(np.mean(errors)) if errors.size else np.nan,
        "mean_abs_error_deg": float(np.mean(abs_errors)) if errors.size else np.nan,
        "rmse_deg": float(np.sqrt(np.mean(errors * errors))) if errors.size else np.nan,
        "max_abs_error_deg": float(np.max(abs_errors)) if errors.size else np.nan,
    }


class SunDiffBatchGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ULTRASIP Sun Center Diff Batch")
        self.current_paths = []
        self.groups = []
        self.results = []
        self.worker_queue = queue.Queue()

        self.intensity_mode = tk.StringVar(value="p0")
        self.base_selector = tk.StringVar(value="")
        self.base_radius_scale = tk.DoubleVar(value=1.0)
        self.compare_radius_scale = tk.DoubleVar(value=1.0)
        self.log_path = tk.StringVar(value=str(DEFAULT_BATCH_LOG))
        self.files_status = tk.StringVar(value="No H5 files loaded.")
        self.status = tk.StringVar(value="Open or drop H5 files, then select one base acquisition and comparison acquisitions.")

        self._build_ui()
        self.root.after(100, self._poll_worker)

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.open_button = ttk.Button(toolbar, text="Open H5 Files", command=self.open_files)
        self.open_button.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Int").pack(side=tk.LEFT, padx=(8, 2))
        ttk.OptionMenu(toolbar, self.intensity_mode, self.intensity_mode.get(), "p0", "p0_p90").pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Base sun size").pack(side=tk.LEFT, padx=(8, 2))
        self.base_radius_label = ttk.Label(toolbar, text="1.00", width=4)
        self.base_radius_label.pack(side=tk.LEFT)
        ttk.Scale(
            toolbar,
            from_=0.60,
            to=1.80,
            orient=tk.HORIZONTAL,
            variable=self.base_radius_scale,
            command=self.on_base_radius_change,
            length=90,
        ).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(toolbar, text="Compare sun size").pack(side=tk.LEFT, padx=(8, 2))
        self.compare_radius_label = ttk.Label(toolbar, text="1.00", width=4)
        self.compare_radius_label.pack(side=tk.LEFT)
        ttk.Scale(
            toolbar,
            from_=0.60,
            to=1.80,
            orient=tk.HORIZONTAL,
            variable=self.compare_radius_scale,
            command=self.on_compare_radius_change,
            length=90,
        ).pack(side=tk.LEFT, padx=(2, 0))

        self.run_button = ttk.Button(toolbar, text="Run Batch", command=self.run_batch, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))

        self.save_button = ttk.Button(toolbar, text="Save CSV", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(toolbar, text="CSV Path", command=self.choose_log).pack(side=tk.LEFT, padx=(4, 0))

        path_row = ttk.Frame(self.root, padding=(4, 0, 4, 4))
        path_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(path_row, textvariable=self.log_path).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.drop_label = ttk.Label(
            self.root,
            text="Drop H5 files here" if DND_AVAILABLE else "Drag/drop needs tkinterdnd2; use Open H5 Files on this machine.",
            anchor="center",
            relief=tk.GROOVE,
            padding=4,
        )
        self.drop_label.pack(side=tk.TOP, fill=tk.X, padx=4)
        if DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

        files_frame = ttk.LabelFrame(self.root, text="Loaded H5 files", padding=6)
        files_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(files_frame, textvariable=self.files_status, justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)

        selectors = ttk.Frame(self.root, padding=4)
        selectors.pack(side=tk.TOP, fill=tk.BOTH)

        base_frame = ttk.LabelFrame(selectors, text="Base acquisition (left)", padding=6)
        base_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.base_combo = ttk.Combobox(base_frame, textvariable=self.base_selector, state="readonly")
        self.base_combo.pack(side=tk.TOP, fill=tk.X)

        compare_frame = ttk.LabelFrame(selectors, text="Comparison acquisitions (right, multi-select)", padding=6)
        compare_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.compare_list = tk.Listbox(compare_frame, selectmode=tk.EXTENDED, exportselection=False, height=8)
        self.compare_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        compare_scroll = ttk.Scrollbar(compare_frame, orient=tk.VERTICAL, command=self.compare_list.yview)
        compare_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.compare_list.configure(yscrollcommand=compare_scroll.set)

        self.info = ttk.Label(self.root, textvariable=self.status, padding=4, justify=tk.LEFT)
        self.info.pack(side=tk.TOP, fill=tk.X)

        table_frame = ttk.Frame(self.root, padding=4)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        columns = (
            "file",
            "right_acquisition",
            "sun_diff",
            "moog_diff",
            "moog_minus_sun",
            "abs_error",
            "right_radius",
            "right_rmse",
            "sun_candidate",
        )
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        headings = {
            "file": "File",
            "right_acquisition": "Right acquisition",
            "sun_diff": "Sun/FOV diff deg",
            "moog_diff": "Moog Actual diff deg",
            "moog_minus_sun": "Moog - Sun deg",
            "abs_error": "Abs error deg",
            "right_radius": "Sun radius px",
            "right_rmse": "RMSE px",
            "sun_candidate": "Sun candidate",
        }
        widths = {
            "file": 220,
            "right_acquisition": 190,
            "sun_diff": 130,
            "moog_diff": 140,
            "moog_minus_sun": 130,
            "abs_error": 110,
            "right_radius": 100,
            "right_rmse": 80,
            "sun_candidate": 95,
        }
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor=tk.CENTER)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=table_scroll.set)

    def open_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*")])
        if paths:
            self.load_files(paths)

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.load_files(paths)

    def choose_log(self):
        path = filedialog.asksaveasfilename(
            initialfile=DEFAULT_BATCH_LOG.name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if path:
            self.log_path.set(path)

    def load_files(self, paths):
        self.current_paths = [str(path) for path in paths]
        self.results = []
        self._clear_table()
        try:
            self.groups = list_acquisition_groups(self.current_paths[0])
            if len(self.groups) < 2:
                raise ValueError("H5 must contain at least two acquisition groups with UV Raw Images")
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            self.files_status.set("No valid H5 files loaded.")
            self.run_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            return

        self.base_combo["values"] = self.groups
        base = "Aquistion_0" if "Aquistion_0" in self.groups else self.groups[0]
        self.base_selector.set(base)

        self.compare_list.delete(0, tk.END)
        for group in self.groups:
            self.compare_list.insert(tk.END, group)
        selected = 0
        for idx, group in enumerate(self.groups):
            if group != base and selected < 3:
                self.compare_list.selection_set(idx)
                selected += 1

        self.run_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        shown = ", ".join(Path(path).name for path in self.current_paths[:4])
        if len(self.current_paths) > 4:
            shown += f", ... {len(self.current_paths) - 4} more"
        self.files_status.set(f"{len(self.current_paths)} file(s): {shown}")
        self.status.set(
            "Select acquisition names from the first file. Each H5 will compare those acquisition names inside itself."
        )

    def on_base_radius_change(self, value):
        self.base_radius_label.config(text=f"{float(value):.2f}")

    def on_compare_radius_change(self, value):
        self.compare_radius_label.config(text=f"{float(value):.2f}")

    def selected_comparison_specs(self):
        return [(i, self.compare_list.get(i)) for i in self.compare_list.curselection()]

    def run_batch(self):
        if not self.current_paths:
            return
        base = self.base_selector.get()
        base_idx = self.groups.index(base) if base in self.groups else 0
        comparisons = [(idx, name) for idx, name in self.selected_comparison_specs() if name != base]
        if not base:
            messagebox.showerror("Missing base", "Select one base acquisition.")
            return
        if not comparisons:
            messagebox.showerror("Missing comparisons", "Select at least one comparison acquisition.")
            return

        self.results = []
        self._clear_table()
        self.open_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.status.set(f"Running {len(self.current_paths)} file(s), {len(comparisons)} comparison(s) per file from base {base} ...")

        thread = threading.Thread(
            target=self._worker,
            args=(
                list(self.current_paths),
                self.intensity_mode.get(),
                self.base_radius_scale.get(),
                self.compare_radius_scale.get(),
                base,
                base_idx,
                comparisons,
            ),
            daemon=True,
        )
        thread.start()

    def _worker(self, paths, intensity_mode, base_radius_scale, compare_radius_scale, base, base_idx, comparisons):
        rows = []
        errors = []
        for path in paths:
            try:
                groups = list_acquisition_groups(path)
                file_base = base if base in groups else groups[base_idx]
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
                continue
            for right_idx, right in comparisons:
                try:
                    file_right = right if right in groups else groups[right_idx]
                except IndexError:
                    errors.append(f"{Path(path).name} {base}->{right}: acquisition index is unavailable")
                    continue
                if file_right == file_base:
                    errors.append(f"{Path(path).name} {file_base}->{file_right}: base and comparison are the same")
                    continue
                try:
                    result = analyze_h5(
                        path,
                        intensity_mode,
                        base_radius_scale,
                        compare_radius_scale,
                        file_base,
                        file_right,
                    )
                    rows.append(self._result_to_row(result))
                except Exception as exc:
                    errors.append(f"{Path(path).name} {file_base}->{file_right}: {exc}")
        self.worker_queue.put(("done", rows, errors))

    def _result_to_row(self, result):
        f0 = result["fit0"]
        f1 = result["fit1"]
        moog_minus_sun = result["center_zen_diff_from_moog"] - result["center_zen_diff_from_sun"]
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "file": result["file"],
            "path": result["path"],
            "mode": result["mode"],
            "base_acquisition": result["aq0_name"],
            "right_acquisition": result["aq1_name"],
            "base_dtilt_deg": result["aq0_dtilt"],
            "right_dtilt_deg": result["aq1_dtilt"],
            "base_radius_scale": result["aq0_radius_scale"],
            "right_radius_scale": result["aq1_radius_scale"],
            "sun_residual_diff_deg": result["diff"],
            "center_zenith_diff_from_sun_deg": result["center_zen_diff_from_sun"],
            "center_zenith_diff_from_moog_actual_deg": result["center_zen_diff_from_moog"],
            "moog_minus_sun_diff_deg": moog_minus_sun,
            "moog_abs_error_deg": abs(moog_minus_sun),
            "base_center_zenith_from_moog_deg": result["aq0_center_zen_from_moog"],
            "right_center_zenith_from_moog_deg": result["aq1_center_zen_from_moog"],
            "base_center_zenith_from_sun_deg": result["aq0_center_zen_from_sun"],
            "right_center_zenith_from_sun_deg": result["aq1_center_zen_from_sun"],
            "fov_center_y_px": result["fov_center_y"],
            "pixel_only_diff_deg": result["pixel_term"],
            "tilt_term_deg": result["tilt_term"],
            "tilt_source": result["tilt_source"],
            "sun_alt_term_deg": result["sun_alt_term"],
            "base_tilt_deg": result["aq0_tilt"],
            "right_tilt_deg": result["aq1_tilt"],
            "base_sun_alt_deg": result["aq0_sun_alt"],
            "right_sun_alt_deg": result["aq1_sun_alt"],
            "base_x_px": f0["x"],
            "base_y_px": f0["y"],
            "base_radius_px": f0["radius"],
            "base_rmse_px": f0["rmse"],
            "base_sun_candidate": result["aq0_ok"],
            "base_fit_method": f0.get("fit_method", ""),
            "right_x_px": f1["x"],
            "right_y_px": f1["y"],
            "right_radius_px": f1["radius"],
            "right_rmse_px": f1["rmse"],
            "right_sun_candidate": result["aq1_ok"],
            "right_fit_method": f1.get("fit_method", ""),
        }

    def _poll_worker(self):
        try:
            kind, rows, errors = self.worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker)
            return

        self.open_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL if self.current_paths else tk.DISABLED)
        if kind == "done":
            self.results = rows
            for row in rows:
                self._insert_row(row)
            self.save_button.config(state=tk.NORMAL if rows else tk.DISABLED)
            summary = summarize_accuracy(rows)
            status = (
                f"Done: {len(rows)} successful comparison(s), {len(errors)} error(s). "
                f"Moog accuracy vs FOV center from sun: "
                f"bias={summary['bias_deg']:.6f} deg, "
                f"MAE={summary['mean_abs_error_deg']:.6f} deg, "
                f"RMSE={summary['rmse_deg']:.6f} deg, "
                f"max abs={summary['max_abs_error_deg']:.6f} deg."
            )
            if errors:
                status += "\nErrors: " + " | ".join(errors[:5])
                if len(errors) > 5:
                    status += f" | ... {len(errors) - 5} more"
            self.status.set(status)
        self.root.after(100, self._poll_worker)

    def _clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)

    def _insert_row(self, row):
        self.table.insert(
            "",
            tk.END,
            values=(
                row["file"],
                row["right_acquisition"],
                f"{row['center_zenith_diff_from_sun_deg']:.6f}",
                f"{row['center_zenith_diff_from_moog_actual_deg']:.6f}",
                f"{row['moog_minus_sun_diff_deg']:.6f}",
                f"{row['moog_abs_error_deg']:.6f}",
                f"{row['right_radius_px']:.1f}",
                f"{row['right_rmse_px']:.1f}",
                row["right_sun_candidate"],
            ),
        )

    def save_results(self):
        if not self.results:
            return
        path = Path(self.log_path.get())
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "timestamp",
            "file",
            "path",
            "mode",
            "base_acquisition",
            "right_acquisition",
            "base_dtilt_deg",
            "right_dtilt_deg",
            "base_radius_scale",
            "right_radius_scale",
            "sun_residual_diff_deg",
            "center_zenith_diff_from_sun_deg",
            "center_zenith_diff_from_moog_actual_deg",
            "moog_minus_sun_diff_deg",
            "moog_abs_error_deg",
            "base_center_zenith_from_moog_deg",
            "right_center_zenith_from_moog_deg",
            "base_center_zenith_from_sun_deg",
            "right_center_zenith_from_sun_deg",
            "fov_center_y_px",
            "pixel_only_diff_deg",
            "tilt_term_deg",
            "tilt_source",
            "sun_alt_term_deg",
            "base_tilt_deg",
            "right_tilt_deg",
            "base_sun_alt_deg",
            "right_sun_alt_deg",
            "base_x_px",
            "base_y_px",
            "base_radius_px",
            "base_rmse_px",
            "base_sun_candidate",
            "base_fit_method",
            "right_x_px",
            "right_y_px",
            "right_radius_px",
            "right_rmse_px",
            "right_sun_candidate",
            "right_fit_method",
        ]
        exists = path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for row in self.results:
                writer.writerow({field: self._format_csv_value(row[field]) for field in fields})
        messagebox.showinfo("Saved", f"Saved {len(self.results)} row(s) to:\n{path}")

    def _format_csv_value(self, value):
        if isinstance(value, float):
            return f"{value:.9f}"
        return value


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    root.geometry("1120x720")
    root.minsize(980, 620)
    SunDiffBatchGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

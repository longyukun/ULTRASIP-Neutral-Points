# -*- coding: utf-8 -*-
"""
GUI for Moon tracking and visualization.

This GUI uses the same Moog/camera controller style as Measurement_QT_GUI.py,
but the tracking target is the Moon. It displays:
  - predicted Moon azimuth/altitude from Skyfield,
  - image-measured Moon azimuth/altitude from the fitted lunar limb,
  - azimuth/altitude errors over time.
"""
#
import math
import os
import sys
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ultrasip_matplotlib"))

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Moon_Tracker import (
    MoonEphemeris,
    LimbFit,
    clamp_pointing,
    fit_lunar_limb,
    image_offset_to_pointing_delta,
    wrap_to_180,
)
from Measurement_QT_GUI import (
    DEFAULT_EXPOSURE_US,
    PAN_MAX_DEG,
    PAN_MIN_DEG,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
    UV_DEG_PER_PIXEL,
    UV_IMAGE_HEIGHT_PX,
    UV_IMAGE_WIDTH_PX,
    PointingStatus,
    RealMoogController,
    RealVmbCameraController,
    SimMoogController,
    frame_to_qimage,
    import_qt,
    write_pointing_attrs,
)


QtCore, QtGui, QtWidgets, QT_API = import_qt()


@dataclass
class TrackSample:
    elapsed_s: float
    predicted_az_deg: float
    predicted_alt_deg: float
    measured_az_deg: float
    measured_alt_deg: float
    az_error_deg: float
    alt_error_deg: float
    target_pan_deg: float
    target_tilt_deg: float
    actual_pan_deg: float
    actual_tilt_deg: float
    fit_ok: bool
    fit_radius_px: float
    fit_residual_px: float


class SimMoonCameraController:
    def __init__(self, exposure_us: float = DEFAULT_EXPOSURE_US):
        self.connected = False
        self.frame_index = 0
        self.exposure_us = float(exposure_us)
        self.width = UV_IMAGE_WIDTH_PX
        self.height = UV_IMAGE_HEIGHT_PX
        self.pixels_per_degree = 1.0 / UV_DEG_PER_PIXEL
        self.moon_az_deg = 0.0
        self.moon_alt_deg = 0.0
        self.pan_offset_deg = -9.0
        self.tilt_offset_deg = 0.0

    def open(self):
        self.connected = True

    def close(self):
        self.connected = False

    def set_exposure(self, exposure_us: float):
        self.exposure_us = float(np.clip(exposure_us, 100.0, 1_000_000.0))

    def get_exposure(self):
        return self.exposure_us

    def set_scene(self, moon_az_deg: float, moon_alt_deg: float, pan_offset_deg: float, tilt_offset_deg: float):
        self.moon_az_deg = float(moon_az_deg)
        self.moon_alt_deg = float(moon_alt_deg)
        self.pan_offset_deg = float(pan_offset_deg)
        self.tilt_offset_deg = float(tilt_offset_deg)

    def get_frame(self, pan_deg: float = 0.0, tilt_deg: float = 0.0):
        self.frame_index += 1
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        boresight_az = pan_deg + self.pan_offset_deg
        boresight_alt = tilt_deg + self.tilt_offset_deg
        cos_alt = max(math.cos(math.radians(self.moon_alt_deg)), 0.2)
        dx_px = (self.moon_az_deg - boresight_az) * cos_alt * self.pixels_per_degree
        dy_px = (self.moon_alt_deg - boresight_alt) * self.pixels_per_degree
        cx = (self.width - 1) / 2.0 + dx_px
        cy = (self.height - 1) / 2.0 + dy_px
        radius = 0.26 / UV_DEG_PER_PIXEL

        r = np.hypot(xx - cx, yy - cy)
        disk = r <= radius
        phase_axis = xx < (cx + 0.18 * radius)
        limb_soft = np.clip((radius - r) / 8.0, 0.0, 1.0)
        albedo = 0.85 + 0.12 * np.sin((xx - cx) / 38.0) + 0.08 * np.cos((yy - cy) / 52.0)
        moon = 2700.0 * disk * phase_axis * limb_soft * albedo
        earthshine = 350.0 * disk * (~phase_axis) * limb_soft
        sky = 120.0 + 10.0 * np.sin(xx / 210.0) + 7.0 * np.cos(yy / 170.0)
        noise = np.random.default_rng(self.frame_index).normal(0.0, 9.0, sky.shape)
        exposure_scale = self.exposure_us / DEFAULT_EXPOSURE_US
        return np.clip((sky + moon + earthshine + noise) * exposure_scale, 0, 4095).astype(np.uint16)


class MoonPlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(8.2, 5.2), tight_layout=True)
        super().__init__(self.figure)
        self.track_ax = self.figure.add_subplot(2, 1, 1)
        self.error_ax = self.figure.add_subplot(2, 1, 2)
        self.samples = []
        self.refresh([])

    def refresh(self, samples):
        self.samples = list(samples)
        self.track_ax.clear()
        self.error_ax.clear()
        self.track_ax.set_title("Moon Az/Alt")
        self.track_ax.set_xlabel("Azimuth [deg]")
        self.track_ax.set_ylabel("Altitude [deg]")
        self.track_ax.grid(True, alpha=0.3)
        self.error_ax.set_title("Image-Measured Error")
        self.error_ax.set_xlabel("Elapsed [s]")
        self.error_ax.set_ylabel("Error [arcsec]")
        self.error_ax.grid(True, alpha=0.3)

        if self.samples:
            pred_az = [s.predicted_az_deg for s in self.samples]
            pred_alt = [s.predicted_alt_deg for s in self.samples]
            meas_az = [s.measured_az_deg for s in self.samples if np.isfinite(s.measured_az_deg)]
            meas_alt = [s.measured_alt_deg for s in self.samples if np.isfinite(s.measured_alt_deg)]
            self.track_ax.plot(pred_az, pred_alt, color="#1f77b4", marker="o", markersize=3, label="Predicted")
            if meas_az:
                self.track_ax.plot(meas_az, meas_alt, color="#d62728", marker="x", markersize=4, label="Measured")
            self.track_ax.legend(loc="best")

            elapsed = [s.elapsed_s for s in self.samples]
            az_err = [s.az_error_deg * 3600.0 for s in self.samples]
            alt_err = [s.alt_error_deg * 3600.0 for s in self.samples]
            self.error_ax.plot(elapsed, az_err, color="#9467bd", label="Az error")
            self.error_ax.plot(elapsed, alt_err, color="#2ca02c", label="Alt error")
            self.error_ax.axhline(0.0, color="black", linewidth=0.8)
            self.error_ax.legend(loc="best")

        self.draw_idle()


class ImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(430, 430)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; color:#ddd;")
        self.setText("No frame")
        self.frame = None
        self.fit = None

    def set_frame(self, frame: np.ndarray, fit: Optional[LimbFit] = None):
        self.frame = frame
        self.fit = fit
        pixmap = QtGui.QPixmap.fromImage(frame_to_qimage(frame, QtGui))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen_center = QtGui.QPen(QtGui.QColor("#00d2ff"))
        pen_center.setWidth(3)
        painter.setPen(pen_center)
        painter.drawLine(pixmap.width() // 2 - 22, pixmap.height() // 2, pixmap.width() // 2 + 22, pixmap.height() // 2)
        painter.drawLine(pixmap.width() // 2, pixmap.height() // 2 - 22, pixmap.width() // 2, pixmap.height() // 2 + 22)

        if fit and fit.ok:
            sx = pixmap.width() / frame.shape[1]
            sy = pixmap.height() / frame.shape[0]
            pen_fit = QtGui.QPen(QtGui.QColor("#ffcc00"))
            pen_fit.setWidth(3)
            painter.setPen(pen_fit)
            painter.drawEllipse(
                QtCore.QPointF(fit.center_x_px * sx, fit.center_y_px * sy),
                fit.radius_px * sx,
                fit.radius_px * sy,
            )
        painter.end()
        self.setPixmap(pixmap.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frame is not None:
            self.set_frame(self.frame, self.fit)


class MoonTrackerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ULTRASIP Moon Tracker")
        self.resize(1240, 820)
        self.simulation = True
        self.moog = SimMoogController()
        self.camera = SimMoonCameraController()
        self.ephemeris = None
        self.h5 = None
        self.h5_path = None
        self.samples = []
        self.start_time = None
        self.running = False
        self.sample_index = 0
        self.last_frame = None
        self.last_fit = None
        self.centering = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tracking_tick)

        self.build_ui()
        self.refresh_status(PointingStatus())

    def build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(390)
        form = QtWidgets.QGridLayout(controls)
        row = 0

        self.sim_check = QtWidgets.QCheckBox("Simulation")
        self.sim_check.setChecked(True)
        self.sim_check.stateChanged.connect(self.set_simulation)
        form.addWidget(self.sim_check, row, 0, 1, 2)
        row += 1

        self.moog_port = QtWidgets.QLineEdit("COM7")
        self.latitude = self.spin(-90, 90, 32.23134, 6)
        self.longitude = self.spin(-180, 180, -110.94712, 6)
        self.elevation_m = self.spin(-500, 10000, 0.0, 1)
        self.pan_offset = self.spin(-360, 360, -9.0, 3)
        self.tilt_offset = self.spin(-180, 180, 0.0, 3)
        self.exposure_us = self.spin(10, 1_000_000, DEFAULT_EXPOSURE_US, 1)
        self.deg_per_pixel = self.spin(0.00001, 0.05, UV_DEG_PER_PIXEL, 6)
        self.x_sign = self.spin(-1, 1, 1.0, 0)
        self.y_sign = self.spin(-1, 1, 1.0, 0)
        self.interval_s = self.spin(0.5, 300, 5.0, 1)
        self.duration_s = self.spin(1, 86400, 900.0, 1)
        self.location = QtWidgets.QLineEdit("MeinelRoof")
        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_output)

        fields = [
            ("Moog port", self.moog_port),
            ("Latitude", self.latitude),
            ("Longitude", self.longitude),
            ("Elevation [m]", self.elevation_m),
            ("Pan offset [deg]", self.pan_offset),
            ("Tilt offset [deg]", self.tilt_offset),
            ("Exposure [us]", self.exposure_us),
            ("Deg per pixel", self.deg_per_pixel),
            ("Image X sign", self.x_sign),
            ("Image Y sign", self.y_sign),
            ("Interval [s]", self.interval_s),
            ("Duration [s]", self.duration_s),
            ("Location", self.location),
        ]
        for label, widget in fields:
            form.addWidget(QtWidgets.QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
            row += 1

        form.addWidget(QtWidgets.QLabel("Output folder"), row, 0)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.output_dir)
        out_row.addWidget(self.browse_btn)
        form.addLayout(out_row, row, 1)
        row += 1

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_devices)
        self.center_btn = QtWidgets.QPushButton("Center Moon")
        self.center_btn.clicked.connect(self.center_moon)
        self.start_btn = QtWidgets.QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.close_btn = QtWidgets.QPushButton("Close Devices")
        self.close_btn.clicked.connect(self.close_devices)
        for btn in (self.connect_btn, self.center_btn, self.start_btn, self.stop_btn, self.close_btn):
            btn.setMinimumHeight(34)
            form.addWidget(btn, row, 0, 1, 2)
            row += 1

        self.status = QtWidgets.QLabel("Not connected")
        self.status.setWordWrap(True)
        self.fit_status = QtWidgets.QLabel("")
        self.fit_status.setWordWrap(True)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(600)
        self.log.setMinimumHeight(150)
        form.addWidget(QtWidgets.QLabel("Status"), row, 0)
        form.addWidget(self.status, row, 1)
        row += 1
        form.addWidget(QtWidgets.QLabel("Fit"), row, 0)
        form.addWidget(self.fit_status, row, 1)
        row += 1
        form.addWidget(self.log, row, 0, 1, 2)
        form.setRowStretch(row, 1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.image_view = ImageView()
        self.plot = MoonPlotCanvas()
        right_layout.addWidget(self.image_view, 3)
        right_layout.addWidget(self.plot, 2)
        root.addWidget(controls)
        root.addWidget(right, 1)

    def spin(self, min_value, max_value, value, decimals):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(float(min_value), float(max_value))
        box.setDecimals(int(decimals))
        box.setValue(float(value))
        box.setSingleStep(10 ** -int(decimals) if decimals > 0 else 1.0)
        return box

    def log_msg(self, message):
        self.log.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')} {message}")

    def browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Moon Tracking Output Folder", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def set_simulation(self):
        self.simulation = self.sim_check.isChecked()
        self.close_devices()
        self.moog = SimMoogController() if self.simulation else RealMoogController()
        self.camera = SimMoonCameraController(self.exposure_us.value()) if self.simulation else RealVmbCameraController(self.exposure_us.value())
        self.log_msg("Simulation mode" if self.simulation else "Hardware mode")

    def ensure_ephemeris(self):
        if self.ephemeris is None:
            self.ephemeris = MoonEphemeris(skyfield_data_dir=os.path.join(os.getcwd(), "skyfield-data"))
        return self.ephemeris

    def connect_devices(self):
        try:
            self.ensure_ephemeris()
            self.camera.set_exposure(self.exposure_us.value())
            self.camera.open()
            status = self.moog.open(self.moog_port.text())
            self.refresh_status(status)
            self.log_msg("Devices connected")
            self.capture_preview()
        except Exception as exc:
            self.log_msg(f"Connect failed: {exc}")

    def close_devices(self):
        self.stop_tracking()
        try:
            if self.h5 is not None:
                self.h5.close()
                self.h5 = None
            self.camera.close()
            self.moog.close()
        except Exception as exc:
            self.log_msg(f"Close warning: {exc}")
        self.refresh_status(self.moog.get_status() if hasattr(self.moog, "get_status") else PointingStatus())

    def refresh_status(self, status):
        self.status.setText(
            f"Pan={status.pan_deg:.3f} deg, Tilt={status.tilt_deg:.3f} deg, "
            f"connected={status.connected}, running={self.running}"
        )

    def current_observation(self):
        return self.ensure_ephemeris().observe(
            datetime.now().astimezone(),
            self.latitude.value(),
            self.longitude.value(),
            self.elevation_m.value(),
        )

    def target_from_observation(self, obs):
        return clamp_pointing(obs.azimuth_deg - self.pan_offset.value(), obs.altitude_deg - self.tilt_offset.value())

    def update_sim_scene(self, obs):
        if isinstance(self.camera, SimMoonCameraController):
            self.camera.set_scene(obs.azimuth_deg, obs.altitude_deg, self.pan_offset.value(), self.tilt_offset.value())

    def capture_preview(self):
        obs = self.current_observation()
        self.update_sim_scene(obs)
        status = self.moog.get_status()
        frame = self.camera.get_frame(status.pan_deg, status.tilt_deg)
        expected_radius_px = (obs.angular_diameter_deg / 2.0) / self.deg_per_pixel.value()
        fit = fit_lunar_limb(frame, expected_radius_px=expected_radius_px)
        self.last_frame = frame
        self.last_fit = fit
        self.image_view.set_frame(frame, fit)
        self.update_fit_label(fit)
        return frame, fit, obs, status

    def update_fit_label(self, fit):
        if fit and fit.ok:
            self.fit_status.setText(
                f"center=({fit.center_x_px:.1f}, {fit.center_y_px:.1f}) px, "
                f"radius={fit.radius_px:.1f} px, residual={fit.residual_px:.2f} px"
            )
        else:
            self.fit_status.setText(f"fit failed: {fit.reason if fit else 'no fit'}")

    def center_moon(self):
        try:
            for idx in range(4):
                frame, fit, obs, status = self.capture_preview()
                if not fit.ok:
                    self.log_msg(f"Center failed: {fit.reason}")
                    return
                image_center_x = (frame.shape[1] - 1) / 2.0
                image_center_y = (frame.shape[0] - 1) / 2.0
                residual_px = math.hypot(fit.center_x_px - image_center_x, fit.center_y_px - image_center_y)
                if residual_px <= 5.0:
                    self.log_msg(f"Centered moon in {idx + 1} iteration(s)")
                    return
                dpan, dtilt = image_offset_to_pointing_delta(
                    fit,
                    frame.shape,
                    self.deg_per_pixel.value(),
                    obs.altitude_deg,
                    self.x_sign.value(),
                    self.y_sign.value(),
                )
                moved = self.moog.move_absolute(status.pan_deg + dpan, status.tilt_deg + dtilt)
                self.refresh_status(moved)
                time.sleep(0.2)
            self.log_msg("Center iterations completed")
        except Exception as exc:
            self.log_msg(f"Center failed: {exc}")

    def open_log_file(self):
        os.makedirs(self.output_dir.text(), exist_ok=True)
        date_dir = datetime.now().strftime("%Y_%m_%d")
        folder = os.path.join(self.output_dir.text(), date_dir)
        os.makedirs(folder, exist_ok=True)
        self.h5_path = os.path.join(folder, f"{self.location.text()}_MoonGUI_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.h5")
        self.h5 = h5py.File(self.h5_path, "w")
        meta = self.h5.create_group("Measurement_Metadata")
        meta.attrs["Mode"] = "Simulation" if self.simulation else "Hardware"
        meta.attrs["Created Local Time"] = datetime.now().astimezone().isoformat(timespec="seconds")
        meta.attrs["Latitude"] = self.latitude.value()
        meta.attrs["Longitude"] = self.longitude.value()
        meta.attrs["Elevation [m]"] = self.elevation_m.value()
        meta.attrs["Pan_Offset"] = self.pan_offset.value()
        meta.attrs["Tilt_Offset"] = self.tilt_offset.value()
        meta.attrs["UV Pixel Scale [deg/pixel]"] = self.deg_per_pixel.value()
        meta.attrs["Center Method"] = "RANSAC lunar limb circle fit; no brightness centroid"

    def start_tracking(self):
        try:
            if not self.camera.connected:
                self.connect_devices()
            if self.h5 is not None:
                self.h5.close()
                self.h5 = None
            self.open_log_file()
            self.samples = []
            self.sample_index = 0
            self.start_time = time.time()
            self.running = True
            self.timer.start(int(self.interval_s.value() * 1000))
            self.tracking_tick()
            self.log_msg(f"Tracking started: {self.h5_path}")
        except Exception as exc:
            self.log_msg(f"Start failed: {exc}")

    def stop_tracking(self):
        if self.running:
            self.log_msg("Tracking stopped")
        self.running = False
        self.timer.stop()
        if self.h5 is not None:
            self.h5["Measurement_Metadata"].attrs["Completed Local Time"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self.h5["Measurement_Metadata"].attrs["Samples Written"] = self.sample_index
            self.h5.close()
            self.h5 = None
        self.refresh_status(self.moog.get_status() if hasattr(self.moog, "get_status") else PointingStatus())

    def tracking_tick(self):
        if not self.running:
            return
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        if elapsed > self.duration_s.value():
            self.stop_tracking()
            return

        try:
            obs = self.current_observation()
            self.update_sim_scene(obs)
            target_pan, target_tilt = self.target_from_observation(obs)
            status = self.moog.move_absolute(target_pan, target_tilt)
            self.refresh_status(status)
            time.sleep(0.05)
            frame = self.camera.get_frame(status.pan_deg, status.tilt_deg)
            expected_radius_px = (obs.angular_diameter_deg / 2.0) / self.deg_per_pixel.value()
            fit = fit_lunar_limb(frame, expected_radius_px=expected_radius_px)
            self.image_view.set_frame(frame, fit)
            self.update_fit_label(fit)

            measured_az = np.nan
            measured_alt = np.nan
            az_error = np.nan
            alt_error = np.nan
            if fit.ok:
                dpan, dtilt = image_offset_to_pointing_delta(
                    fit,
                    frame.shape,
                    self.deg_per_pixel.value(),
                    obs.altitude_deg,
                    self.x_sign.value(),
                    self.y_sign.value(),
                )
                measured_az = status.pan_deg + self.pan_offset.value() + dpan
                measured_alt = status.tilt_deg + self.tilt_offset.value() + dtilt
                az_error = wrap_to_180(measured_az - obs.azimuth_deg)
                alt_error = measured_alt - obs.altitude_deg

            sample = TrackSample(
                elapsed_s=elapsed,
                predicted_az_deg=obs.azimuth_deg,
                predicted_alt_deg=obs.altitude_deg,
                measured_az_deg=float(measured_az),
                measured_alt_deg=float(measured_alt),
                az_error_deg=float(az_error),
                alt_error_deg=float(alt_error),
                target_pan_deg=float(target_pan),
                target_tilt_deg=float(target_tilt),
                actual_pan_deg=float(status.pan_deg),
                actual_tilt_deg=float(status.tilt_deg),
                fit_ok=bool(fit.ok),
                fit_radius_px=float(fit.radius_px),
                fit_residual_px=float(fit.residual_px),
            )
            self.samples.append(sample)
            self.plot.refresh(self.samples)
            self.write_sample(sample, obs, status, fit, frame)
        except Exception as exc:
            self.log_msg(f"Tracking tick failed: {exc}")

    def write_sample(self, sample, obs, status, fit, frame):
        if self.h5 is None:
            return
        group = self.h5.create_group(f"Tracking_{self.sample_index}")
        group.attrs["Timestamp Local"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        group.attrs["Timestamp UTC"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        group.attrs["Predicted Moon Azimuth [deg]"] = sample.predicted_az_deg
        group.attrs["Predicted Moon Altitude [deg]"] = sample.predicted_alt_deg
        group.attrs["Measured Moon Azimuth [deg]"] = sample.measured_az_deg
        group.attrs["Measured Moon Altitude [deg]"] = sample.measured_alt_deg
        group.attrs["Moon Azimuth Error [deg]"] = sample.az_error_deg
        group.attrs["Moon Altitude Error [deg]"] = sample.alt_error_deg
        group.attrs["Moon Distance [km]"] = obs.distance_km
        group.attrs["Moon Angular Diameter [deg]"] = obs.angular_diameter_deg
        group.attrs["Ephemeris Backend"] = obs.backend
        write_pointing_attrs(group.attrs, status, sample.target_pan_deg, sample.target_tilt_deg)
        group.attrs["Fit OK"] = int(fit.ok)
        group.attrs["Fit Reason"] = fit.reason
        group.attrs["Moon Center X [px]"] = fit.center_x_px
        group.attrs["Moon Center Y [px]"] = fit.center_y_px
        group.attrs["Moon Radius [px]"] = fit.radius_px
        group.attrs["Fit Median Residual [px]"] = fit.residual_px
        group.attrs["Fit Inlier Count"] = fit.inlier_count
        group.attrs["Fit Total Edge Count"] = fit.total_edge_count
        group.create_dataset("UV Image", data=frame, compression="gzip")
        self.sample_index += 1
        self.h5.flush()

    def closeEvent(self, event):
        self.close_devices()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MoonTrackerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

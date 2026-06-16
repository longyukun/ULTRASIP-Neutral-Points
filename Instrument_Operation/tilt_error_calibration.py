# -*- coding: utf-8 -*-
"""
Tilt Error Sinusoidal Calibration Tool

At each commanded pan angle the operator clicks a ground feature to lock a
template.  The system then dithers the tilt axis through a small sequence of
offsets (e.g. ±0.5°, ±1.0°, ±1.5°) and records the pixel displacement of the
feature at each dither step.  A linear fit of  Δy_px vs Δtilt_cmd  gives the
pixel scale and the zero-crossing offset, which is the tilt axis error at that
pan angle.  After all pan angles are measured a sinusoidal model

    δtilt(pan) = A · sin(pan + φ) + C

is fitted to the per-pan tilt errors.

Hardware: Moog pan/tilt · Zaber polarizer stage (fixed angle) · Allied Vision UV camera.
"""

import csv
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from Measurement_QT_GUI import (
    UV_ARCSEC_PER_PIXEL,
    UV_IMAGE_HEIGHT_PX,
    UV_IMAGE_WIDTH_PX,
    MOOG_SETTLE_BEFORE_CAPTURE_SEC,
    PREVIEW_MAX_SIDE_PX,
    DEFAULT_MOOG_PORT,
    DEFAULT_ZABER_PORT,
    DEFAULT_EXPOSURE_US,
    PointingStatus,
    RealMoogController,
    RealPolarizerController,
    RealVmbCameraController,
    import_qt,
    clamp_pointing,
)

APP_TITLE = "Tilt Error Calibration"
APP_VER   = "2.1.0"

MATCH_WARN_SCORE  = 0.45           # NCC score below this → yellow warning
DITHER_SETTLE_S   = 1.2            # settle time between dither steps (seconds)
NOMINAL_PX_PER_DEG = 3600.0 / 7.20  # ≈ 500 px/° — used only for yellow preview marker

DEFAULT_DITHER_OFFSET_DEG = 1.0   # ±N° in both pan and tilt → 3×3 grid



# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class DitherStep:
    dpan_cmd_deg:    float
    dtilt_cmd_deg:   float
    dpan_actual_deg: float   # actual pan  − pan_0
    dtilt_actual_deg: float  # actual tilt − tilt_0
    dcx_px: float            # centroid x displacement (positive = right)
    dcy_px: float            # centroid y displacement (positive = down)
    match_score: float


@dataclass
class PanPoint:
    """Measurement at one pan angle."""
    pan_index:      int
    pan_cmd_deg:    float
    pan_actual_deg: float
    tilt_0_deg:     float
    dither_steps:   List[DitherStep] = field(default_factory=list)
    # filled after 2-D linear fit
    pan_error_arcsec:  float = 0.0
    tilt_error_arcsec: float = 0.0
    pan_px_per_deg:    float = 0.0
    tilt_px_per_deg:   float = 0.0
    fit_r2_x:          float = 0.0
    fit_r2_y:          float = 0.0
    skipped:           bool  = False
    timestamp:         str   = ""


# ── image helpers ──────────────────────────────────────────────────────────────

def weighted_centroid(patch: np.ndarray) -> Tuple[float, float]:
    img = patch.astype(np.float64)
    img = np.maximum(img - np.percentile(img, 20.0), 0.0)
    total = img.sum()
    if total <= 0:
        h, w = img.shape
        return w / 2.0, h / 2.0
    yy, xx = np.indices(img.shape)
    return float((xx * img).sum() / total), float((yy * img).sum() / total)


def match_and_centroid(
    frame: np.ndarray,
    template: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Full-frame NCC template match followed by centroid refinement.
    Returns (cx, cy, ncc_score) in full-frame pixels.
    """
    h, w   = frame.shape[:2]
    th, tw = template.shape[:2]

    if cv2 is None or h <= th or w <= tw:
        lcx, lcy = weighted_centroid(frame)
        return lcx, lcy, 0.0

    res = cv2.matchTemplate(
        frame.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED
    )
    _, score, _, (mx, my) = cv2.minMaxLoc(res)
    pr0 = my;              pr1 = min(h, pr0 + th)
    pc0 = mx;              pc1 = min(w, pc0 + tw)
    lcx, lcy = weighted_centroid(frame[pr0:pr1, pc0:pc1])
    return pc0 + lcx, pr0 + lcy, float(score)


# ── math helpers ───────────────────────────────────────────────────────────────

def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def fit_2d(dpan: np.ndarray, dtilt: np.ndarray, dcx: np.ndarray, dcy: np.ndarray):
    """
    Fit independently:
        dcx = a_x * dpan + b_x * dtilt + cx
        dcy = a_y * dpan + b_y * dtilt + cy

    Returns dict with keys:
        a_x, b_x, cx, r2_x   (x / pan-pixel equation)
        a_y, b_y, cy, r2_y   (y / tilt-pixel equation)
        pan_error_deg         = -cx / a_x  (pan error at nominal position)
        tilt_error_deg        = -cy / b_y  (tilt error at nominal position)
    """
    if len(dpan) < 4:
        return None
    A = np.column_stack([dpan, dtilt, np.ones_like(dpan)])
    try:
        cx_coef, *_ = np.linalg.lstsq(A, dcx, rcond=None)
        cy_coef, *_ = np.linalg.lstsq(A, dcy, rcond=None)
    except Exception:
        return None

    a_x, b_x, cx = cx_coef
    a_y, b_y, cy = cy_coef
    r2_x = _r2(dcx, A @ cx_coef)
    r2_y = _r2(dcy, A @ cy_coef)

    pan_error_deg  = -float(cx) / float(a_x) if abs(a_x) > 1e-6 else 0.0
    tilt_error_deg = -float(cy) / float(b_y) if abs(b_y) > 1e-6 else 0.0

    return dict(
        a_x=float(a_x), b_x=float(b_x), cx=float(cx), r2_x=float(r2_x),
        a_y=float(a_y), b_y=float(b_y), cy=float(cy), r2_y=float(r2_y),
        pan_error_deg=pan_error_deg,
        tilt_error_deg=tilt_error_deg,
    )


def fit_sinusoid(pan_deg: np.ndarray, dtilt_arcsec: np.ndarray):
    """Fit A·sin(pan_rad + φ) + C.  Returns (A, phi_deg, C, fitted) or None."""
    if len(pan_deg) < 3:
        return None
    x = np.deg2rad(pan_deg)
    y = dtilt_arcsec
    if _SCIPY:
        def model(xv, A, phi, C):
            return A * np.sin(xv + phi) + C
        try:
            p0 = [float(np.ptp(y)) / 2.0, 0.0, float(np.mean(y))]
            popt, _ = curve_fit(model, x, y, p0=p0, maxfev=8000)
            return float(popt[0]), float(np.degrees(popt[1])), float(popt[2]), model(x, *popt)
        except Exception:
            pass
    # fallback: a·sin + b·cos + C
    A_mat = np.column_stack([np.sin(x), np.cos(x), np.ones_like(x)])
    try:
        coeffs, *_ = np.linalg.lstsq(A_mat, y, rcond=None)
        a, b, C = coeffs
        A   = math.sqrt(a ** 2 + b ** 2)
        phi = math.atan2(b, a)
        return float(A), float(np.degrees(phi)), float(C), A_mat @ coeffs
    except Exception:
        return None


# ── Qt GUI ─────────────────────────────────────────────────────────────────────

def main():
    QtCore, QtGui, QtWidgets, _qt_name = import_qt()

    align_center = getattr(getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt), "AlignCenter")
    keep_aspect  = getattr(getattr(QtCore.Qt, "AspectRatioMode", QtCore.Qt), "KeepAspectRatio")
    smooth_xform = getattr(getattr(QtCore.Qt, "TransformationMode", QtCore.Qt), "SmoothTransformation")

    # ── spinbox subclasses that ignore mouse-wheel to prevent accidental edits ─
    class NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
        def wheelEvent(self, event):
            event.ignore()

    class NoWheelSpinBox(QtWidgets.QSpinBox):
        def wheelEvent(self, event):
            event.ignore()

    # ── drag-to-select camera preview ─────────────────────────────────────────
    class CameraView(QtWidgets.QLabel):
        """
        Displays camera frames.  The user drags a rubber-band rectangle to
        define the template region; on mouse release the signal `roi_selected`
        fires with (x0, y0, x1, y1) in full-image pixel coordinates.
        """
        roi_selected = QtCore.Signal(float, float, float, float)

        def __init__(self):
            super().__init__()
            self.setFixedSize(500, 500)
            self.setAlignment(align_center)
            self.setText("Camera not connected")
            self.setStyleSheet("background:#111;color:#ddd;border:1px solid #444;")
            self._pixmap      = None
            self._frame_shape = (UV_IMAGE_HEIGHT_PX, UV_IMAGE_WIDTH_PX)
            cursor = getattr(getattr(QtCore.Qt, "CursorShape", QtCore.Qt), "CrossCursor")
            self.setCursor(QtGui.QCursor(cursor))
            self._drag_start: Optional[QtCore.QPoint] = None
            self._drag_rect:  Optional[QtCore.QRect]  = None   # in widget coords

        def set_frame(self, frame: np.ndarray, overlay_fn=None):
            self._frame_shape = frame.shape[:2]
            stride = max(1, int(math.ceil(max(frame.shape[:2]) / float(PREVIEW_MAX_SIDE_PX))))
            preview = frame[::stride, ::stride]
            lo, hi  = np.percentile(preview, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            view = np.clip(
                (preview.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255
            ).astype(np.uint8)
            h, w   = view.shape
            fmt_ns = getattr(QtGui.QImage, "Format", QtGui.QImage)
            fmt    = getattr(fmt_ns, "Format_Grayscale8", getattr(fmt_ns, "Format_Indexed8"))
            qimg   = QtGui.QImage(view.data, w, h, view.strides[0], fmt).copy()
            pix    = QtGui.QPixmap.fromImage(qimg)
            if overlay_fn:
                overlay_fn(pix, stride)
            self._pixmap = pix
            self._rescale()

        # ── coordinate helpers ─────────────────────────────────────────────────
        def _widget_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
            """Convert widget pixel → full-image pixel (clamped)."""
            if self._pixmap is None:
                return 0.0, 0.0
            side  = min(self.width(), self.height())
            ox    = (self.width()  - side) / 2
            oy    = (self.height() - side) / 2
            scale = self._pixmap.width() / side       # pixmap px per widget px
            px    = (wx - ox) * scale
            py    = (wy - oy) * scale
            ih, iw = self._frame_shape
            img_x = px / self._pixmap.width()  * iw
            img_y = py / self._pixmap.height() * ih
            return float(np.clip(img_x, 0, iw - 1)), float(np.clip(img_y, 0, ih - 1))

        # ── mouse events ───────────────────────────────────────────────────────
        def mousePressEvent(self, event):
            left = getattr(getattr(QtCore.Qt, "MouseButton", QtCore.Qt), "LeftButton")
            if event.button() != left or self._pixmap is None:
                return
            self._drag_start = event.pos()
            self._drag_rect  = None

        def mouseMoveEvent(self, event):
            if self._drag_start is None:
                return
            self._drag_rect = QtCore.QRect(self._drag_start, event.pos()).normalized()
            self.update()   # trigger paintEvent for rubber-band

        def mouseReleaseEvent(self, event):
            left = getattr(getattr(QtCore.Qt, "MouseButton", QtCore.Qt), "LeftButton")
            if event.button() != left or self._drag_start is None:
                return
            end = event.pos()
            rect = QtCore.QRect(self._drag_start, end).normalized()
            self._drag_start = None
            self._drag_rect  = None
            self.update()
            # ignore tiny accidental clicks (< 10 px in either dimension)
            if rect.width() < 10 or rect.height() < 10:
                return
            x0, y0 = self._widget_to_image(rect.left(),  rect.top())
            x1, y1 = self._widget_to_image(rect.right(), rect.bottom())
            self.roi_selected.emit(x0, y0, x1, y1)

        def paintEvent(self, event):
            super().paintEvent(event)
            if self._drag_rect is not None:
                painter = QtGui.QPainter(self)
                pen = QtGui.QPen(QtGui.QColor(255, 220, 0))
                pen.setWidth(2)
                pen.setStyle(
                    getattr(getattr(QtCore.Qt, "PenStyle", QtCore.Qt), "DashLine")
                )
                painter.setPen(pen)
                painter.drawRect(self._drag_rect)
                painter.end()

        def _rescale(self):
            if not self._pixmap:
                return
            side   = max(1, min(self.width(), self.height()))
            scaled = self._pixmap.scaled(QtCore.QSize(side, side), keep_aspect, smooth_xform)
            self.setPixmap(scaled)

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._rescale()

    # ── main window ───────────────────────────────────────────────────────────
    class CalibWindow(QtWidgets.QMainWindow):

        _sig_frame      = QtCore.Signal(object)   # np.ndarray
        _sig_log        = QtCore.Signal(str)
        _sig_status     = QtCore.Signal(object)   # PointingStatus
        _sig_await_click = QtCore.Signal()         # ask user to click target
        _sig_dither_result = QtCore.Signal(object) # PanPoint (dithering done)

        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"{APP_TITLE}  v{APP_VER}")
            self.resize(1150, 780)
            self.setStyleSheet(
                "QWidget{font-size:11px;}"
                "QGroupBox{font-weight:600;margin-top:7px;}"
                "QGroupBox::title{subcontrol-origin:margin;left:6px;padding:0 3px;}"
                "QPushButton{min-height:22px;padding:2px 8px;}"
                "QLineEdit,QDoubleSpinBox,QSpinBox{min-height:20px;}"
            )

            self.moog      = RealMoogController()
            self.polarizer = RealPolarizerController()
            self.camera    = RealVmbCameraController(exposure_us=DEFAULT_EXPOSURE_US)
            self._camera_lock = threading.Lock()   # one grab at a time

            self._last_frame: Optional[np.ndarray] = None
            self._template:   Optional[np.ndarray] = None
            self._target_cx:  Optional[float]      = None
            self._target_cy:  Optional[float]      = None
            self._match_cx:   Optional[float]      = None   # latest matched centroid
            self._match_cy:   Optional[float]      = None
            self._pred_cx:    Optional[float]      = None   # theoretical centroid (yellow)
            self._pred_cy:    Optional[float]      = None
            # template bounding box in full-image coords (x0,y0,x1,y1)
            self._tmpl_rect:  Optional[Tuple[int,int,int,int]] = None
            self._live_preview = False   # True while move/settle preview thread runs

            self._pan_points: List[PanPoint] = []
            self._running     = False
            self._click_event = threading.Event()  # user clicked target
            self._step_event  = threading.Event()  # user accepted/retried/skipped
            self._step_action = ""

            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.setInterval(800)
            self._preview_timer.timeout.connect(self._request_preview)

            self._sig_frame.connect(self._on_frame)
            self._sig_log.connect(self._on_log)
            self._sig_status.connect(self._on_status)
            self._sig_await_click.connect(self._on_await_click)
            self._sig_dither_result.connect(self._on_dither_result)

            self._build_ui()
            self._refresh_ports()

        # ── UI ────────────────────────────────────────────────────────────────
        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)

            # ── left panel ────────────────────────────────────────────────────
            left = QtWidgets.QVBoxLayout()
            left.setSpacing(6)
            root.addLayout(left, stretch=0)

            # hardware
            hw = QtWidgets.QGroupBox("Hardware")
            hf = QtWidgets.QFormLayout(hw)
            self._moog_port  = QtWidgets.QComboBox()
            self._zaber_port = QtWidgets.QComboBox()
            ref_btn = QtWidgets.QPushButton("Refresh")
            ref_btn.clicked.connect(self._refresh_ports)
            pr = QtWidgets.QHBoxLayout()
            pr.addWidget(self._moog_port, 1)
            pr.addWidget(self._zaber_port, 1)
            pr.addWidget(ref_btn)
            hf.addRow("Ports (Moog / Zaber):", pr)
            self._conn_btn    = QtWidgets.QPushButton("Connect All")
            self._hw_lbl      = QtWidgets.QLabel("Disconnected")
            self._hw_lbl.setStyleSheet("color:#c00;")
            self._conn_btn.clicked.connect(self._toggle_connect)
            hf.addRow(self._conn_btn, self._hw_lbl)
            left.addWidget(hw)

            # parameters
            pg = QtWidgets.QGroupBox("Parameters")
            pf = QtWidgets.QFormLayout(pg)

            self._pol_spin = NoWheelDoubleSpinBox()
            self._pol_spin.setRange(0.0, 360.0); self._pol_spin.setDecimals(1)
            self._pol_spin.setValue(0.0); self._pol_spin.setSuffix(" °")
            pf.addRow("Polarizer angle:", self._pol_spin)

            self._start_pan = NoWheelDoubleSpinBox()
            self._start_pan.setRange(-210.0, 210.0); self._start_pan.setDecimals(1)
            self._start_pan.setValue(-150.0); self._start_pan.setSuffix(" °")
            pf.addRow("Start pan:", self._start_pan)

            self._step_pan = NoWheelDoubleSpinBox()
            self._step_pan.setRange(1.0, 90.0); self._step_pan.setDecimals(1)
            self._step_pan.setValue(30.0); self._step_pan.setSuffix(" °")
            pf.addRow("Pan step:", self._step_pan)

            self._n_steps = NoWheelSpinBox()
            self._n_steps.setRange(2, 30); self._n_steps.setValue(11)
            pf.addRow("# Pan positions:", self._n_steps)

            self._tilt_0 = NoWheelDoubleSpinBox()
            self._tilt_0.setRange(-90.0, 90.0); self._tilt_0.setDecimals(1)
            self._tilt_0.setValue(0.0); self._tilt_0.setSuffix(" °")
            pf.addRow("Nominal tilt:", self._tilt_0)

            self._dither_offset_spin = NoWheelDoubleSpinBox()
            self._dither_offset_spin.setRange(0.1, 5.0)
            self._dither_offset_spin.setDecimals(2)
            self._dither_offset_spin.setValue(DEFAULT_DITHER_OFFSET_DEG)
            self._dither_offset_spin.setSuffix(" °")
            self._dither_offset_spin.setToolTip(
                "Grid step: pan ∈ {-N, 0, +N}° × tilt ∈ {-N, 0, +N}° → 9 captures"
            )
            pf.addRow("Dither offset ±N (°):", self._dither_offset_spin)

            self._exposure_spin = NoWheelDoubleSpinBox()
            self._exposure_spin.setRange(100.0, 1_000_000.0)
            self._exposure_spin.setDecimals(0)
            self._exposure_spin.setValue(DEFAULT_EXPOSURE_US)
            self._exposure_spin.setSuffix(" µs")
            pf.addRow("Exposure:", self._exposure_spin)

            left.addWidget(pg)

            # run controls
            rg = QtWidgets.QGroupBox("Run")
            rl = QtWidgets.QVBoxLayout(rg)

            self._start_btn = QtWidgets.QPushButton("Start Calibration")
            self._start_btn.setEnabled(False)
            self._start_btn.setStyleSheet("font-weight:bold;background:#0a7cff;color:white;")
            self._start_btn.clicked.connect(self._start_calibration)
            rl.addWidget(self._start_btn)

            self._status_lbl = QtWidgets.QLabel("Idle")
            self._status_lbl.setWordWrap(True)
            rl.addWidget(self._status_lbl)

            # click instruction banner
            self._click_banner = QtWidgets.QLabel("✏  Drag to draw a template rectangle")
            self._click_banner.setAlignment(align_center)
            self._click_banner.setStyleSheet(
                "background:#fff3cd;color:#856404;border:1px solid #ffc107;"
                "padding:4px;border-radius:3px;font-weight:bold;"
            )
            self._click_banner.setVisible(False)
            rl.addWidget(self._click_banner)

            # pan jog row
            jog_grp = QtWidgets.QGroupBox("Pan Jog")
            jog_lay = QtWidgets.QHBoxLayout(jog_grp)
            jog_lay.setContentsMargins(4, 4, 4, 4)
            self._jog_left_btn  = QtWidgets.QPushButton("◀◀")
            self._jog_left_btn.setToolTip("Pan left (hold to repeat)")
            self._jog_right_btn = QtWidgets.QPushButton("▶▶")
            self._jog_right_btn.setToolTip("Pan right (hold to repeat)")
            self._jog_step_spin = NoWheelDoubleSpinBox()
            self._jog_step_spin.setRange(0.01, 10.0)
            self._jog_step_spin.setDecimals(2)
            self._jog_step_spin.setValue(0.5)
            self._jog_step_spin.setSuffix(" °")
            self._jog_step_spin.setFixedWidth(72)
            jog_lay.addWidget(self._jog_left_btn)
            jog_lay.addWidget(self._jog_step_spin)
            jog_lay.addWidget(self._jog_right_btn)
            self._jog_left_btn.clicked.connect(lambda: self._jog_pan(-self._jog_step_spin.value()))
            self._jog_right_btn.clicked.connect(lambda: self._jog_pan(+self._jog_step_spin.value()))
            rl.addWidget(jog_grp)

            self._score_lbl = QtWidgets.QLabel("Match score: —")
            rl.addWidget(self._score_lbl)

            btn_row = QtWidgets.QHBoxLayout()
            self._accept_btn = QtWidgets.QPushButton("Accept ✓")
            self._retry_btn  = QtWidgets.QPushButton("Retry ↺")
            self._skip_btn   = QtWidgets.QPushButton("Skip ✗")
            for b in (self._accept_btn, self._retry_btn, self._skip_btn):
                b.setEnabled(False)
            self._accept_btn.setStyleSheet("background:#28a745;color:white;")
            self._retry_btn.setStyleSheet("background:#ffc107;")
            self._skip_btn.setStyleSheet("background:#dc3545;color:white;")
            self._accept_btn.clicked.connect(lambda: self._user_action("accept"))
            self._retry_btn.clicked.connect(lambda: self._user_action("retry"))
            self._skip_btn.clicked.connect(lambda: self._user_action("skip"))
            btn_row.addWidget(self._accept_btn)
            btn_row.addWidget(self._retry_btn)
            btn_row.addWidget(self._skip_btn)
            rl.addLayout(btn_row)

            self._abort_btn = QtWidgets.QPushButton("Abort")
            self._abort_btn.setEnabled(False)
            self._abort_btn.setStyleSheet("background:#dc3545;color:white;")
            self._abort_btn.clicked.connect(self._abort)
            rl.addWidget(self._abort_btn)

            left.addWidget(rg)

            # export
            eg = QtWidgets.QGroupBox("Export")
            el = QtWidgets.QHBoxLayout(eg)
            csv_btn = QtWidgets.QPushButton("Export CSV")
            h5_btn  = QtWidgets.QPushButton("Export HDF5")
            csv_btn.clicked.connect(self._export_csv)
            h5_btn.clicked.connect(self._export_h5)
            el.addWidget(csv_btn); el.addWidget(h5_btn)
            left.addWidget(eg)

            left.addStretch(1)

            # ── right panel ───────────────────────────────────────────────────
            right = QtWidgets.QVBoxLayout()
            right.setSpacing(6)
            root.addLayout(right, stretch=1)

            cam_grp = QtWidgets.QGroupBox("Camera  (drag to draw template rectangle at each pan position)")
            cl = QtWidgets.QVBoxLayout(cam_grp)
            self._cam = CameraView()
            self._cam.roi_selected.connect(self._on_roi_selected)
            cl.addWidget(self._cam, alignment=align_center)

            pt_row = QtWidgets.QHBoxLayout()
            self._pan_lbl  = QtWidgets.QLabel("Pan: —")
            self._tilt_lbl = QtWidgets.QLabel("Tilt: —")
            pt_row.addWidget(self._pan_lbl)
            pt_row.addWidget(self._tilt_lbl)
            cl.addLayout(pt_row)
            right.addWidget(cam_grp)

            # results table
            res_grp = QtWidgets.QGroupBox("Results  (one row per pan position)")
            res_l   = QtWidgets.QVBoxLayout(res_grp)
            self._table = QtWidgets.QTableWidget(0, 7)
            self._table.setHorizontalHeaderLabels([
                "Pan cmd (°)", "Pan actual (°)",
                "δPan (\")", "δTilt (\")",
                "R²x", "R²y", "Status",
            ])
            self._table.horizontalHeader().setStretchLastSection(True)
            no_edit = (
                QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
                if hasattr(QtWidgets.QAbstractItemView, "EditTrigger")
                else QtWidgets.QAbstractItemView.NoEditTriggers
            )
            self._table.setEditTriggers(no_edit)
            res_l.addWidget(self._table)

            self._fit_lbl = QtWidgets.QLabel("Sinusoid fit: —")
            self._fit_lbl.setStyleSheet("font-weight:bold;font-size:12px;")
            res_l.addWidget(self._fit_lbl)
            right.addWidget(res_grp, stretch=1)

            # log
            lg = QtWidgets.QGroupBox("Log")
            ll = QtWidgets.QVBoxLayout(lg)
            self._log = QtWidgets.QPlainTextEdit()
            self._log.setReadOnly(True)
            self._log.setMaximumHeight(100)
            self._log.setStyleSheet(
                "QPlainTextEdit{background:#fff;color:#202020;border:1px solid #b8b8b8;}"
            )
            ll.addWidget(self._log)
            right.addWidget(lg)

        # ── port / connect ─────────────────────────────────────────────────────
        def _refresh_ports(self):
            for cb in (self._moog_port, self._zaber_port):
                cb.clear()
            ports = [p.device for p in serial.tools.list_ports.comports()] if serial else []
            for p in ports:
                self._moog_port.addItem(p)
                self._zaber_port.addItem(p)
            if DEFAULT_MOOG_PORT in ports:
                self._moog_port.setCurrentText(DEFAULT_MOOG_PORT)
            if DEFAULT_ZABER_PORT in ports:
                self._zaber_port.setCurrentText(DEFAULT_ZABER_PORT)

        def _toggle_connect(self):
            if self.moog.status.connected:
                self._disconnect()
            else:
                self._connect()

        def _connect(self):
            self._log_msg("Connecting…")
            try:
                self.moog.open(self._moog_port.currentText())
                self.polarizer.open(self._zaber_port.currentText())
                self.camera.set_exposure(self._exposure_spin.value())
                self.camera.open()
                self._hw_lbl.setText("Connected"); self._hw_lbl.setStyleSheet("color:#080;")
                self._conn_btn.setText("Disconnect All")
                self._start_btn.setEnabled(True)
                self._preview_timer.start()
                self._log_msg("Hardware connected.")
            except Exception as exc:
                self._log_msg(f"Connection failed: {exc}")
                QtWidgets.QMessageBox.critical(self, "Connection Error", str(exc))

        def _disconnect(self):
            self._preview_timer.stop()
            for hw in (self.camera, self.polarizer, self.moog):
                try:
                    hw.close()
                except Exception:
                    pass
            self._hw_lbl.setText("Disconnected"); self._hw_lbl.setStyleSheet("color:#c00;")
            self._conn_btn.setText("Connect All")
            self._start_btn.setEnabled(False)
            self._log_msg("Hardware disconnected.")

        # ── camera grab (always lock-protected) ───────────────────────────────
        def _grab_frame(self) -> np.ndarray:
            with self._camera_lock:
                return self.camera.get_frame()

        # ── idle preview (timer-driven, only when not running) ─────────────────
        def _request_preview(self):
            if self._running or not self.camera.connected:
                return
            threading.Thread(target=self._fetch_preview, daemon=True).start()

        def _fetch_preview(self):
            try:
                frame = self._grab_frame()
                self._sig_frame.emit(frame)
            except Exception as exc:
                self._sig_log.emit(f"Preview: {exc}")

        # ── continuous live preview during Moog moves / settle ─────────────────
        def _start_live_preview(self):
            self._live_preview = True
            threading.Thread(target=self._live_preview_loop, daemon=True).start()

        def _stop_live_preview(self):
            self._live_preview = False

        def _live_preview_loop(self):
            while self._live_preview and self.camera.connected:
                try:
                    frame = self._grab_frame()
                    self._sig_frame.emit(frame)
                except Exception:
                    pass
                time.sleep(0.25)

        def _on_frame(self, frame: np.ndarray):
            self._last_frame = frame

            def overlay(pix, stride):
                painter = QtGui.QPainter(pix)
                # green box: user-drawn template rectangle
                if self._tmpl_rect is not None:
                    pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    c0, r0, c1, r1 = self._tmpl_rect
                    painter.drawRect(
                        int(c0 / stride), int(r0 / stride),
                        int((c1 - c0) / stride), int((r1 - r0) / stride),
                    )
                    # centre crosshair
                    cx = self._target_cx / stride
                    cy = self._target_cy / stride
                    painter.drawLine(int(cx - 12), int(cy), int(cx + 12), int(cy))
                    painter.drawLine(int(cx), int(cy - 12), int(cx), int(cy + 12))
                # yellow crosshair: theoretical (predicted) centroid
                if self._pred_cx is not None:
                    pen = QtGui.QPen(QtGui.QColor(255, 220, 0))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    px_ = self._pred_cx / stride
                    py_ = self._pred_cy / stride
                    painter.drawLine(int(px_ - 18), int(py_), int(px_ + 18), int(py_))
                    painter.drawLine(int(px_), int(py_ - 18), int(px_), int(py_ + 18))
                    painter.drawEllipse(int(px_ - 6), int(py_ - 6), 12, 12)
                # red crosshair: latest matched centroid
                if self._match_cx is not None:
                    pen = QtGui.QPen(QtGui.QColor(255, 60, 60))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    mx = self._match_cx / stride
                    my = self._match_cy / stride
                    painter.drawLine(int(mx - 18), int(my), int(mx + 18), int(my))
                    painter.drawLine(int(mx), int(my - 18), int(mx), int(my + 18))
                    painter.drawEllipse(int(mx - 6), int(my - 6), 12, 12)
                painter.end()

            self._cam.set_frame(frame, overlay_fn=overlay)

        def _on_status(self, st: PointingStatus):
            self._pan_lbl.setText(f"Pan: {st.pan_deg:+.2f}°")
            self._tilt_lbl.setText(f"Tilt: {st.tilt_deg:+.2f}°")

        def _jog_pan(self, delta_deg: float):
            if not self.moog.status.connected:
                return
            threading.Thread(target=self._jog_worker, args=(delta_deg,), daemon=True).start()

        def _jog_worker(self, delta_deg: float):
            try:
                st = self.moog.move_relative(delta_deg, 0.0)
                self._sig_status.emit(st)
            except Exception as exc:
                self._sig_log.emit(f"Pan jog failed: {exc}")

        def _on_log(self, msg: str):
            self._log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

        def _log_msg(self, msg: str):
            self._sig_log.emit(msg)

        # ── image click → lock target ──────────────────────────────────────────
        def _on_roi_selected(self, x0: float, y0: float, x1: float, y1: float):
            if self._last_frame is None:
                return
            frame = self._last_frame
            h, w  = frame.shape[:2]
            c0, r0 = int(np.clip(x0, 0, w - 1)), int(np.clip(y0, 0, h - 1))
            c1, r1 = int(np.clip(x1, 1, w)),     int(np.clip(y1, 1, h))
            if c1 <= c0 or r1 <= r0:
                return
            self._template  = frame[r0:r1, c0:c1].copy()
            self._tmpl_rect = (c0, r0, c1, r1)
            self._target_cx = (c0 + c1) / 2.0
            self._target_cy = (r0 + r1) / 2.0
            self._log_msg(
                f"Template selected: ({c0},{r0})→({c1},{r1})  size {c1-c0}×{r1-r0} px  "
                f"centre ({self._target_cx:.0f}, {self._target_cy:.0f})"
            )
            self._click_banner.setVisible(False)
            self._on_frame(frame)
            self._click_event.set()   # release worker if waiting

        def _on_await_click(self):
            """Called on main thread to show the click-target banner."""
            self._click_banner.setVisible(True)
            self._target_cx = None
            self._target_cy = None
            self._template  = None
            self._tmpl_rect = None
            self._match_cx  = None
            self._match_cy  = None
            self._pred_cx   = None
            self._pred_cy   = None
            # Allow skipping before a target is chosen
            self._skip_btn.setEnabled(True)
            self._accept_btn.setEnabled(False)
            self._retry_btn.setEnabled(False)

        # ── calibration start ──────────────────────────────────────────────────
        def _dither_grid(self) -> List[Tuple[float, float]]:
            """Return list of (dpan, dtilt) for a 3×3 grid: {-N,0,+N}²."""
            N = self._dither_offset_spin.value()
            offsets = [-N, 0.0, N]
            return [(dp, dt) for dp in offsets for dt in offsets]

        def _start_calibration(self):
            self._pan_points.clear()
            self._table.setRowCount(0)
            self._fit_lbl.setText("Sinusoid fit: —")
            self._running = True
            self._click_event.clear()
            self._step_event.clear()
            self._start_btn.setEnabled(False)
            self._abort_btn.setEnabled(True)
            threading.Thread(target=self._worker, daemon=True).start()

        # ── calibration worker ─────────────────────────────────────────────────
        def _worker(self):
            try:
                self._run()
            except Exception as exc:
                self._sig_log.emit(f"Worker error: {exc}")
            finally:
                self._running = False
                QtCore.QMetaObject.invokeMethod(
                    self, "_on_done",
                    QtCore.Qt.ConnectionType.QueuedConnection
                    if hasattr(QtCore.Qt, "ConnectionType")
                    else QtCore.Qt.QueuedConnection,
                )

        def _run(self):
            pol_angle  = self._pol_spin.value()
            start_pan  = self._start_pan.value()
            step_size  = self._step_pan.value()
            n_steps    = self._n_steps.value()
            tilt_0     = self._tilt_0.value()
            grid       = self._dither_grid()

            self._sig_log.emit(f"Setting polarizer → {pol_angle:.1f}°")
            try:
                self.polarizer.move_absolute(pol_angle)
            except Exception as exc:
                self._sig_log.emit(f"Polarizer warning: {exc}")

            pan_cmds = [start_pan + i * step_size for i in range(n_steps)]

            for idx, pan_cmd in enumerate(pan_cmds):
                if not self._running:
                    break

                pan_c, tilt_c = clamp_pointing(pan_cmd, tilt_0)
                self._sig_log.emit(
                    f"━━ Pan position {idx+1}/{n_steps}: pan={pan_c:+.1f}°, tilt={tilt_c:+.1f}°"
                )
                self._sig_status_update(f"Moving to pan {pan_c:+.1f}°…")

                self._start_live_preview()
                try:
                    st = self.moog.move_absolute(pan_c, tilt_c)
                    self._sig_status.emit(st)
                except Exception as exc:
                    self._stop_live_preview()
                    self._sig_log.emit(f"Move failed: {exc}"); continue

                time.sleep(MOOG_SETTLE_BEFORE_CAPTURE_SEC)
                self._stop_live_preview()

                # -- ask user to click target ---------------------------------
                self._click_event.clear()
                self._sig_await_click.emit()
                self._sig_log.emit("Waiting for user to click target…")
                self._sig_status_update(f"Pan {idx+1}/{n_steps}  ← Click target in image")

                self._start_live_preview()
                self._click_event.wait()        # blocks until _on_image_click fires or Skip
                self._stop_live_preview()
                self._step_event.clear()        # reset so dither's own wait works correctly

                if not self._running:
                    break
                # Skip pressed before target was clicked
                if self._template is None or self._step_action == "skip":
                    self._step_action = ""
                    self._sig_log.emit(f"Pan position {idx+1} skipped.")
                    skipped_pt = PanPoint(
                        pan_index=idx,
                        pan_cmd_deg=pan_c,
                        pan_actual_deg=pan_c,
                        tilt_0_deg=tilt_c,
                        skipped=True,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    self._pan_points.append(skipped_pt)
                    continue

                # Read actual pan after any jog adjustments made during click-wait.
                actual_st = self.moog.get_status()
                pan_0_actual  = actual_st.pan_deg
                tilt_0_actual = actual_st.tilt_deg

                pan_point = PanPoint(
                    pan_index=idx,
                    pan_cmd_deg=pan_c,
                    pan_actual_deg=pan_0_actual,
                    tilt_0_deg=tilt_0_actual,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                # -- dither loop (with retry support) -------------------------
                while self._running:
                    success = self._do_dither(pan_point, pan_0_actual, tilt_0_actual, grid)
                    if not success:
                        break

                    # return to the (possibly jogged) nominal position
                    try:
                        self.moog.move_absolute(pan_0_actual, tilt_0_actual)
                    except Exception:
                        pass

                    self._sig_dither_result.emit(pan_point)

                    # wait for Accept / Retry / Skip
                    self._step_event.clear()
                    self._step_event.wait()
                    action = self._step_action

                    if action == "accept":
                        self._pan_points.append(pan_point)
                        break
                    elif action == "skip":
                        pan_point.skipped = True
                        self._pan_points.append(pan_point)
                        break
                    else:   # retry — ask user to re-click target
                        pan_point.dither_steps.clear()
                        self._click_event.clear()
                        self._sig_await_click.emit()
                        self._sig_log.emit("Retry: click target again…")
                        self._start_live_preview()
                        self._click_event.wait()
                        self._stop_live_preview()
                        self._step_event.clear()

            self._sig_log.emit("All pan positions complete.")
            self._update_sinusoid_fit()

        def _do_dither(
            self,
            pan_point: PanPoint,
            pan_0: float,
            tilt_0: float,
            grid: List[Tuple[float, float]],
        ) -> bool:
            """
            Capture at each point of the 3×3 (dpan, dtilt) grid.
            The centre point (0, 0) is always included; the system is already
            there at the start, so the reference centroid is captured first.

            Returns True when done.
            """
            # reference capture at nominal (pan_0, tilt_0)
            self._sig_log.emit("Capturing reference frame at nominal position…")
            try:
                ref_frame = self._grab_frame()
                self._sig_frame.emit(ref_frame)
            except Exception as exc:
                self._sig_log.emit(f"Reference capture failed: {exc}")
                return False

            # Re-match the reference frame to get a sub-pixel centroid at (0,0).
            # Use a small search window here since we're at the lock position.
            rx, ry, _ = match_and_centroid(ref_frame, self._template)
            ref_cx, ref_cy = rx, ry
            pan_point.dither_steps.clear()
            min_score = 1.0

            for gi, (dpan, dtilt) in enumerate(grid):
                if not self._running:
                    break
                pan_c2, tilt_c2 = clamp_pointing(pan_0 + dpan, tilt_0 + dtilt)
                self._sig_status_update(
                    f"Grid {gi+1}/{len(grid)}: Δpan={dpan:+.1f}° Δtilt={dtilt:+.1f}°"
                )
                self._start_live_preview()
                try:
                    st = self.moog.move_absolute(pan_c2, tilt_c2)
                    self._sig_status.emit(st)
                except Exception as exc:
                    self._stop_live_preview()
                    self._sig_log.emit(f"Grid move failed: {exc}"); continue

                time.sleep(DITHER_SETTLE_S)
                self._stop_live_preview()

                try:
                    frame = self._grab_frame()
                except Exception as exc:
                    self._sig_log.emit(f"Capture failed: {exc}"); continue

                # Full-frame search: no dependency on position prediction or margins.
                cx, cy, score = match_and_centroid(frame, self._template)
                cur_cx, cur_cy = cx, cy
                self._match_cx, self._match_cy = cx, cy
                min_score = min(min_score, score)

                dpan_actual  = st.pan_deg  - pan_0
                dtilt_actual = st.tilt_deg - tilt_0

                # Theoretical centroid: where target should be if tilt error = 0.
                # pan right → scene moves left (cx↓); tilt up → scene moves down (cy↓).
                self._pred_cx = ref_cx - dpan_actual  * NOMINAL_PX_PER_DEG
                self._pred_cy = ref_cy - dtilt_actual * NOMINAL_PX_PER_DEG
                self._sig_frame.emit(frame)   # emit after all markers are set
                dcx = cx - ref_cx
                dcy = cy - ref_cy

                pan_point.dither_steps.append(DitherStep(
                    dpan_cmd_deg=dpan,
                    dtilt_cmd_deg=dtilt,
                    dpan_actual_deg=dpan_actual,
                    dtilt_actual_deg=dtilt_actual,
                    dcx_px=dcx,
                    dcy_px=dcy,
                    match_score=score,
                ))
                self._sig_log.emit(
                    f"  Δpan={dpan:+.1f}° Δtilt={dtilt:+.1f}°  "
                    f"Δcx={dcx:+.1f}px Δcy={dcy:+.1f}px  score={score:.3f}"
                )

            # 2-D linear fit
            if len(pan_point.dither_steps) >= 4:
                dpan_a  = np.array([s.dpan_actual_deg  for s in pan_point.dither_steps])
                dtilt_a = np.array([s.dtilt_actual_deg for s in pan_point.dither_steps])
                dcx_a   = np.array([s.dcx_px           for s in pan_point.dither_steps])
                dcy_a   = np.array([s.dcy_px           for s in pan_point.dither_steps])
                fit = fit_2d(dpan_a, dtilt_a, dcx_a, dcy_a)
                if fit:
                    pan_point.pan_error_arcsec  = fit["pan_error_deg"]  * 3600.0
                    pan_point.tilt_error_arcsec = fit["tilt_error_deg"] * 3600.0
                    pan_point.pan_px_per_deg    = fit["a_x"]
                    pan_point.tilt_px_per_deg   = fit["b_y"]
                    pan_point.fit_r2_x          = fit["r2_x"]
                    pan_point.fit_r2_y          = fit["r2_y"]
                    self._sig_log.emit(
                        f"  2-D fit: δpan={pan_point.pan_error_arcsec:+.2f}\"  "
                        f"δtilt={pan_point.tilt_error_arcsec:+.2f}\"  "
                        f"R²x={fit['r2_x']:.3f} R²y={fit['r2_y']:.3f}"
                    )

            self._sig_score_update(min_score)
            return True

        def _sig_status_update(self, msg: str):
            QtCore.QMetaObject.invokeMethod(
                self._status_lbl, "setText",
                QtCore.Qt.ConnectionType.QueuedConnection
                if hasattr(QtCore.Qt, "ConnectionType")
                else QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, msg),
            )

        def _sig_score_update(self, score: float):
            txt = f"Min match score: {score:.3f}"
            color = "#c80" if score < MATCH_WARN_SCORE else "#080"
            warn  = "  ⚠ LOW" if score < MATCH_WARN_SCORE else ""
            QtCore.QMetaObject.invokeMethod(
                self._score_lbl, "setText",
                QtCore.Qt.ConnectionType.QueuedConnection
                if hasattr(QtCore.Qt, "ConnectionType")
                else QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, txt + warn),
            )

        # ── step result / user action ──────────────────────────────────────────
        def _on_dither_result(self, pt: PanPoint):
            for b in (self._accept_btn, self._retry_btn, self._skip_btn):
                b.setEnabled(True)
            self._status_lbl.setText(
                f"Pan {pt.pan_index+1}/{self._n_steps.value()}  "
                f"pan={pt.pan_cmd_deg:+.1f}°  "
                f"δpan={pt.pan_error_arcsec:+.2f}\"  "
                f"δtilt={pt.tilt_error_arcsec:+.2f}\"  "
                f"R²x={pt.fit_r2_x:.3f} R²y={pt.fit_r2_y:.3f}"
            )

        def _user_action(self, action: str):
            for b in (self._accept_btn, self._retry_btn, self._skip_btn):
                b.setEnabled(False)
            self._click_banner.setVisible(False)
            self._step_action = action
            # If skip is pressed while still waiting for a target click,
            # release the click event so the worker unblocks.
            self._click_event.set()
            self._step_event.set()

        def _abort(self):
            self._running = False
            self._click_event.set()
            self._step_action = "skip"
            self._step_event.set()
            self._abort_btn.setEnabled(False)
            self._log_msg("Aborted.")

        @QtCore.Slot()
        def _on_done(self):
            self._start_btn.setEnabled(True)
            self._abort_btn.setEnabled(False)
            for b in (self._accept_btn, self._retry_btn, self._skip_btn):
                b.setEnabled(False)
            self._click_banner.setVisible(False)
            self._status_lbl.setText("Done")
            self._preview_timer.start()

        # ── sinusoid fit + table ───────────────────────────────────────────────
        def _update_sinusoid_fit(self):
            valid = [p for p in self._pan_points if not p.skipped]
            QtCore.QMetaObject.invokeMethod(
                self, "_populate_table",
                QtCore.Qt.ConnectionType.QueuedConnection
                if hasattr(QtCore.Qt, "ConnectionType")
                else QtCore.Qt.QueuedConnection,
            )
            if len(valid) < 3:
                self._sig_log.emit("Not enough valid points for sinusoid fit (need ≥ 3).")
                return
            pan_arr = np.array([p.pan_cmd_deg for p in valid])
            lines = []
            for label, arr in [
                ("δtilt", np.array([p.tilt_error_arcsec for p in valid])),
                ("δpan",  np.array([p.pan_error_arcsec  for p in valid])),
            ]:
                res = fit_sinusoid(pan_arr, arr)
                if res:
                    A, phi, C, _ = res
                    lines.append(f"{label}(pan) = {A:.3f}\" · sin(pan + {phi:.1f}°) + {C:+.3f}\"")
                    self._sig_log.emit(f"Sinusoid fit {label}: A={A:.3f}\" φ={phi:.1f}° C={C:+.3f}\"")
            txt = "   |   ".join(lines) if lines else "Fit failed"
            QtCore.QMetaObject.invokeMethod(
                self._fit_lbl, "setText",
                QtCore.Qt.ConnectionType.QueuedConnection
                if hasattr(QtCore.Qt, "ConnectionType")
                else QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, txt),
            )

        @QtCore.Slot()
        def _populate_table(self):
            self._table.setRowCount(0)
            for pt in self._pan_points:
                row = self._table.rowCount()
                self._table.insertRow(row)
                vals = [
                    f"{pt.pan_cmd_deg:+.1f}",
                    f"{pt.pan_actual_deg:+.2f}",
                    "—" if pt.skipped else f"{pt.pan_error_arcsec:+.2f}",
                    "—" if pt.skipped else f"{pt.tilt_error_arcsec:+.2f}",
                    "—" if pt.skipped else f"{pt.fit_r2_x:.3f}",
                    "—" if pt.skipped else f"{pt.fit_r2_y:.3f}",
                    "skipped" if pt.skipped else "ok",
                ]
                for col, v in enumerate(vals):
                    item = QtWidgets.QTableWidgetItem(v)
                    item.setTextAlignment(align_center)
                    if pt.skipped:
                        item.setForeground(QtGui.QColor("#999"))
                    self._table.setItem(row, col, item)

        # ── export ─────────────────────────────────────────────────────────────
        def _export_csv(self):
            if not self._pan_points:
                QtWidgets.QMessageBox.information(self, "No Data", "No data to export.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export CSV", "tilt_calibration.csv", "CSV (*.csv)"
            )
            if not path:
                return
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "pan_index", "pan_cmd_deg", "pan_actual_deg", "tilt_0_deg",
                    "pan_error_arcsec", "tilt_error_arcsec",
                    "pan_px_per_deg", "tilt_px_per_deg",
                    "fit_r2_x", "fit_r2_y", "skipped", "timestamp",
                    "n_grid_steps",
                    "grid_dpan_cmd", "grid_dtilt_cmd",
                    "grid_dpan_actual", "grid_dtilt_actual",
                    "grid_dcx_px", "grid_dcy_px", "grid_score",
                ])
                for pt in self._pan_points:
                    dither_fields = [
                        json.dumps([s.dpan_cmd_deg    for s in pt.dither_steps]),
                        json.dumps([s.dtilt_cmd_deg   for s in pt.dither_steps]),
                        json.dumps([s.dpan_actual_deg  for s in pt.dither_steps]),
                        json.dumps([s.dtilt_actual_deg for s in pt.dither_steps]),
                        json.dumps([s.dcx_px           for s in pt.dither_steps]),
                        json.dumps([s.dcy_px           for s in pt.dither_steps]),
                        json.dumps([s.match_score      for s in pt.dither_steps]),
                    ]
                    w.writerow([
                        pt.pan_index, pt.pan_cmd_deg, pt.pan_actual_deg, pt.tilt_0_deg,
                        pt.pan_error_arcsec, pt.tilt_error_arcsec,
                        pt.pan_px_per_deg, pt.tilt_px_per_deg,
                        pt.fit_r2_x, pt.fit_r2_y,
                        int(pt.skipped), pt.timestamp, len(pt.dither_steps),
                        *dither_fields,
                    ])
            self._log_msg(f"CSV exported: {path}")

        def _export_h5(self):
            if h5py is None:
                QtWidgets.QMessageBox.critical(self, "h5py missing", "Install h5py to export HDF5.")
                return
            if not self._pan_points:
                QtWidgets.QMessageBox.information(self, "No Data", "No data to export.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export HDF5", "tilt_calibration.h5", "HDF5 (*.h5)"
            )
            if not path:
                return
            with h5py.File(path, "w") as f:
                grp = f.create_group("tilt_calibration")
                grp.attrs["polarizer_angle_deg"] = self._pol_spin.value()
                grp.attrs["nominal_tilt_deg"]    = self._tilt_0.value()
                grp.attrs["pan_step_deg"]        = self._step_pan.value()
                grp.attrs["arcsec_per_pixel"]    = UV_ARCSEC_PER_PIXEL
                grp.attrs["created"]             = datetime.now(timezone.utc).isoformat()

                valid = [p for p in self._pan_points if not p.skipped]
                for attr in (
                    "pan_cmd_deg", "pan_actual_deg", "tilt_0_deg",
                    "pan_error_arcsec", "tilt_error_arcsec",
                    "pan_px_per_deg", "tilt_px_per_deg", "fit_r2_x", "fit_r2_y",
                ):
                    grp.create_dataset(attr, data=np.array([getattr(p, attr) for p in self._pan_points]))
                grp.create_dataset("skipped", data=np.array([int(p.skipped) for p in self._pan_points]))

                for i, pt in enumerate(self._pan_points):
                    pg = grp.create_group(f"pan_{i:02d}")
                    pg.attrs["pan_cmd_deg"]        = pt.pan_cmd_deg
                    pg.attrs["pan_error_arcsec"]   = pt.pan_error_arcsec
                    pg.attrs["tilt_error_arcsec"]  = pt.tilt_error_arcsec
                    if pt.dither_steps:
                        for key, vals in [
                            ("dpan_cmd_deg",    [s.dpan_cmd_deg    for s in pt.dither_steps]),
                            ("dtilt_cmd_deg",   [s.dtilt_cmd_deg   for s in pt.dither_steps]),
                            ("dpan_actual_deg", [s.dpan_actual_deg  for s in pt.dither_steps]),
                            ("dtilt_actual_deg",[s.dtilt_actual_deg for s in pt.dither_steps]),
                            ("dcx_px",          [s.dcx_px           for s in pt.dither_steps]),
                            ("dcy_px",          [s.dcy_px           for s in pt.dither_steps]),
                            ("match_score",     [s.match_score      for s in pt.dither_steps]),
                        ]:
                            pg.create_dataset(key, data=np.array(vals))

                if len(valid) >= 3:
                    pan_arr = np.array([p.pan_cmd_deg for p in valid])
                    fg = grp.create_group("sinusoid_fit")
                    fg.create_dataset("pan_deg_used", data=pan_arr)
                    for label, arr_vals in [
                        ("tilt", np.array([p.tilt_error_arcsec for p in valid])),
                        ("pan",  np.array([p.pan_error_arcsec  for p in valid])),
                    ]:
                        res = fit_sinusoid(pan_arr, arr_vals)
                        if res:
                            A, phi, C, fitted = res
                            sg = fg.create_group(label)
                            sg.attrs["amplitude_arcsec"] = A
                            sg.attrs["phase_deg"]        = phi
                            sg.attrs["offset_arcsec"]    = C
                            sg.create_dataset("error_arcsec",  data=arr_vals)
                            sg.create_dataset("fitted_arcsec", data=fitted)
            self._log_msg(f"HDF5 exported: {path}")

        def closeEvent(self, e):
            self._running = False
            self._click_event.set()
            self._step_event.set()
            self._preview_timer.stop()
            self._disconnect()
            super().closeEvent(e)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = CalibWindow()
    win.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()

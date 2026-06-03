# -*- coding: utf-8 -*-
"""
Qt GUI for manual ULTRASIP pointing calibration.

The GUI is intentionally hardware-optional. In simulation mode it can be run
without a Moog, Zaber stage, or UV camera attached. The same UI exposes hardware
hooks for later field use and a small HTTP API for remote control.
"""
#
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    from suncalc import get_position as suncalc_get_position
except ImportError:
    suncalc_get_position = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


PAN_MIN_DEG = -217.5
PAN_MAX_DEG = 217.5
TILT_MIN_DEG = -90.0
TILT_MAX_DEG = 90.0
MOOG_RESOLUTION_DEG = 0.01
MOOG_COMMAND_RESOLUTION_DEG = 0.1
MOOG_SETTLE_BEFORE_CAPTURE_SEC = 2.0
PREVIEW_MAX_SIDE_PX = 720
PREVIEW_CENTER_MAX_SIDE_PX = 900
PREVIEW_CENTER_DETECT_INTERVAL = 3
PREVIEW_AUTO_EXPOSURE_INTERVAL = 5
PREVIEW_FRAME_ERROR_BACKOFF_SEC = 3.0
UV_IMAGE_WIDTH_PX = 2848
UV_IMAGE_HEIGHT_PX = 2848
UV_FULL_FOV_DEG = 5.78
UV_ARCSEC_PER_PIXEL = 7.20
UV_DEG_PER_PIXEL = UV_ARCSEC_PER_PIXEL / 3600.0
DEFAULT_EXPOSURE_US = 1000.0
AUTO_EXPOSURE_TARGET_MEDIAN = 2600.0
AUTO_EXPOSURE_TEST_US = 100_000.0
AUTO_EXPOSURE_FIXED_NEAR_SUN_US = 800.0
AUTO_EXPOSURE_MIN_US = 100.0
AUTO_EXPOSURE_MAX_US = 1_000_000.0
AUTO_EXPOSURE_SATURATION_FRACTION = 0.97
UV_BIT_DEPTH = 12
DEFAULT_MOOG_PORT = "COM7"
DEFAULT_ZABER_PORT = "COM6"
CALIBRATION_DTILTS_DEG = (-0.5, 0.5, -1.0, 1.0, -1.5, 1.5)


def acquisition_group_name_for_dtilt(prefix: str, dtilt: float) -> str:
    sign = "up" if dtilt > 0 else "down" if dtilt < 0 else "center"
    magnitude = f"{abs(float(dtilt)):g}".replace(".", "p")
    return f"{prefix}_{sign}_{magnitude}"
AUTO_SCAN_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".ultrasip_auto_scan_settings.json")


def quantize_pointing(value: float) -> float:
    return round(float(value) / MOOG_RESOLUTION_DEG) * MOOG_RESOLUTION_DEG


def clamp_pointing(pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
    pan = min(max(quantize_pointing(pan_deg), PAN_MIN_DEG), PAN_MAX_DEG)
    tilt = min(max(quantize_pointing(tilt_deg), TILT_MIN_DEG), TILT_MAX_DEG)
    return pan, tilt


def moog_command_pointing(pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
    """Return the positions represented by the integer Moog command units."""
    pan, tilt = clamp_pointing(pan_deg, tilt_deg)
    return int(pan * 10) / 10.0, int(tilt * 10) / 10.0


def solar_position_deg(dt: datetime, latitude_deg: float, longitude_deg: float) -> Tuple[float, float]:
    # Return compass azimuth: north=0, east=90, south=180, west=270.
    # Do not use SunCalc here; Python SunCalc variants disagree on azimuth
    # convention, which can flip a south-facing Moog command by 180 degrees.
    local_dt = dt.astimezone()
    day = local_dt.timetuple().tm_yday
    hour = local_dt.hour + local_dt.minute / 60.0 + local_dt.second / 3600.0
    utc_offset_minutes = local_dt.utcoffset().total_seconds() / 60.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    time_offset = eq_time + 4.0 * longitude_deg - utc_offset_minutes
    true_solar_time = (hour * 60.0 + time_offset) % 1440.0
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)
    lat = math.radians(latitude_deg)
    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    cos_zenith = min(max(cos_zenith, -1.0), 1.0)
    zenith = math.acos(cos_zenith)
    altitude = 90.0 - math.degrees(zenith)
    az = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat) - math.tan(decl) * math.cos(lat),
    )
    azimuth = (math.degrees(az) + 180.0) % 360.0
    return azimuth, altitude


def azimuth_to_moog_pan(azimuth_deg: float) -> float:
    """Convert compass azimuth to Moog/SunCalc pan: south=0, west positive."""
    return (float(azimuth_deg) - 180.0 + 180.0) % 360.0 - 180.0


def moog_pan_to_azimuth(pan_deg: float) -> float:
    return (float(pan_deg) + 180.0) % 360.0


def import_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets, "PySide6"
    except ImportError:
        pass

    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets, "PyQt6"
    except ImportError:
        pass

    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets, "PyQt5"
    except ImportError:
        pass

    raise ImportError(
        "No Qt binding found. Install one of: PySide6, PyQt6, or PyQt5."
    )


@dataclass
class PointingStatus:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    connected: bool = False
    move_complete: bool = True
    raw: Optional[object] = None


def write_pointing_attrs(attrs, status: PointingStatus, target_pan_deg: float, target_tilt_deg: float):
    attrs["Moog Target Pan [deg]"] = float(target_pan_deg)
    attrs["Moog Target Tilt [deg]"] = float(target_tilt_deg)
    attrs["Moog Actual Pan [deg]"] = float(status.pan_deg)
    attrs["Moog Actual Tilt [deg]"] = float(status.tilt_deg)
    attrs["Moog Pan Error [deg]"] = float(status.pan_deg - target_pan_deg)
    attrs["Moog Tilt Error [deg]"] = float(status.tilt_deg - target_tilt_deg)
    attrs["Moog Move Complete"] = int(status.move_complete)
    if status.raw is not None and hasattr(status.raw, "raw_bytes"):
        attrs["Moog Status Raw Bytes"] = status.raw.raw_bytes
        attrs["Moog EXEC Bit"] = int(status.raw.gen_status.executing)
        attrs["Moog Moving Bits"] = json.dumps({
            "cw": int(status.raw.gen_status.moving_cw),
            "ccw": int(status.raw.gen_status.moving_ccw),
            "up": int(status.raw.gen_status.moving_up),
            "down": int(status.raw.gen_status.moving_down),
        })
        attrs["Moog Soft Limit Bits"] = json.dumps({
            "pan_cw": int(status.raw.pan_status.cw_soft_lim),
            "pan_ccw": int(status.raw.pan_status.ccw_soft_lim),
            "tilt_up": int(status.raw.tilt_status.up_soft_lim),
            "tilt_down": int(status.raw.tilt_status.down_soft_lim),
        })
        attrs["Moog Hard Limit Bits"] = json.dumps({
            "pan_cw": int(status.raw.pan_status.cw_hard_lim),
            "pan_ccw": int(status.raw.pan_status.ccw_hard_lim),
            "tilt_up": int(status.raw.tilt_status.up_hard_lim),
            "tilt_down": int(status.raw.tilt_status.down_hard_lim),
        })
    else:
        attrs["Moog Status Raw Bytes"] = []
        attrs["Moog EXEC Bit"] = 0
        attrs["Moog Moving Bits"] = json.dumps({"cw": 0, "ccw": 0, "up": 0, "down": 0})
        attrs["Moog Soft Limit Bits"] = json.dumps({"pan_cw": 0, "pan_ccw": 0, "tilt_up": 0, "tilt_down": 0})
        attrs["Moog Hard Limit Bits"] = json.dumps({"pan_cw": 0, "pan_ccw": 0, "tilt_up": 0, "tilt_down": 0})


class SimMoogController:
    def __init__(self):
        self.status = PointingStatus()

    def open(self, port: str):
        self.status.connected = True
        return self.status

    def close(self):
        self.home()
        self.status.connected = False

    def home(self):
        current_tilt = self.status.tilt_deg
        self.move_absolute(0.0, current_tilt)
        return self.move_absolute(0.0, 0.0)

    def move_absolute(self, pan_deg: float, tilt_deg: float):
        pan_deg, tilt_deg = moog_command_pointing(pan_deg, tilt_deg)
        self.status.pan_deg = pan_deg
        self.status.tilt_deg = tilt_deg
        self.status.move_complete = True
        return self.status

    def move_relative(self, dpan_deg: float, dtilt_deg: float):
        return self.move_absolute(
            self.status.pan_deg + float(dpan_deg),
            self.status.tilt_deg + float(dtilt_deg),
        )

    def get_status(self):
        return self.status


class RealMoogController:
    def __init__(self):
        import moog_functions as mf

        self.mf = mf
        self.serial_port = None
        self.status = PointingStatus()

    def open(self, port: str):
        if serial is None:
            raise RuntimeError("pyserial is not installed.")

        try:
            self.serial_port = serial.Serial()
            self.serial_port.baudrate = 9600
            self.serial_port.port = port
            self.serial_port.timeout = 1.0
            self.serial_port.write_timeout = 1.0
            self.serial_port.open()
            self.mf.init_autobaud(self.serial_port)
            return self.get_status()
        except Exception:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            self.status.connected = False
            raise

    def close(self):
        self.home()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.status.connected = False

    def home(self):
        if not self.serial_port or not self.serial_port.is_open:
            return self.status

        current = self.get_status()
        self.move_absolute(0.0, current.tilt_deg)
        return self.move_absolute(0.0, 0.0)

    def move_absolute(self, pan_deg: float, tilt_deg: float):
        if not self.serial_port or not self.serial_port.is_open:
            raise RuntimeError("Moog serial port is not open.")

        pan_deg, tilt_deg = moog_command_pointing(pan_deg, tilt_deg)
        raw = self.mf.move_to_coord_and_wait(
            self.serial_port,
            int(pan_deg * 10),
            int(tilt_deg * 10),
            verbose=False,
        )
        self.status = PointingStatus(
            pan_deg=float(raw.pan_coord),
            tilt_deg=float(raw.tilt_coord),
            connected=True,
            move_complete=bool(self.mf.is_move_complete(raw)),
            raw=raw,
        )
        return self.status

    def move_relative(self, dpan_deg: float, dtilt_deg: float):
        current = self.get_status()
        return self.move_absolute(current.pan_deg + dpan_deg, current.tilt_deg + dtilt_deg)

    def get_status(self):
        if not self.serial_port or not self.serial_port.is_open:
            self.status.connected = False
            return self.status

        raw = self.mf.get_status_jog(self.serial_port, verbose=False)
        self.status = PointingStatus(
            pan_deg=float(raw.pan_coord),
            tilt_deg=float(raw.tilt_coord),
            connected=True,
            move_complete=bool(self.mf.is_move_complete(raw)),
            raw=raw,
        )
        return self.status


class SimPolarizerController:
    def __init__(self):
        self.connected = False
        self.angle_deg = 0.0

    def open(self, port: str):
        self.connected = True

    def close(self):
        self.connected = False

    def move_absolute(self, angle_deg: float):
        self.angle_deg = float(angle_deg) % 360.0


class RealPolarizerController:
    def __init__(self):
        self.connection = None
        self.axis = None
        self.connected = False
        self.angle_deg = 0.0

    def open(self, port: str):
        from zaber_motion import Units
        from zaber_motion.ascii import Connection

        try:
            self.units = Units
            self.connection = Connection.open_serial_port(port)
            device = self.connection.detect_devices()[0]
            self.axis = device.get_axis(1)
            if not self.axis.is_homed():
                self.axis.home()
            self.connected = True
        except Exception:
            if self.connection:
                self.connection.close()
            self.connection = None
            self.axis = None
            self.connected = False
            raise

    def close(self):
        if self.connection:
            self.connection.close()
        self.connected = False

    def move_absolute(self, angle_deg: float):
        if not self.axis:
            raise RuntimeError("Polarizer port is not open.")
        self.axis.move_absolute(float(angle_deg), self.units.ANGLE_DEGREES)
        self.axis.wait_until_idle()
        self.angle_deg = float(angle_deg)


class SimCameraController:
    def __init__(self, exposure_us: float = DEFAULT_EXPOSURE_US):
        self.connected = False
        self.frame_index = 0
        self.exposure_us = float(exposure_us)
        self.width = UV_IMAGE_WIDTH_PX
        self.height = UV_IMAGE_HEIGHT_PX
        self.sun_x = self.width * 0.5
        self.sun_y = self.height * 0.5
        self.pixels_per_degree = 1.0 / UV_DEG_PER_PIXEL
        self.latitude_deg = 32.23134
        self.longitude_deg = -110.94712
        self.pan_mount_offset_deg = float(np.random.default_rng().uniform(-10.0, 10.0))

    def open(self):
        self.connected = True

    def close(self):
        self.connected = False

    def set_exposure(self, exposure_us: float):
        self.exposure_us = float(np.clip(exposure_us, AUTO_EXPOSURE_MIN_US, AUTO_EXPOSURE_MAX_US))

    def get_exposure(self):
        return self.exposure_us

    def set_site(self, latitude_deg: float, longitude_deg: float):
        self.latitude_deg = float(latitude_deg)
        self.longitude_deg = float(longitude_deg)

    def reset_mount_offset(self):
        self.pan_mount_offset_deg = float(np.random.default_rng().uniform(-10.0, 10.0))

    @staticmethod
    def azimuth_error_deg(target_azimuth_deg: float, pointing_azimuth_deg: float):
        return (float(target_azimuth_deg) - float(pointing_azimuth_deg) + 180.0) % 360.0 - 180.0

    def get_frame(self, pan_deg: float = 0.0, tilt_deg: float = 0.0):
        self.frame_index += 1
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        drift_x = 22.0 * math.sin(self.frame_index / 28.0)
        drift_y = 14.0 * math.cos(self.frame_index / 31.0)
        sun_azimuth, sun_altitude = solar_position_deg(datetime.now(), self.latitude_deg, self.longitude_deg)
        pointing_azimuth = moog_pan_to_azimuth(float(pan_deg) + self.pan_mount_offset_deg)
        pan_error = self.azimuth_error_deg(sun_azimuth, pointing_azimuth)
        tilt_error = sun_altitude - float(tilt_deg)
        cx = self.sun_x + pan_error * self.pixels_per_degree + drift_x
        cy = self.sun_y - tilt_error * self.pixels_per_degree + drift_y

        sky = 250.0 + 20.0 * np.sin(xx / 180.0) + 10.0 * np.cos(yy / 130.0)
        disk = 3500.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 35.0 ** 2))
        halo = 900.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 105.0 ** 2))
        noise = np.random.default_rng(self.frame_index).normal(0.0, 12.0, sky.shape)
        exposure_scale = self.exposure_us / DEFAULT_EXPOSURE_US
        frame = np.clip((sky + disk + halo + noise) * exposure_scale, 0, 4095).astype(np.uint16)
        return frame


class RealVmbCameraController:
    def __init__(self, exposure_us: float = 1000.0):
        self.exposure_us = exposure_us
        self.connected = False
        self.vmb_cm = None
        self.cam_cm = None
        self.camera = None
        self.frame_count = 0
        self.failed_frame_count = 0
        self.last_frame_duration_ms = 0.0
        self.last_frame_error = ""

    def open(self):
        import uv_cam_functions as uv

        try:
            self.uv = uv
            self.vmb_cm = uv.VmbSystem.get_instance()
            self.vmb_cm.__enter__()
            self.camera = uv.get_camera(None)
            self.cam_cm = self.camera
            self.cam_cm.__enter__()
            uv.setup_camera(self.camera, self.exposure_us)
            self.connected = True
            self.frame_count = 0
            self.failed_frame_count = 0
            self.last_frame_error = ""
        except Exception:
            self.close()
            raise

    def close(self):
        if self.cam_cm:
            self.cam_cm.__exit__(None, None, None)
            self.cam_cm = None
        if self.vmb_cm:
            self.vmb_cm.__exit__(None, None, None)
            self.vmb_cm = None
        self.camera = None
        self.connected = False

    def set_exposure(self, exposure_us: float):
        self.exposure_us = float(np.clip(exposure_us, AUTO_EXPOSURE_MIN_US, AUTO_EXPOSURE_MAX_US))
        if self.connected and self.camera is not None:
            self.uv.setup_camera(self.camera, self.exposure_us)

    def get_exposure(self):
        return self.exposure_us

    def _feature_value(self, name):
        if self.camera is None:
            return None
        feature = getattr(self.camera, name, None)
        if feature is None:
            return None
        try:
            return feature.get()
        except Exception:
            return None

    def diagnostics(self):
        values = {
            "connected": self.connected,
            "camera_id": None,
            "exposure_us": self.exposure_us,
            "frames_ok": self.frame_count,
            "frames_failed": self.failed_frame_count,
            "last_frame_ms": round(self.last_frame_duration_ms, 1),
        }
        if self.camera is not None:
            try:
                values["camera_id"] = self.camera.get_id()
            except Exception:
                values["camera_id"] = None
            for name in (
                "ExposureTime",
                "Width",
                "Height",
                "PixelFormat",
                "AcquisitionMode",
                "TriggerMode",
                "DeviceLinkThroughputLimit",
                "GevSCPSPacketSize",
            ):
                value = self._feature_value(name)
                if value is not None:
                    values[name] = value
        if self.last_frame_error:
            values["last_error"] = self.last_frame_error
        return values

    def get_frame(self, pan_deg: float = 0.0, tilt_deg: float = 0.0):
        if not self.connected or self.camera is None:
            raise RuntimeError("Camera is not open.")
        start = time.time()
        try:
            frame = self.camera.get_frame()
        except Exception as exc:
            self.failed_frame_count += 1
            self.last_frame_duration_ms = (time.time() - start) * 1000.0
            self.last_frame_error = str(exc)
            raise RuntimeError(
                "Camera get_frame timed out or failed "
                f"after {self.last_frame_duration_ms:.1f} ms; "
                f"diagnostics={self.diagnostics()}; original={exc}"
            ) from exc
        self.frame_count += 1
        self.last_frame_duration_ms = (time.time() - start) * 1000.0
        self.last_frame_error = ""
        data = np.frombuffer(frame.get_buffer(), dtype=np.uint16)
        height = int(getattr(frame, "get_height", lambda: 0)())
        width = int(getattr(frame, "get_width", lambda: 0)())
        if width > 0 and height > 0 and data.size == width * height:
            return data.reshape(height, width)
        side = int(math.sqrt(data.size))
        return data[:side * side].reshape(side, side)


def detect_sun_center(frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], dict]:
    if frame is None or frame.size == 0:
        return None, {"reason": "empty frame"}

    image = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.8])
    if hi <= lo:
        return None, {"reason": "low contrast"}

    norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    threshold = max(float(np.percentile(norm, 99.3)), 0.72)
    mask = (norm >= threshold).astype(np.uint8)

    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None, {"reason": "no bright component"}

        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        area = int(stats[best, cv2.CC_STAT_AREA])
        if area < 10:
            return None, {"reason": "component too small", "area": area}

        component = labels == best
        weights = np.maximum(image - np.percentile(image[component], 20.0), 0.0) * component
        total = float(weights.sum())
        if total <= 0:
            cx, cy = centroids[best]
        else:
            yy, xx = np.indices(image.shape)
            cx = float((xx * weights).sum() / total)
            cy = float((yy * weights).sum() / total)
        return (cx, cy), {"threshold": threshold, "area": area}

    yy, xx = np.indices(image.shape)
    weights = norm * mask
    total = float(weights.sum())
    if total <= 0:
        return None, {"reason": "no bright pixels"}
    return (float((xx * weights).sum() / total), float((yy * weights).sum() / total)), {"threshold": threshold}


def frame_to_qimage(frame: np.ndarray, QtGui):
    image = np.asarray(frame)
    if image.ndim == 2:
        lo, hi = np.percentile(image, [1.0, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        view = np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        h, w = view.shape
        qimage_format = getattr(QtGui.QImage, "Format", QtGui.QImage)
        fmt = getattr(qimage_format, "Format_Grayscale8", getattr(qimage_format, "Format_Indexed8"))
        return QtGui.QImage(view.data, w, h, view.strides[0], fmt).copy()
    if image.ndim == 3:
        h, w, _ = image.shape
        rgb = np.ascontiguousarray(image[:, :, :3].astype(np.uint8))
        qimage_format = getattr(QtGui.QImage, "Format", QtGui.QImage)
        return QtGui.QImage(rgb.data, w, h, rgb.strides[0], qimage_format.Format_RGB888).copy()
    raise ValueError("Unsupported frame shape.")


class RemoteControlServer:
    def __init__(self, window, host="127.0.0.1", port=8765):
        self.window = window
        self.host = host
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        window = self.window

        class Handler(BaseHTTPRequestHandler):
            def _send(self, payload, code=200):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                try:
                    if parsed.path == "/status":
                        self._send(window.remote_status())
                    elif parsed.path == "/move":
                        dpan = float(query.get("dpan", ["0"])[0])
                        dtilt = float(query.get("dtilt", ["0"])[0])
                        self._send(window.remote_move(dpan, dtilt))
                    elif parsed.path == "/goto":
                        pan = float(query["pan"][0])
                        tilt = float(query["tilt"][0])
                        self._send(window.remote_goto(pan, tilt))
                    elif parsed.path == "/center":
                        self._send(window.remote_center())
                    elif parsed.path == "/polarizer":
                        angle = float(query["angle"][0])
                        self._send(window.remote_polarizer(angle))
                    elif parsed.path == "/exposure":
                        auto = query.get("auto", [None])[0]
                        exposure = query.get("us", [None])[0]
                        self._send(window.remote_exposure(auto, exposure))
                    else:
                        self._send({"error": "unknown endpoint"}, 404)
                except Exception as exc:
                    self._send({"error": str(exc)}, 500)

            def log_message(self, fmt, *args):
                return

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def build_app_classes(QtCore, QtGui, QtWidgets):
    align_center = getattr(getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt), "AlignCenter")
    keep_aspect = getattr(getattr(QtCore.Qt, "AspectRatioMode", QtCore.Qt), "KeepAspectRatio")
    smooth_transform = getattr(getattr(QtCore.Qt, "TransformationMode", QtCore.Qt), "SmoothTransformation")
    expanding_policy = getattr(getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy), "Expanding")
    no_frame = getattr(getattr(QtWidgets.QFrame, "Shape", QtWidgets.QFrame), "NoFrame")

    class ToggleSwitch(QtWidgets.QCheckBox):
        def __init__(self, label=""):
            super().__init__(label)
            pointing_cursor = getattr(getattr(QtCore.Qt, "CursorShape", QtCore.Qt), "PointingHandCursor")
            self.setCursor(QtGui.QCursor(pointing_cursor))
            self.setMinimumHeight(28)

        def sizeHint(self):
            metrics = self.fontMetrics()
            if self.text() and hasattr(metrics, "horizontalAdvance"):
                text_width = metrics.horizontalAdvance(self.text())
            else:
                text_width = metrics.width(self.text()) if self.text() else 0
            return QtCore.QSize(56 + text_width, 28)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            antialiasing = getattr(getattr(QtGui.QPainter, "RenderHint", QtGui.QPainter), "Antialiasing")
            painter.setRenderHint(antialiasing)
            rect = QtCore.QRectF(0, 3, 46, 22)
            no_pen = getattr(getattr(QtCore.Qt, "PenStyle", QtCore.Qt), "NoPen")
            painter.setPen(no_pen)
            painter.setBrush(QtGui.QColor("#0a7cff" if self.isChecked() else "#d8d8d8"))
            painter.drawRoundedRect(rect, 14, 14)
            knob_x = 25 if self.isChecked() else 3
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.drawEllipse(QtCore.QRectF(knob_x, 5, 18, 18))
            if self.text():
                painter.setPen(QtGui.QColor("#222222"))
                painter.drawText(QtCore.QRectF(56, 0, self.width() - 56, self.height()), align_center, self.text())
            painter.end()

    def make_switch(label=""):
        return ToggleSwitch(label)

    def style_log_box(log):
        log.setReadOnly(True)
        log.setMinimumHeight(100)
        log.setStyleSheet(
            "QPlainTextEdit { background: #ffffff; color: #202020; border: 1px solid #b8b8b8; }"
        )

    class CameraView(QtWidgets.QLabel):
        def __init__(self):
            super().__init__()
            self.display_side = 480
            self.setFixedSize(self.display_side, self.display_side)
            if hasattr(self, "setScaledContents"):
                self.setScaledContents(False)
            self.setAlignment(align_center)
            self.setText("Camera closed")
            self.setStyleSheet("background: #111; color: #ddd; border: 1px solid #444;")
            self.last_pixmap = None

        def hasHeightForWidth(self):
            return True

        def heightForWidth(self, width):
            return width

        def sizeHint(self):
            return QtCore.QSize(self.display_side, self.display_side)

        def set_display_side(self, side):
            self.display_side = max(120, int(side))
            self.setFixedSize(self.display_side, self.display_side)
            self._rescale()

        def set_frame(self, frame, center=None):
            image = np.asarray(frame)
            stride = max(1, int(math.ceil(max(image.shape[:2]) / float(PREVIEW_MAX_SIDE_PX))))
            preview = image[::stride, ::stride]
            qimage = frame_to_qimage(preview, QtGui)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            if center:
                painter = QtGui.QPainter(pixmap)
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                cx, cy = center
                cx = float(cx) / stride
                cy = float(cy) / stride
                painter.drawLine(int(cx - 12), int(cy), int(cx + 12), int(cy))
                painter.drawLine(int(cx), int(cy - 12), int(cx), int(cy + 12))
                painter.end()
            self.last_pixmap = pixmap
            self._rescale()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._rescale()

        def _rescale(self):
            if not self.last_pixmap:
                return
            side = max(1, min(self.width(), self.height()))
            scaled = self.last_pixmap.scaled(
                QtCore.QSize(side, side),
                keep_aspect,
                smooth_transform,
            )
            self.setPixmap(scaled)

    class AutoScanPreviewTile(QtWidgets.QWidget):
        def __init__(self, title):
            super().__init__()
            self.title = title
            self.image_label = QtWidgets.QLabel(title)
            self.image_label.setFixedSize(150, 150)
            self.image_label.setAlignment(align_center)
            self.image_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #444;")
            self.text_label = QtWidgets.QLabel(title)
            self.text_label.setAlignment(align_center)
            self.text_label.setWordWrap(True)
            self.text_label.setStyleSheet("font-size: 9px;")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)
            layout.addWidget(self.image_label)
            layout.addWidget(self.text_label)
            self.last_pixmap = None

        def set_frame(self, frame, text=None):
            if text is not None:
                self.title = text
            image = np.asarray(frame)
            if image.ndim == 2:
                max_side = max(image.shape)
                step = max(1, int(math.ceil(max_side / 300.0)))
                image = image[::step, ::step]
            qimage = frame_to_qimage(image, QtGui)
            self.last_pixmap = QtGui.QPixmap.fromImage(qimage)
            if text is not None:
                self.text_label.setText(text)
            self._rescale()

        def clear_frame(self):
            self.last_pixmap = None
            self.image_label.clear()
            self.image_label.setText(self.title)
            self.text_label.setText(self.title)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._rescale()

        def _rescale(self):
            if not self.last_pixmap:
                return
            scaled = self.last_pixmap.scaled(self.image_label.size(), keep_aspect, smooth_transform)
            self.image_label.setPixmap(scaled)

    class DigitAxisWidget(QtWidgets.QWidget):
        def __init__(self, name, move_callback):
            super().__init__()
            self.name = name
            self.move_callback = move_callback
            self.steps = [100.0, 10.0, 1.0, 0.1, 0.01]
            self.digit_labels = []
            self.sign_label = QtWidgets.QLabel("+")
            self.sign_label.setAlignment(align_center)
            self.sign_label.setStyleSheet(
                "background: #050505; color: #39ff88; font: 700 23px 'DS-Digital', 'Courier New', monospace;"
            )

            layout = QtWidgets.QGridLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setHorizontalSpacing(2)
            layout.setVerticalSpacing(2)
            title = QtWidgets.QLabel(name)
            title.setAlignment(align_center)
            layout.addWidget(title, 0, 0, 1, 7)

            layout.addWidget(QtWidgets.QLabel(""), 1, 0)
            layout.addWidget(self.sign_label, 2, 0)
            layout.addWidget(QtWidgets.QLabel(""), 3, 0)

            digit_columns = [1, 2, 3, 5, 6]
            for col, step in zip(digit_columns, self.steps):
                up = QtWidgets.QPushButton("▲")
                down = QtWidgets.QPushButton("▼")
                up.setFixedHeight(18)
                down.setFixedHeight(18)
                up.clicked.connect(lambda checked=False, s=step: self.move_callback(s))
                down.clicked.connect(lambda checked=False, s=step: self.move_callback(-s))
                digit = QtWidgets.QLabel("0")
                digit.setAlignment(align_center)
                digit.setMinimumWidth(23)
                digit.setStyleSheet(
                    "background: #050505; color: #39ff88; font: 700 24px 'DS-Digital', 'Courier New', monospace;"
                )
                self.digit_labels.append(digit)
                layout.addWidget(up, 1, col)
                layout.addWidget(digit, 2, col)
                layout.addWidget(down, 3, col)

            dot = QtWidgets.QLabel(".")
            dot.setAlignment(align_center)
            dot.setStyleSheet(
                "background: #050505; color: #39ff88; font: 700 24px 'DS-Digital', 'Courier New', monospace;"
            )
            layout.addWidget(dot, 2, 4)

        def set_value(self, value):
            value = quantize_pointing(value)
            self.sign_label.setText("-" if value < 0 else "+")
            text = f"{abs(value):06.2f}".replace(".", "")
            for label, char in zip(self.digit_labels, text):
                label.setText(char)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ULTRASIP Auto Measurement")
            self.resize(1000, 680)
            self.setStyleSheet(
                "QWidget { font-size: 11px; }"
                "QGroupBox { font-weight: 600; margin-top: 7px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }"
                "QPushButton { min-height: 22px; padding: 2px 7px; }"
                "QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { min-height: 20px; }"
            )

            self.simulation = True
            self.moog = SimMoogController()
            self.polarizer = SimPolarizerController()
            self.camera = SimCameraController()
            self.last_frame = None
            self.last_center = None
            self.last_center_result = None
            self.last_message = ""
            self.current_output_text = ""
            self.pan_offset = 0.0
            self.tilt_offset = 0.0
            self.auto_exposure_mode = "median"
            self.auto_exposure_max_us = AUTO_EXPOSURE_MAX_US
            self.calibration_complete = False
            self.completed_sun_targets = set()
            self.queued_sun_trigger = None
            self.last_sun_altitude = None
            self.sun_trigger_enabled = True
            self.auto_monitoring = False
            self.auto_job_running = False
            self.auto_stop_requested = False
            self.stop_acquisition_event = threading.Event()
            self.remote_server = None
            self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ultrasip-gui")
            self.motion_lock = threading.RLock()
            self.camera_lock = threading.RLock()
            self.result_queue = queue.Queue()
            self.camera_pending = False
            self.status_pending = False
            self.resume_preview_after_scan = False
            self.preview_frame_counter = 0
            self.frame_error_count = 0
            self.camera_retry_until = 0.0
            self.current_status = self.moog.get_status()

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(700)
            self.timer.timeout.connect(self.request_frame_update)

            self.result_timer = QtCore.QTimer(self)
            self.result_timer.setInterval(50)
            self.result_timer.timeout.connect(self.process_worker_results)
            self.result_timer.start()

            self.auto_monitor_timer = QtCore.QTimer(self)
            self.auto_monitor_timer.timeout.connect(self.check_sun_trigger)

            self._build_ui()
            self.refresh_ports()
            self.apply_status()

        def _build_ui(self):
            self.pages = QtWidgets.QStackedWidget()
            self.setCentralWidget(self.pages)

            auto_page = QtWidgets.QWidget()
            auto_layout = QtWidgets.QVBoxLayout(auto_page)
            auto_layout.setContentsMargins(9, 9, 9, 9)
            auto_layout.setSpacing(6)

            title = QtWidgets.QLabel("ULTRASIP Auto Measurement")
            title.setStyleSheet("font: 700 18px sans-serif;")
            auto_layout.addWidget(title)

            calibration_group = QtWidgets.QGroupBox("Calibration")
            calibration_layout = QtWidgets.QGridLayout(calibration_group)
            auto_layout.addWidget(calibration_group)

            self.calibration_status_label = QtWidgets.QLabel("Not complete")
            self.auto_pan_offset_label = QtWidgets.QLabel("")
            self.auto_tilt_offset_label = QtWidgets.QLabel("")
            self.auto_centered_pan_label = QtWidgets.QLabel("")
            self.calibration_btn = QtWidgets.QPushButton("Calibration")
            self.calibration_btn.setMinimumHeight(28)
            self.calibration_btn.clicked.connect(self.show_calibration_page)
            calibration_layout.addWidget(QtWidgets.QLabel("Status"), 0, 0)
            calibration_layout.addWidget(self.calibration_status_label, 0, 1)
            calibration_layout.addWidget(QtWidgets.QLabel("Pan Offset [deg]"), 1, 0)
            calibration_layout.addWidget(self.auto_pan_offset_label, 1, 1)
            calibration_layout.addWidget(QtWidgets.QLabel("Tilt Offset [deg]"), 2, 0)
            calibration_layout.addWidget(self.auto_tilt_offset_label, 2, 1)
            calibration_layout.addWidget(QtWidgets.QLabel("Centered Pan [deg]"), 3, 0)
            calibration_layout.addWidget(self.auto_centered_pan_label, 3, 1)
            calibration_layout.addWidget(self.calibration_btn, 4, 0, 1, 2)

            auto_connection_group = QtWidgets.QGroupBox("Device Control")
            auto_connection_layout = QtWidgets.QGridLayout(auto_connection_group)
            auto_layout.addWidget(auto_connection_group)
            self.auto_sim_check = make_switch("Simulation")
            self.auto_sim_check.setChecked(True)
            self.auto_sim_check.toggled.connect(self.set_simulation)
            self.auto_moog_port = QtWidgets.QComboBox()
            self.auto_pol_port = QtWidgets.QComboBox()
            self.auto_refresh_btn = QtWidgets.QPushButton("Refresh ports")
            self.auto_refresh_btn.clicked.connect(self.refresh_ports)
            self.auto_moog_switch = make_switch("Moog")
            self.auto_camera_switch = make_switch("Camera")
            self.auto_pol_switch = make_switch("Polarizer")
            self.auto_home_btn = QtWidgets.QPushButton("Home / Reset")
            self.auto_home_btn.setEnabled(False)
            self.auto_refresh_position_btn = QtWidgets.QPushButton("Refresh Position")
            self.auto_refresh_position_btn.setEnabled(False)
            self.auto_stop_btn = QtWidgets.QPushButton("STOP - close all")
            self.auto_stop_btn.setStyleSheet("background: #b00020; color: white; font-weight: 700;")
            self.auto_moog_switch.toggled.connect(self.toggle_moog)
            self.auto_camera_switch.toggled.connect(self.toggle_camera)
            self.auto_pol_switch.toggled.connect(self.toggle_polarizer)
            self.auto_home_btn.clicked.connect(self.home_moog)
            self.auto_refresh_position_btn.clicked.connect(self.request_moog_status)
            self.auto_stop_btn.clicked.connect(self.stop_all)
            auto_connection_layout.addWidget(self.auto_sim_check, 0, 0)
            auto_connection_layout.addWidget(self.auto_refresh_btn, 0, 1)
            auto_connection_layout.addWidget(QtWidgets.QLabel("Moog"), 1, 0)
            auto_connection_layout.addWidget(self.auto_moog_port, 1, 1)
            auto_connection_layout.addWidget(self.auto_moog_switch, 1, 2)
            auto_connection_layout.addWidget(QtWidgets.QLabel("Camera"), 2, 0)
            auto_connection_layout.addWidget(self.auto_camera_switch, 2, 1, 1, 2)
            auto_connection_layout.addWidget(QtWidgets.QLabel("Polarizer"), 3, 0)
            auto_connection_layout.addWidget(self.auto_pol_port, 3, 1)
            auto_connection_layout.addWidget(self.auto_pol_switch, 3, 2)
            auto_connection_layout.addWidget(self.auto_home_btn, 4, 0)
            auto_connection_layout.addWidget(self.auto_refresh_position_btn, 4, 1)
            auto_connection_layout.addWidget(self.auto_stop_btn, 4, 2)

            scan_group = QtWidgets.QGroupBox("Auto Scan")
            scan_layout = QtWidgets.QGridLayout(scan_group)
            auto_layout.addWidget(scan_group)
            self.auto_ready_label = QtWidgets.QLabel("Run controls will use pan_offset from calibration.")
            self.auto_ready_label.setWordWrap(True)
            scan_layout.addWidget(self.auto_ready_label, 0, 0, 1, 2)

            self.latitude_input = QtWidgets.QDoubleSpinBox()
            self.latitude_input.setRange(-90.0, 90.0)
            self.latitude_input.setDecimals(6)
            self.latitude_input.setValue(32.23134)
            self.longitude_input = QtWidgets.QDoubleSpinBox()
            self.longitude_input.setRange(-180.0, 180.0)
            self.longitude_input.setDecimals(6)
            self.longitude_input.setValue(-110.94712)
            self.target_start_input = QtWidgets.QDoubleSpinBox()
            self.target_start_input.setRange(-10.0, 90.0)
            self.target_start_input.setDecimals(1)
            self.target_start_input.setValue(4.0)
            self.target_end_input = QtWidgets.QDoubleSpinBox()
            self.target_end_input.setRange(-10.0, 90.0)
            self.target_end_input.setDecimals(1)
            self.target_end_input.setValue(90.0)
            self.target_step_input = QtWidgets.QDoubleSpinBox()
            self.target_step_input.setRange(0.1, 10.0)
            self.target_step_input.setDecimals(1)
            self.target_step_input.setValue(0.2)
            self.tolerance_input = QtWidgets.QDoubleSpinBox()
            self.tolerance_input.setRange(0.01, 5.0)
            self.tolerance_input.setDecimals(2)
            self.tolerance_input.setValue(0.10)
            self.check_interval_input = QtWidgets.QSpinBox()
            self.check_interval_input.setRange(1, 3600)
            self.check_interval_input.setValue(10)
            self.start_tilt_input = QtWidgets.QDoubleSpinBox()
            self.start_tilt_input.setRange(0.0, 90.0)
            self.start_tilt_input.setDecimals(1)
            self.start_tilt_input.setValue(0.0)
            self.scan_step_tilt_input = QtWidgets.QDoubleSpinBox()
            self.scan_step_tilt_input.setRange(0.1, 20.0)
            self.scan_step_tilt_input.setDecimals(1)
            self.scan_step_tilt_input.setValue(2.0)
            self.polarizer_angles_input = QtWidgets.QLineEdit("0,45,90,135")
            self.location_input = QtWidgets.QLineEdit("MeinelRoof")
            self.output_dir_input = QtWidgets.QLineEdit(os.getcwd())
            self.browse_output_btn = QtWidgets.QPushButton("Browse")
            self.browse_output_btn.clicked.connect(self.browse_output_dir)
            self.load_auto_scan_settings()
            self.run_now_btn = QtWidgets.QPushButton()
            self.run_now_btn.clicked.connect(self.auto_scan_button_clicked)
            self.set_auto_scan_button_state("idle")
            self.settings_btn = QtWidgets.QPushButton("Acquisition Settings...")
            self.settings_btn.clicked.connect(self.open_auto_scan_settings)
            self.settings_summary_label = QtWidgets.QLabel("")
            self.settings_summary_label.setWordWrap(True)
            self.sun_status_label = QtWidgets.QLabel("Sun trigger monitor stopped")
            self.sun_status_label.setWordWrap(True)

            row = 1
            scan_layout.addWidget(self.settings_btn, row, 0, 1, 2)
            row += 1
            scan_layout.addWidget(self.settings_summary_label, row, 0, 1, 2)
            row += 1
            scan_layout.addWidget(self.run_now_btn, row, 0, 1, 2)
            row += 1
            scan_layout.addWidget(self.sun_status_label, row, 0, 1, 2)
            self.update_settings_summary()

            preview_group = QtWidgets.QGroupBox("Acquisition 0 Preview")
            preview_layout = QtWidgets.QGridLayout(preview_group)
            self.auto_preview_labels = []
            for preview_idx in range(4):
                label = AutoScanPreviewTile(f"Image {preview_idx + 1}")
                self.auto_preview_labels.append(label)
                preview_layout.addWidget(label, 0, preview_idx)
            auto_layout.addWidget(preview_group)

            self.auto_log = QtWidgets.QPlainTextEdit()
            style_log_box(self.auto_log)
            auto_layout.addWidget(self.auto_log)
            auto_layout.addStretch(1)

            auto_scroll = QtWidgets.QScrollArea()
            auto_scroll.setWidgetResizable(True)
            auto_scroll.setFrameShape(no_frame)
            auto_scroll.setWidget(auto_page)

            central = QtWidgets.QWidget()
            root = QtWidgets.QHBoxLayout(central)
            root.setContentsMargins(6, 6, 6, 6)
            root.setSpacing(6)

            controls = QtWidgets.QWidget()
            controls.setMinimumWidth(300)
            controls.setMaximumWidth(330)
            controls_layout = QtWidgets.QVBoxLayout(controls)
            controls_layout.setContentsMargins(5, 5, 5, 5)
            controls_layout.setSpacing(5)
            controls_scroll = QtWidgets.QScrollArea()
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setFrameShape(no_frame)
            controls_scroll.setMinimumWidth(315)
            controls_scroll.setMaximumWidth(350)
            controls_scroll.setWidget(controls)
            root.addWidget(controls_scroll)

            self.done_calibration_btn = QtWidgets.QPushButton("Done Calibration - Back to Auto")
            self.done_calibration_btn.setMinimumHeight(28)
            self.done_calibration_btn.clicked.connect(self.finish_calibration)
            controls_layout.addWidget(self.done_calibration_btn)

            connection_group = QtWidgets.QGroupBox("Connections")
            connection_layout = QtWidgets.QGridLayout(connection_group)
            controls_layout.addWidget(connection_group)

            self.sim_check = make_switch("Simulation")
            self.sim_check.setChecked(True)
            self.sim_check.toggled.connect(self.set_simulation)
            self.refresh_btn = QtWidgets.QPushButton("Refresh ports")
            self.refresh_btn.clicked.connect(self.refresh_ports)
            connection_layout.addWidget(self.sim_check, 0, 0)
            connection_layout.addWidget(self.refresh_btn, 0, 1)

            self.moog_port = QtWidgets.QComboBox()
            self.pol_port = QtWidgets.QComboBox()
            connection_layout.addWidget(QtWidgets.QLabel("Moog port"), 1, 0)
            connection_layout.addWidget(self.moog_port, 1, 1)
            self.moog_switch = make_switch("Moog")
            self.camera_switch = make_switch("Camera")
            self.pol_switch = make_switch("Polarizer")
            self.moog_switch.toggled.connect(self.toggle_moog)
            self.camera_switch.toggled.connect(self.toggle_camera)
            self.pol_switch.toggled.connect(self.toggle_polarizer)
            connection_layout.addWidget(self.moog_switch, 1, 2)
            connection_layout.addWidget(self.camera_switch, 2, 0, 1, 3)
            connection_layout.addWidget(QtWidgets.QLabel("Polarizer port"), 3, 0)
            connection_layout.addWidget(self.pol_port, 3, 1)
            connection_layout.addWidget(self.pol_switch, 3, 2)

            self.stop_btn = QtWidgets.QPushButton("STOP - close all")
            self.stop_btn.setMinimumHeight(30)
            self.stop_btn.setStyleSheet("background: #b00020; color: white; font-weight: 700;")
            self.stop_btn.clicked.connect(self.stop_all)
            self.home_btn = QtWidgets.QPushButton("Home / Reset")
            self.home_btn.setEnabled(False)
            self.home_btn.clicked.connect(self.home_moog)
            self.refresh_position_btn = QtWidgets.QPushButton("Refresh Position")
            self.refresh_position_btn.setEnabled(False)
            self.refresh_position_btn.clicked.connect(self.request_moog_status)
            connection_layout.addWidget(self.home_btn, 4, 0)
            connection_layout.addWidget(self.refresh_position_btn, 4, 1)
            connection_layout.addWidget(self.stop_btn, 4, 2)

            center_group = QtWidgets.QGroupBox("Sun centering")
            center_layout = QtWidgets.QGridLayout(center_group)
            controls_layout.addWidget(center_group)
            self.deg_per_pixel = QtWidgets.QDoubleSpinBox()
            self.deg_per_pixel.setRange(0.000001, 1.0)
            self.deg_per_pixel.setDecimals(6)
            self.deg_per_pixel.setValue(UV_DEG_PER_PIXEL)
            self.deg_per_pixel.setSingleStep(0.0001)
            self.pan_positive_moves_right_check = QtWidgets.QCheckBox("+Pan moves sun right")
            self.pan_positive_moves_right_check.setChecked(False)
            self.aim_sun_btn = QtWidgets.QPushButton("Aim at calculated sun")
            self.aim_sun_btn.setEnabled(False)
            self.aim_sun_btn.clicked.connect(self.aim_at_calculated_sun)
            self.center_btn = QtWidgets.QPushButton("Center sun in FOV")
            self.center_btn.clicked.connect(self.center_sun)
            center_layout.addWidget(QtWidgets.QLabel("Deg per pixel"), 0, 0)
            center_layout.addWidget(self.deg_per_pixel, 0, 1)
            center_layout.addWidget(self.pan_positive_moves_right_check, 1, 0, 1, 2)
            center_layout.addWidget(self.aim_sun_btn, 2, 0, 1, 2)
            center_layout.addWidget(self.center_btn, 3, 0, 1, 2)

            status_group = QtWidgets.QGroupBox("Pointing")
            status_layout = QtWidgets.QVBoxLayout(status_group)
            controls_layout.addWidget(status_group)
            self.pan_display = DigitAxisWidget("Pan [deg]", lambda delta: self.jog(delta, 0.0))
            self.tilt_display = DigitAxisWidget("Tilt [deg]", lambda delta: self.jog(0.0, delta))
            status_layout.addWidget(self.pan_display)
            status_layout.addWidget(self.tilt_display)

            pan_offset_layout = QtWidgets.QHBoxLayout()
            self.pan_offset_input = QtWidgets.QDoubleSpinBox()
            self.pan_offset_input.setRange(PAN_MIN_DEG, PAN_MAX_DEG)
            self.pan_offset_input.setDecimals(2)
            self.pan_offset_input.setSingleStep(MOOG_RESOLUTION_DEG)
            self.pan_offset_input.setValue(self.pan_offset)
            self.pan_offset_input.valueChanged.connect(self.set_calibration_pan_offset)
            pan_offset_layout.addWidget(QtWidgets.QLabel("Pan Offset [deg]"))
            pan_offset_layout.addWidget(self.pan_offset_input)
            controls_layout.addLayout(pan_offset_layout)

            polarizer_group = QtWidgets.QGroupBox("Polarizer")
            polarizer_layout = QtWidgets.QHBoxLayout(polarizer_group)
            controls_layout.addWidget(polarizer_group)
            self.pol_angle = QtWidgets.QComboBox()
            self.pol_angle.addItems(["0", "45", "90", "135"])
            self.pol_angle.currentTextChanged.connect(lambda text: self.set_polarizer(float(text)))
            polarizer_layout.addWidget(QtWidgets.QLabel("Angle"))
            polarizer_layout.addWidget(self.pol_angle)

            exposure_group = QtWidgets.QGroupBox("Exposure")
            exposure_layout = QtWidgets.QGridLayout(exposure_group)
            controls_layout.addWidget(exposure_group)
            self.auto_exposure_check = make_switch("Auto exposure")
            self.auto_exposure_check.toggled.connect(self.set_auto_exposure)
            self.exposure_us = QtWidgets.QDoubleSpinBox()
            self.exposure_us.setRange(AUTO_EXPOSURE_MIN_US, AUTO_EXPOSURE_MAX_US)
            self.exposure_us.setDecimals(0)
            self.exposure_us.setSingleStep(100.0)
            self.exposure_us.setValue(DEFAULT_EXPOSURE_US)
            self.exposure_us.valueChanged.connect(self.set_manual_exposure)
            self.target_median = QtWidgets.QDoubleSpinBox()
            self.target_median.setRange(1.0, 4095.0)
            self.target_median.setDecimals(0)
            self.target_median.setSingleStep(100.0)
            self.target_median.setValue(AUTO_EXPOSURE_TARGET_MEDIAN)
            self.auto_exposure_metric = QtWidgets.QComboBox()
            self.auto_exposure_metric.addItems(["Median", "Max"])
            self.auto_exposure_metric.currentTextChanged.connect(self.set_auto_exposure_mode)
            self.max_exposure_us = QtWidgets.QDoubleSpinBox()
            self.max_exposure_us.setRange(AUTO_EXPOSURE_MIN_US, AUTO_EXPOSURE_MAX_US)
            self.max_exposure_us.setDecimals(0)
            self.max_exposure_us.setSingleStep(10_000.0)
            self.max_exposure_us.setValue(self.auto_exposure_max_us)
            self.max_exposure_us.valueChanged.connect(self.set_auto_exposure_max)
            self.exposure_status = QtWidgets.QLabel("Manual: 1000 us")
            exposure_layout.addWidget(self.auto_exposure_check, 0, 0, 1, 2)
            exposure_layout.addWidget(QtWidgets.QLabel("Exposure [us]"), 1, 0)
            exposure_layout.addWidget(self.exposure_us, 1, 1)
            exposure_layout.addWidget(QtWidgets.QLabel("Target median"), 2, 0)
            exposure_layout.addWidget(self.target_median, 2, 1)
            exposure_layout.addWidget(QtWidgets.QLabel("Auto metric"), 3, 0)
            exposure_layout.addWidget(self.auto_exposure_metric, 3, 1)
            exposure_layout.addWidget(QtWidgets.QLabel("Max exposure [us]"), 4, 0)
            exposure_layout.addWidget(self.max_exposure_us, 4, 1)
            exposure_layout.addWidget(self.exposure_status, 5, 0, 1, 2)

            remote_group = QtWidgets.QGroupBox("Remote control")
            remote_layout = QtWidgets.QGridLayout(remote_group)
            controls_layout.addWidget(remote_group)
            self.remote_check = make_switch("HTTP API")
            self.remote_check.toggled.connect(self.toggle_remote)
            self.remote_port = QtWidgets.QSpinBox()
            self.remote_port.setRange(1024, 65535)
            self.remote_port.setValue(8765)
            remote_layout.addWidget(self.remote_check, 0, 0)
            remote_layout.addWidget(self.remote_port, 0, 1)
            self.remote_label = QtWidgets.QLabel("Stopped")
            remote_layout.addWidget(self.remote_label, 1, 0, 1, 2)

            self.save_h5_btn = QtWidgets.QPushButton("Save calibration H5")
            self.save_h5_btn.clicked.connect(self.save_calibration_h5)
            controls_layout.addWidget(self.save_h5_btn)
            controls_layout.addStretch(1)

            self.log = QtWidgets.QPlainTextEdit()
            style_log_box(self.log)
            controls_layout.addWidget(self.log)

            preview_panel = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_panel)
            preview_layout.setContentsMargins(0, 0, 0, 0)
            preview_layout.setSpacing(5)
            zoom_layout = QtWidgets.QHBoxLayout()
            self.fit_preview_btn = QtWidgets.QPushButton("Fit")
            self.fit_preview_btn.clicked.connect(self.fit_camera_preview)
            self.zoom_out_btn = QtWidgets.QPushButton("-")
            self.zoom_out_btn.setFixedWidth(32)
            self.zoom_out_btn.clicked.connect(lambda: self.adjust_camera_zoom(-10))
            self.zoom_in_btn = QtWidgets.QPushButton("+")
            self.zoom_in_btn.setFixedWidth(32)
            self.zoom_in_btn.clicked.connect(lambda: self.adjust_camera_zoom(10))
            self.zoom_slider = QtWidgets.QSlider(
                getattr(getattr(QtCore.Qt, "Orientation", QtCore.Qt), "Horizontal")
            )
            self.zoom_slider.setRange(25, 250)
            self.zoom_slider.setValue(100)
            self.zoom_slider.valueChanged.connect(self.set_camera_zoom)
            self.zoom_label = QtWidgets.QLabel("Fit")
            self.zoom_label.setMinimumWidth(46)
            zoom_layout.addWidget(QtWidgets.QLabel("Preview"))
            zoom_layout.addStretch(1)
            zoom_layout.addWidget(self.fit_preview_btn)
            zoom_layout.addWidget(self.zoom_out_btn)
            zoom_layout.addWidget(self.zoom_slider)
            zoom_layout.addWidget(self.zoom_in_btn)
            zoom_layout.addWidget(self.zoom_label)
            preview_layout.addLayout(zoom_layout)

            self.camera_view = CameraView()
            self.camera_scroll = QtWidgets.QScrollArea()
            self.camera_scroll.setWidgetResizable(False)
            self.camera_scroll.setAlignment(align_center)
            self.camera_scroll.setWidget(self.camera_view)
            preview_layout.addWidget(self.camera_scroll, 1)
            root.addWidget(preview_panel, 1)

            self.pages.addWidget(auto_scroll)
            self.pages.addWidget(central)
            self.pages.setCurrentWidget(auto_scroll)
            self.update_auto_calibration_labels()
            self.preview_fit = True
            QtCore.QTimer.singleShot(0, self.fit_camera_preview)

        def log_msg(self, text):
            self.last_message = f"{time.strftime('%H:%M:%S')}  {text}"
            for log_name in ("auto_log", "log"):
                log = getattr(self, log_name, None)
                if log is not None:
                    log.appendPlainText(self.last_message)

        def camera_timeout_context(self, error):
            diagnostics_fn = getattr(self.camera, "diagnostics", None)
            diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
            exposure = diagnostics.get("ExposureTime", diagnostics.get("exposure_us", self.camera.get_exposure()))
            width = diagnostics.get("Width", "?")
            height = diagnostics.get("Height", "?")
            pixel_format = diagnostics.get("PixelFormat", "?")
            trigger = diagnostics.get("TriggerMode", "?")
            last_ms = diagnostics.get("last_frame_ms", "?")
            failures = diagnostics.get("frames_failed", self.frame_error_count)
            likely = []
            message = str(error).lower()
            if "timed out" in message or "timeout" in message:
                likely.append("相机在 SDK 等待时间内没有返回完整帧")
            if self.auto_exposure_check.isChecked():
                likely.append("预览自动曝光可能正在改变曝光时间")
            if float(self.camera.get_exposure()) > 50_000:
                likely.append("当前曝光较长，单帧等待时间会变长")
            likely.append("USB/网口带宽、线缆、相机被其他程序占用或刚重连后未稳定也会导致超时")
            return (
                f"Preview frame failed; retrying in {PREVIEW_FRAME_ERROR_BACKOFF_SEC:.0f}s. "
                f"camera={diagnostics.get('camera_id', '?')}, exposure={exposure} us, "
                f"size={width}x{height}, pixel={pixel_format}, trigger={trigger}, "
                f"last_wait={last_ms} ms, failures={failures}. "
                f"Likely: {'; '.join(likely)}"
            )

        def update_sim_camera_site(self):
            if isinstance(self.camera, SimCameraController):
                self.camera.set_site(self.latitude_input.value(), self.longitude_input.value())

        def fit_camera_preview(self):
            self.preview_fit = True
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(100)
            self.zoom_slider.blockSignals(False)
            self.zoom_label.setText("Fit")
            viewport = self.camera_scroll.viewport().size()
            side = max(160, min(viewport.width() - 4, viewport.height() - 4))
            self.camera_view.set_display_side(side)

        def set_camera_zoom(self, value):
            self.preview_fit = False
            side = max(160, int(560 * float(value) / 100.0))
            self.camera_view.set_display_side(side)
            self.zoom_label.setText(f"{value}%")

        def adjust_camera_zoom(self, delta):
            self.zoom_slider.setValue(self.zoom_slider.value() + delta)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if getattr(self, "preview_fit", False):
                QtCore.QTimer.singleShot(0, self.fit_camera_preview)

        def show_calibration_page(self):
            self.pages.setCurrentIndex(1)
            if self.preview_fit:
                QtCore.QTimer.singleShot(0, self.fit_camera_preview)
            self.log_msg("Calibration page opened")

        def finish_calibration(self):
            self.pan_offset = quantize_pointing(self.pan_offset_input.value())
            self.save_auto_scan_settings()
            self.calibration_complete = True
            self.update_auto_calibration_labels()
            self.pages.setCurrentIndex(0)
            self.log_msg(
                "Calibration complete: "
                f"pan_offset={self.pan_offset:.2f}, tilt_offset={self.tilt_offset:.2f}"
            )

        def update_auto_calibration_labels(self):
            status_text = "Complete" if self.calibration_complete else "Not complete"
            self.calibration_status_label.setText(status_text)
            self.auto_pan_offset_label.setText(f"{self.pan_offset:.2f}")
            self.auto_tilt_offset_label.setText(f"{self.tilt_offset:.2f}")
            centered = self.last_center_result.get("centered_pan_deg") if self.last_center_result else None
            self.auto_centered_pan_label.setText("none" if centered is None else f"{float(centered):.2f}")
            pan_offset_input = getattr(self, "pan_offset_input", None)
            if pan_offset_input is not None and pan_offset_input.value() != self.pan_offset:
                pan_offset_input.blockSignals(True)
                pan_offset_input.setValue(self.pan_offset)
                pan_offset_input.blockSignals(False)

        def set_calibration_pan_offset(self, value):
            self.pan_offset = quantize_pointing(value)
            self.update_auto_calibration_labels()
            self.log_msg(f"Pan offset set to {self.pan_offset:.2f} deg")

        def browse_output_dir(self):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Auto Scan Output Folder", self.output_dir_input.text())
            if path:
                self.output_dir_input.setText(path)

        def auto_scan_settings_payload(self):
            return {
                "latitude": self.latitude_input.value(),
                "longitude": self.longitude_input.value(),
                "target_start": self.target_start_input.value(),
                "target_end": self.target_end_input.value(),
                "target_step": self.target_step_input.value(),
                "tolerance": self.tolerance_input.value(),
                "check_interval": self.check_interval_input.value(),
                "pan_offset": self.pan_offset,
                "tilt_offset": self.tilt_offset,
                "start_tilt": self.start_tilt_input.value(),
                "scan_step_tilt": self.scan_step_tilt_input.value(),
                "polarizer_angles": self.polarizer_angles_input.text().strip(),
                "location": self.location_input.text().strip(),
                "output_dir": self.output_dir_input.text().strip(),
                "sun_trigger_enabled": bool(self.sun_trigger_enabled),
                "auto_exposure_mode": self.auto_exposure_mode,
                "auto_exposure_max_us": self.auto_exposure_max_us,
            }

        def apply_auto_scan_settings_payload(self, settings):
            setters = (
                ("latitude", self.latitude_input.setValue),
                ("longitude", self.longitude_input.setValue),
                ("target_start", self.target_start_input.setValue),
                ("target_end", self.target_end_input.setValue),
                ("target_step", self.target_step_input.setValue),
                ("tolerance", self.tolerance_input.setValue),
                ("check_interval", self.check_interval_input.setValue),
                ("start_tilt", self.start_tilt_input.setValue),
                ("scan_step_tilt", self.scan_step_tilt_input.setValue),
            )
            for key, setter in setters:
                if key in settings:
                    setter(settings[key])
            if "polarizer_angles" in settings:
                self.polarizer_angles_input.setText(str(settings["polarizer_angles"]))
            if "location" in settings:
                self.location_input.setText(str(settings["location"]))
            if "output_dir" in settings:
                self.output_dir_input.setText(str(settings["output_dir"]))
            if "pan_offset" in settings:
                self.pan_offset = quantize_pointing(settings["pan_offset"])
            if "tilt_offset" in settings:
                self.tilt_offset = quantize_pointing(settings["tilt_offset"])
            if "auto_exposure_mode" in settings:
                self.auto_exposure_mode = str(settings["auto_exposure_mode"]).lower()
                if self.auto_exposure_mode not in ("median", "max"):
                    self.auto_exposure_mode = "median"
                if hasattr(self, "auto_exposure_metric"):
                    self.auto_exposure_metric.setCurrentText(self.auto_exposure_mode.title())
            if "auto_exposure_max_us" in settings:
                self.auto_exposure_max_us = float(settings["auto_exposure_max_us"])
                if hasattr(self, "max_exposure_us"):
                    self.max_exposure_us.setValue(self.auto_exposure_max_us)
            self.sun_trigger_enabled = bool(settings.get("sun_trigger_enabled", True))

        def load_auto_scan_settings(self):
            try:
                with open(AUTO_SCAN_SETTINGS_PATH, "r", encoding="utf-8") as handle:
                    settings = json.load(handle)
            except FileNotFoundError:
                return
            except Exception as exc:
                self.log_msg(f"Could not load auto scan settings: {exc}")
                return
            if isinstance(settings, dict):
                self.apply_auto_scan_settings_payload(settings)

        def save_auto_scan_settings(self):
            settings = self.auto_scan_settings_payload()
            os.makedirs(os.path.dirname(AUTO_SCAN_SETTINGS_PATH), exist_ok=True)
            with open(AUTO_SCAN_SETTINGS_PATH, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, indent=2, sort_keys=True)

        def update_settings_summary(self):
            if not hasattr(self, "settings_summary_label"):
                return
            self.settings_summary_label.setText(
                "Location={location}, output={output}\n"
                "Sun trigger={sun_trigger}; targets {start:.1f}-{end:.1f} deg step {step:.1f}, tol {tol:.2f}, check {interval:d}s\n"
                "Offsets pan={pan_offset:.2f}, tilt={tilt_offset:.2f}; scan dtilt start {tilt_start:.1f}, step {tilt_step:.1f}\n"
                "Auto exposure: {metric}, fixed {fixed_exp:.0f} us for dtilt<3, min {min_exp:.0f} us, max {max_exp:.0f} us; Polarizer={angles}".format(
                    location=self.location_input.text().strip() or "ULTRASIP",
                    output=self.output_dir_input.text().strip() or os.getcwd(),
                    sun_trigger="ON" if self.sun_trigger_enabled else "OFF",
                    start=self.target_start_input.value(),
                    end=self.target_end_input.value(),
                    step=self.target_step_input.value(),
                    tol=self.tolerance_input.value(),
                    interval=self.check_interval_input.value(),
                    pan_offset=self.pan_offset,
                    tilt_offset=self.tilt_offset,
                    tilt_start=self.start_tilt_input.value(),
                    tilt_step=self.scan_step_tilt_input.value(),
                    metric=self.auto_exposure_mode,
                    fixed_exp=AUTO_EXPOSURE_FIXED_NEAR_SUN_US,
                    min_exp=AUTO_EXPOSURE_MIN_US,
                    max_exp=self.auto_exposure_max_us,
                    angles=self.polarizer_angles_input.text().strip(),
                )
            )

        def open_auto_scan_settings(self):
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Acquisition Settings")
            layout = QtWidgets.QGridLayout(dialog)
            fields = {}

            def add_double(row, key, label, source, min_value, max_value, decimals, step=None):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(min_value, max_value)
                widget.setDecimals(decimals)
                if step is not None:
                    widget.setSingleStep(step)
                widget.setValue(source.value())
                layout.addWidget(QtWidgets.QLabel(label), row, 0)
                layout.addWidget(widget, row, 1)
                fields[key] = widget

            sun_trigger_enabled = make_switch("Sun Trigger")
            sun_trigger_enabled.setChecked(bool(self.sun_trigger_enabled))
            layout.addWidget(QtWidgets.QLabel("Sun trigger"), 0, 0)
            layout.addWidget(sun_trigger_enabled, 0, 1)
            fields["sun_trigger_enabled"] = sun_trigger_enabled

            add_double(1, "latitude", "Latitude", self.latitude_input, -90.0, 90.0, 6)
            add_double(2, "longitude", "Longitude", self.longitude_input, -180.0, 180.0, 6)
            add_double(3, "target_start", "Target start altitude [deg]", self.target_start_input, -10.0, 90.0, 1, 0.1)
            add_double(4, "target_end", "Target end altitude [deg]", self.target_end_input, -10.0, 90.0, 1, 0.1)
            add_double(5, "target_step", "Target step [deg]", self.target_step_input, 0.1, 10.0, 1, 0.1)
            add_double(6, "tolerance", "Tolerance [deg]", self.tolerance_input, 0.01, 5.0, 2, 0.01)

            check_interval = QtWidgets.QSpinBox()
            check_interval.setRange(1, 3600)
            check_interval.setValue(self.check_interval_input.value())
            layout.addWidget(QtWidgets.QLabel("Check interval [sec]"), 7, 0)
            layout.addWidget(check_interval, 7, 1)
            fields["check_interval"] = check_interval

            pan_offset = QtWidgets.QDoubleSpinBox()
            pan_offset.setRange(PAN_MIN_DEG, PAN_MAX_DEG)
            pan_offset.setDecimals(2)
            pan_offset.setSingleStep(MOOG_RESOLUTION_DEG)
            pan_offset.setValue(self.pan_offset)
            layout.addWidget(QtWidgets.QLabel("Pan offset [deg]"), 8, 0)
            layout.addWidget(pan_offset, 8, 1)
            fields["pan_offset"] = pan_offset

            tilt_offset = QtWidgets.QDoubleSpinBox()
            tilt_offset.setRange(TILT_MIN_DEG, TILT_MAX_DEG)
            tilt_offset.setDecimals(2)
            tilt_offset.setSingleStep(MOOG_RESOLUTION_DEG)
            tilt_offset.setValue(self.tilt_offset)
            layout.addWidget(QtWidgets.QLabel("Tilt offset [deg]"), 9, 0)
            layout.addWidget(tilt_offset, 9, 1)
            fields["tilt_offset"] = tilt_offset

            add_double(10, "start_tilt", "Scan start tilt offset [deg]", self.start_tilt_input, 0.0, 90.0, 1, 0.1)
            add_double(11, "scan_step_tilt", "Scan step tilt [deg]", self.scan_step_tilt_input, 0.1, 20.0, 1, 0.1)

            auto_exposure_mode = QtWidgets.QComboBox()
            auto_exposure_mode.addItems(["Median", "Max"])
            auto_exposure_mode.setCurrentText(self.auto_exposure_mode.title())
            layout.addWidget(QtWidgets.QLabel("Auto exposure metric"), 12, 0)
            layout.addWidget(auto_exposure_mode, 12, 1)
            fields["auto_exposure_mode"] = auto_exposure_mode

            auto_exposure_max = QtWidgets.QDoubleSpinBox()
            auto_exposure_max.setRange(AUTO_EXPOSURE_MIN_US, AUTO_EXPOSURE_MAX_US)
            auto_exposure_max.setDecimals(0)
            auto_exposure_max.setSingleStep(10_000.0)
            auto_exposure_max.setValue(self.auto_exposure_max_us)
            layout.addWidget(QtWidgets.QLabel("Max exposure [us]"), 13, 0)
            layout.addWidget(auto_exposure_max, 13, 1)
            fields["auto_exposure_max_us"] = auto_exposure_max

            polarizer_angles = QtWidgets.QLineEdit(self.polarizer_angles_input.text())
            layout.addWidget(QtWidgets.QLabel("Polarizer angles [deg]"), 14, 0)
            layout.addWidget(polarizer_angles, 14, 1)
            fields["polarizer_angles"] = polarizer_angles

            location = QtWidgets.QLineEdit(self.location_input.text())
            layout.addWidget(QtWidgets.QLabel("Location"), 15, 0)
            layout.addWidget(location, 15, 1)
            fields["location"] = location

            output_dir = QtWidgets.QLineEdit(self.output_dir_input.text())
            browse = QtWidgets.QPushButton("Browse")
            def browse_dialog_output():
                path = QtWidgets.QFileDialog.getExistingDirectory(dialog, "Select Auto Scan Output Folder", output_dir.text())
                if path:
                    output_dir.setText(path)
            browse.clicked.connect(browse_dialog_output)
            output_row = QtWidgets.QHBoxLayout()
            output_row.addWidget(output_dir)
            output_row.addWidget(browse)
            layout.addWidget(QtWidgets.QLabel("Output folder"), 16, 0)
            layout.addLayout(output_row, 16, 1)
            fields["output_dir"] = output_dir

            dialog_buttons = getattr(QtWidgets.QDialogButtonBox, "StandardButton", QtWidgets.QDialogButtonBox)
            buttons = QtWidgets.QDialogButtonBox(dialog_buttons.Save | dialog_buttons.Cancel)
            layout.addWidget(buttons, 17, 0, 1, 2)

            def save_and_close():
                try:
                    parsed_angles = []
                    for part in fields["polarizer_angles"].text().split(","):
                        part = part.strip()
                        if part:
                            parsed_angles.append(float(part))
                    if not parsed_angles:
                        raise ValueError("At least one polarizer angle is required.")
                    self.latitude_input.setValue(fields["latitude"].value())
                    self.longitude_input.setValue(fields["longitude"].value())
                    self.target_start_input.setValue(fields["target_start"].value())
                    self.target_end_input.setValue(fields["target_end"].value())
                    self.target_step_input.setValue(fields["target_step"].value())
                    self.tolerance_input.setValue(fields["tolerance"].value())
                    self.check_interval_input.setValue(fields["check_interval"].value())
                    self.pan_offset = quantize_pointing(fields["pan_offset"].value())
                    self.tilt_offset = quantize_pointing(fields["tilt_offset"].value())
                    self.start_tilt_input.setValue(fields["start_tilt"].value())
                    self.scan_step_tilt_input.setValue(fields["scan_step_tilt"].value())
                    self.set_auto_exposure_mode(fields["auto_exposure_mode"].currentText())
                    self.set_auto_exposure_max(fields["auto_exposure_max_us"].value())
                    self.polarizer_angles_input.setText(",".join(f"{angle:g}" for angle in parsed_angles))
                    self.location_input.setText(fields["location"].text())
                    self.output_dir_input.setText(fields["output_dir"].text())
                    self.sun_trigger_enabled = bool(fields["sun_trigger_enabled"].isChecked())
                    self.save_auto_scan_settings()
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(dialog, "Invalid settings", str(exc))
                    return
                self.update_settings_summary()
                self.log_msg(f"Auto scan settings saved to {AUTO_SCAN_SETTINGS_PATH}")
                dialog.accept()

            buttons.accepted.connect(save_and_close)
            buttons.rejected.connect(dialog.reject)
            dialog.exec()

        def set_auto_scan_button_state(self, state):
            if state == "idle":
                self.run_now_btn.setText("Start Auto Scan")
                self.run_now_btn.setEnabled(True)
                self.run_now_btn.setStyleSheet("background: #15803d; color: white; font-weight: 700;")
            elif state == "running":
                self.run_now_btn.setText("Stop Auto Scan")
                self.run_now_btn.setEnabled(True)
                self.run_now_btn.setStyleSheet("background: #b91c1c; color: white; font-weight: 700;")
            elif state == "stopping":
                self.run_now_btn.setText("Stop requested")
                self.run_now_btn.setEnabled(False)
                self.run_now_btn.setStyleSheet("background: #6b7280; color: white; font-weight: 700;")

        def auto_scan_button_clicked(self):
            if self.auto_job_running or self.auto_monitoring:
                self.stop_acquisition()
            else:
                self.run_auto_scan_now()

        def polarizer_angles(self):
            values = []
            for part in self.polarizer_angles_input.text().split(","):
                part = part.strip()
                if part:
                    values.append(float(part))
            if not values:
                raise ValueError("At least one polarizer angle is required.")
            return values

        def auto_scan_config(self):
            return {
                "latitude": self.latitude_input.value(),
                "longitude": self.longitude_input.value(),
                "location": self.location_input.text().strip() or "ULTRASIP",
                "output_dir": self.output_dir_input.text().strip() or os.getcwd(),
                "start_tilt": self.start_tilt_input.value(),
                "step_tilt": self.scan_step_tilt_input.value(),
                "angles": self.polarizer_angles(),
                "pan_offset": self.pan_offset,
                "tilt_offset": self.tilt_offset,
                "uv_wavelength": "355 FWHM 10nm",
                "auto_exposure": self.auto_exposure_check.isChecked(),
                "target_median": self.target_median.value(),
                "auto_exposure_mode": self.auto_exposure_mode,
                "auto_exposure_max_us": self.auto_exposure_max_us,
            }

        def run_auto_scan_now(self):
            try:
                config = self.auto_scan_config()
            except Exception as exc:
                self.log_msg(f"Auto scan config invalid: {exc}")
                return
            if self.sun_trigger_enabled:
                self.start_sun_monitor()
            else:
                self.trigger_auto_measurement(None, None, None, datetime.now(), config=config)

        def set_sun_trigger_mode(self, checked):
            if checked:
                self.sun_status_label.setText("Sun trigger mode enabled; press Start Auto Scan to monitor")
                self.log_msg("Sun trigger mode enabled")
            else:
                self.sun_status_label.setText("Sun trigger mode disabled; Start Auto Scan runs immediately")
                self.log_msg("Sun trigger mode disabled")

        def sun_target_values(self):
            start = self.target_start_input.value()
            end = self.target_end_input.value()
            step = self.target_step_input.value()
            if end < start:
                start, end = end, start
            count = int(math.floor((end - start) / step)) + 1
            return [round(start + i * step, 3) for i in range(count + 1) if start + i * step <= end + 1e-9]

        def start_sun_monitor(self):
            if self.auto_monitoring:
                self.log_msg("Sun trigger monitor already running")
                return
            self.completed_sun_targets = set()
            self.queued_sun_trigger = None
            self.last_sun_altitude = None
            interval_ms = int(self.check_interval_input.value() * 1000)
            self.auto_monitor_timer.setInterval(interval_ms)
            self.auto_monitor_timer.start()
            self.auto_monitoring = True
            self.auto_stop_requested = False
            self.set_auto_scan_button_state("running")
            self.log_msg("Sun trigger monitor started")
            self.check_sun_trigger()

        def stop_sun_monitor(self):
            self.auto_monitor_timer.stop()
            self.auto_monitoring = False
            self.queued_sun_trigger = None
            self.last_sun_altitude = None
            self.sun_status_label.setText("Sun trigger monitor stopped")
            self.log_msg("Sun trigger monitor stopped")

        def sun_due_targets(self, pending, altitude, tolerance):
            due = {target for target in pending if abs(altitude - target) <= tolerance}
            if self.last_sun_altitude is not None:
                low = min(self.last_sun_altitude, altitude)
                high = max(self.last_sun_altitude, altitude)
                due.update(target for target in pending if low <= target <= high)
            return sorted(due)

        def check_sun_trigger(self):
            dt = datetime.now()
            latitude = self.latitude_input.value()
            longitude = self.longitude_input.value()
            azimuth, altitude = solar_position_deg(dt, latitude, longitude)
            tolerance = self.tolerance_input.value()
            targets = self.sun_target_values()
            pending = [target for target in targets if target not in self.completed_sun_targets]
            nearest = min(pending, key=lambda target: abs(altitude - target)) if pending else None
            status = (
                f"{dt.strftime('%H:%M:%S')} sun altitude={altitude:.2f} deg, "
                f"azimuth={azimuth:.2f} deg"
            )
            if nearest is not None:
                status += f", next target={nearest:.2f} deg, error={altitude - nearest:+.2f} deg"
            else:
                status += ", all targets completed"
            if self.auto_job_running:
                status += ", acquisition running"
            if self.queued_sun_trigger is not None:
                status += f", queued target={self.queued_sun_trigger['target_altitude']:.2f} deg"
            self.sun_status_label.setText(status)

            due_targets = self.sun_due_targets(pending, altitude, tolerance)
            self.last_sun_altitude = altitude

            if nearest is None:
                return
            if not due_targets:
                return

            trigger_target = min(due_targets, key=lambda target: abs(altitude - target))
            self.completed_sun_targets.update(due_targets)
            if self.auto_job_running:
                self.queued_sun_trigger = {
                    "target_altitude": trigger_target,
                    "actual_altitude": altitude,
                    "azimuth": azimuth,
                    "dt": dt,
                    "missed_count": max(0, len(due_targets) - 1),
                }
                message = (
                    f"Sun target queued while scan is running: target={trigger_target:.2f}, "
                    f"actual={altitude:.2f}, azimuth={azimuth:.2f}"
                )
                if len(due_targets) > 1:
                    message += f", {len(due_targets) - 1} older target(s) marked handled"
                self.log_msg(message)
                return

            self.trigger_auto_measurement(trigger_target, altitude, azimuth, dt)

        def trigger_auto_measurement(self, target_altitude, actual_altitude, azimuth, dt, config=None):
            if h5py is None:
                self.log_msg("Auto scan failed: h5py is not installed")
                return
            if self.auto_job_running:
                self.log_msg("Auto scan already running")
                return
            self.auto_job_running = True
            self.auto_stop_requested = False
            self.stop_acquisition_event.clear()
            self.set_auto_scan_button_state("running")
            self.pause_preview_for_auto_scan()
            if config is None:
                try:
                    config = self.auto_scan_config()
                except Exception as exc:
                    self.auto_job_running = False
                    self.resume_preview_after_auto_scan()
                    if not self.auto_monitoring:
                        self.set_auto_scan_button_state("idle")
                    self.log_msg(f"Auto scan config invalid: {exc}")
                    return
            self.log_msg(
                "Auto scan triggered"
                + (f": target={target_altitude:.2f}, actual={actual_altitude:.2f}, azimuth={azimuth:.2f}" if target_altitude is not None else "")
            )

            def task():
                return self.run_auto_scan_worker(config, target_altitude, actual_altitude, azimuth, dt)

            self.submit_worker("auto_trigger", task)

        def stop_acquisition(self):
            if not self.auto_job_running:
                if self.auto_monitoring:
                    self.stop_sun_monitor()
                    self.set_auto_scan_button_state("idle")
                    self.log_msg("Auto scan stopped")
                else:
                    self.log_msg("No auto scan is running")
                return
            if self.auto_monitoring:
                self.stop_sun_monitor()
            self.queued_sun_trigger = None
            self.auto_stop_requested = True
            self.stop_acquisition_event.set()
            self.set_auto_scan_button_state("stopping")
            self.log_msg("Stop auto scan requested; current scan group will finish first")

        def run_auto_scan_worker(self, config, target_altitude, actual_altitude, trigger_azimuth, trigger_dt):
            output_dir = config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            date_dir = datetime.now().strftime("%Y_%m_%d")
            datapath = os.path.join(output_dir, date_dir)
            os.makedirs(datapath, exist_ok=True)
            timestamp = datetime.now().strftime("%H_%M_%S")
            filename = f"{config['location']}_{datetime.now().strftime('%Y%m%d')}_{timestamp}.h5"
            filepath = os.path.join(datapath, filename)

            angles = config["angles"]
            _, initial_sun_altitude = solar_position_deg(datetime.now(), config["latitude"], config["longitude"])
            end_tilt = int(88.0 - initial_sun_altitude)
            if end_tilt <= config["start_tilt"]:
                dtilts = np.array([], dtype=float)
            else:
                dtilts = np.arange(config["start_tilt"], end_tilt, config["step_tilt"], dtype=float)
            normal_dtilts = [float(dtilt) for dtilt in dtilts if float(dtilt) >= 0.0]
            if end_tilt > 0.0 and not any(abs(dtilt) < 1e-9 for dtilt in normal_dtilts):
                normal_dtilts.insert(0, 0.0)

            acquisition_plan = []
            if normal_dtilts:
                acquisition_plan.append({
                    "group_name": "Aquistion_0",
                    "dtilt": normal_dtilts[0],
                    "kind": "normal",
                    "normal_index": 0,
                })
                for calibration_index, calibration_dtilt in enumerate(CALIBRATION_DTILTS_DEG):
                    acquisition_plan.append({
                        "group_name": acquisition_group_name_for_dtilt("calibration_acqui", calibration_dtilt),
                        "dtilt": float(calibration_dtilt),
                        "kind": "calibration",
                        "calibration_index": calibration_index,
                    })
                for normal_index, dtilt in enumerate(normal_dtilts[1:], start=1):
                    acquisition_plan.append({
                        "group_name": f"Aquistion_{normal_index}",
                        "dtilt": float(dtilt),
                        "kind": "normal",
                        "normal_index": normal_index,
                    })
            measstart = time.time()
            acquisitions_written = 0
            normal_acquisitions_written = 0
            calibration_acquisitions_written = 0
            stopped = False

            with h5py.File(filepath, "w") as h5:
                meas = h5.create_group("Measurement_Metadata")
                meas.attrs["Latitude"] = str(config["latitude"])
                meas.attrs["Longitude"] = str(config["longitude"])
                meas.attrs["Pan_Offset"] = str(config["pan_offset"])
                meas.attrs["Tilt_Offset"] = str(config["tilt_offset"])
                meas.attrs["Location"] = config["location"]
                meas.attrs["Trigger Target Altitude [deg]"] = np.nan if target_altitude is None else float(target_altitude)
                meas.attrs["Trigger Actual Altitude [deg]"] = np.nan if actual_altitude is None else float(actual_altitude)
                meas.attrs["Trigger Azimuth [deg]"] = np.nan if trigger_azimuth is None else float(trigger_azimuth)
                meas.attrs["Trigger Timestamp"] = trigger_dt.isoformat(timespec="seconds")
                meas.attrs["Scan Initial Sun Altitude [deg]"] = float(initial_sun_altitude)
                meas.attrs["Scan Start Tilt Offset [deg]"] = float(config["start_tilt"])
                meas.attrs["Scan Computed End Tilt Offset [deg]"] = float(end_tilt)
                meas.attrs["Scan Step Tilt [deg]"] = float(config["step_tilt"])
                meas.attrs["Calibration Delta Tilts [deg]"] = np.array(CALIBRATION_DTILTS_DEG, dtype=float)
                meas.attrs["Auto Exposure Enabled"] = int(config["auto_exposure"])
                meas.attrs["Auto Exposure Metric"] = config["auto_exposure_mode"]
                meas.attrs["Auto Exposure Target Median"] = float(config["target_median"])
                meas.attrs["Auto Exposure Initial Test [us]"] = AUTO_EXPOSURE_TEST_US
                meas.attrs["Auto Exposure Fixed Near Sun [us]"] = AUTO_EXPOSURE_FIXED_NEAR_SUN_US
                meas.attrs["Auto Exposure Min [us]"] = AUTO_EXPOSURE_MIN_US
                meas.attrs["Auto Exposure Max [us]"] = float(config["auto_exposure_max_us"])
                meas.attrs["UV Image Width [px]"] = UV_IMAGE_WIDTH_PX
                meas.attrs["UV Image Height [px]"] = UV_IMAGE_HEIGHT_PX
                meas.attrs["UV Pixel Scale [deg/pixel]"] = UV_DEG_PER_PIXEL
                meas.attrs["Moog Command Resolution [deg]"] = MOOG_COMMAND_RESOLUTION_DEG
                meas.attrs["Moog Settle Before Capture [s]"] = MOOG_SETTLE_BEFORE_CAPTURE_SEC

                for plan_index, acquisition in enumerate(acquisition_plan):
                    if self.stop_acquisition_event.is_set():
                        stopped = True
                        break

                    group_name = acquisition["group_name"]
                    dtilt = float(acquisition["dtilt"])
                    acquisition_kind = acquisition["kind"]
                    dt = datetime.now().astimezone()
                    utc_dt = dt.astimezone(timezone.utc)
                    sun_azimuth, sun_altitude = solar_position_deg(dt, config["latitude"], config["longitude"])
                    pan = azimuth_to_moog_pan(sun_azimuth)
                    tilt = sun_altitude + float(dtilt)
                    requested_pan = pan - config["pan_offset"]
                    requested_tilt = tilt - config["tilt_offset"]
                    target_pan, target_tilt = moog_command_pointing(requested_pan, requested_tilt)

                    with self.motion_lock:
                        moog_status = self.moog.move_absolute(target_pan, target_tilt)
                        self.current_status = moog_status
                    self.result_queue.put((
                        "auto_status",
                        (
                            f"Moog reached {group_name}: pan={moog_status.pan_deg:.2f}, "
                            f"tilt={moog_status.tilt_deg:.2f}; settling "
                            f"{MOOG_SETTLE_BEFORE_CAPTURE_SEC:.1f}s before capture"
                        ),
                        None,
                    ))
                    if self.stop_acquisition_event.wait(MOOG_SETTLE_BEFORE_CAPTURE_SEC):
                        stopped = True
                        break

                    exposure_info = None
                    if config["auto_exposure"]:
                        if dtilt < 3:
                            with self.camera_lock:
                                self.camera.set_exposure(AUTO_EXPOSURE_FIXED_NEAR_SUN_US)
                            exposure_info = f"Fixed exposure for dtilt<3: {AUTO_EXPOSURE_FIXED_NEAR_SUN_US:.0f} us"
                        else:
                            exposure_info = self.auto_exposure_all_angles(
                                angles,
                                moog_status.pan_deg,
                                moog_status.tilt_deg,
                                target_median=config["target_median"],
                                metric=config["auto_exposure_mode"],
                                max_exposure_us=config["auto_exposure_max_us"],
                            )
                    with self.camera_lock:
                        capture_exposure_us = float(self.camera.get_exposure())

                    uv_frames = []
                    capture_start = time.time()
                    for angle in angles:
                        with self.motion_lock:
                            self.polarizer.move_absolute(angle)
                        time.sleep(0.05)
                        with self.camera_lock:
                            self.camera.set_exposure(capture_exposure_us)
                            self.update_sim_camera_site()
                            frame = self.camera.get_frame(moog_status.pan_deg, moog_status.tilt_deg)
                        uv_frames.append(np.asarray(frame, dtype=np.uint16))
                    uvmeastime = time.time() - capture_start

                    aq = h5.create_group(group_name)
                    aq.attrs["Acquisition Type"] = acquisition_kind
                    if acquisition_kind == "normal":
                        aq.attrs["Normal Acquisition Index"] = int(acquisition["normal_index"])
                    else:
                        aq.attrs["Calibration Acquisition Index"] = int(acquisition["calibration_index"])
                        aq.attrs["Calibration Direction"] = "up" if dtilt > 0 else "down" if dtilt < 0 else "center"
                        aq.attrs["Calibration Step Magnitude [deg]"] = abs(float(dtilt))
                    aq.attrs["Timestamp Local"] = dt.strftime("%H_%M_%S")
                    aq.attrs["Timestamp Local New"] = dt.isoformat(timespec="microseconds")
                    aq.attrs["Timestamp UTC New"] = utc_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
                    aq.attrs["Timestamp Local Time Zone New"] = dt.tzname() or ""
                    aq.attrs["Timestamp Local UTC Offset New"] = dt.strftime("%z")
                    aq.attrs["Pan"] = pan
                    aq.attrs["Tilt"] = tilt
                    aq.attrs["Pan Offset"] = config["pan_offset"]
                    aq.attrs["Tilt Offset"] = config["tilt_offset"]
                    aq.attrs["Delta Tilt From Sun [deg]"] = float(dtilt)
                    aq.attrs["Sun Position Azimuth"] = sun_azimuth
                    aq.attrs["Sun Position Altitude"] = sun_altitude
                    aq.attrs["Moog Requested Pan [deg]"] = requested_pan
                    aq.attrs["Moog Requested Tilt [deg]"] = requested_tilt
                    write_pointing_attrs(aq.attrs, moog_status, target_pan, target_tilt)

                    uvimg = aq.create_group("UV Image Data")
                    uv_stack = np.stack(uv_frames, axis=0)
                    uvimg.create_dataset("UV Raw Images", data=uv_stack, compression="gzip")
                    uvimg.attrs["UV Exposure Time"] = float(capture_exposure_us)
                    uvimg.attrs["UV Auto Exposure Enabled"] = int(config["auto_exposure"])
                    uvimg.attrs["UV Auto Exposure Metric"] = config["auto_exposure_mode"]
                    uvimg.attrs["UV Auto Exposure Info"] = "" if exposure_info is None else exposure_info
                    uvimg.attrs["UV Auto Exposure Min [us]"] = AUTO_EXPOSURE_MIN_US
                    uvimg.attrs["UV Auto Exposure Max [us]"] = float(config["auto_exposure_max_us"])
                    uvimg.attrs["UV Bandpass"] = config["uv_wavelength"]
                    uvimg.attrs["UV Image Capture Time"] = uvmeastime
                    uvimg.attrs["UV Polarizer Angles"] = str(angles)
                    uvimg.attrs["UV Image Shape"] = str(uv_stack.shape)
                    acquisitions_written += 1
                    if acquisition_kind == "normal":
                        normal_acquisitions_written += 1
                    else:
                        calibration_acquisitions_written += 1

                    if group_name == "Aquistion_0":
                        preview_frames = [frame.copy() for frame in uv_frames[:4]]
                        self.result_queue.put((
                            "auto_preview",
                            {
                                "frames": preview_frames,
                                "angles": list(angles[:len(preview_frames)]),
                                "timestamp": dt.strftime("%H:%M:%S"),
                                "sun_altitude": float(sun_altitude),
                                "sun_zenith": float(90.0 - sun_altitude),
                                "actual_pan": float(moog_status.pan_deg),
                                "actual_tilt": float(moog_status.tilt_deg),
                                "filepath": filepath,
                            },
                            None,
                        ))

                    self.result_queue.put((
                        "auto_progress",
                        {
                            "aq_num": plan_index,
                            "total": len(acquisition_plan),
                            "group_name": group_name,
                            "acquisition_type": acquisition_kind,
                            "dtilt": float(dtilt),
                            "filepath": filepath,
                            "status": moog_status,
                        },
                        None,
                    ))

                if self.stop_acquisition_event.is_set():
                    stopped = True
                meas.attrs["Total Measurement Time"] = (time.time() - measstart) / 60.0
                meas.attrs["Acquisition Stopped"] = int(stopped)
                meas.attrs["Completed Acquisitions"] = acquisitions_written
                meas.attrs["Completed Normal Acquisitions"] = normal_acquisitions_written
                meas.attrs["Completed Calibration Acquisitions"] = calibration_acquisitions_written

            return {
                "filepath": filepath,
                "acquisitions": acquisitions_written,
                "planned_acquisitions": len(acquisition_plan),
                "normal_acquisitions": normal_acquisitions_written,
                "calibration_acquisitions": calibration_acquisitions_written,
                "stopped": stopped,
                "target_altitude": target_altitude,
                "actual_altitude": actual_altitude,
                "pan_offset": config["pan_offset"],
                "tilt_offset": config["tilt_offset"],
            }

        def submit_worker(self, kind, fn, *args, **kwargs):
            def run():
                try:
                    result = fn(*args, **kwargs)
                    self.result_queue.put((kind, result, None))
                except Exception as exc:
                    self.result_queue.put((kind, None, exc))

            self.executor.submit(run)

        def pause_preview_for_auto_scan(self):
            if self.timer.isActive():
                self.resume_preview_after_scan = True
                self.timer.stop()

        def resume_preview_after_auto_scan(self):
            if self.resume_preview_after_scan and bool(getattr(self.camera, "connected", False)):
                self.timer.start()
            self.resume_preview_after_scan = False

        def process_worker_results(self):
            while True:
                try:
                    kind, result, error = self.result_queue.get_nowait()
                except queue.Empty:
                    break

                if kind == "frame":
                    self.camera_pending = False
                elif kind == "status":
                    self.status_pending = False

                if error is not None:
                    if kind == "frame":
                        self.frame_error_count += 1
                        self.camera_retry_until = time.time() + PREVIEW_FRAME_ERROR_BACKOFF_SEC
                        self.exposure_status.setText("Preview frame timeout; retrying")
                        if self.frame_error_count == 1 or self.frame_error_count % 5 == 0:
                            self.log_msg(self.camera_timeout_context(error))
                    elif kind == "auto_trigger":
                        self.auto_job_running = False
                        self.auto_stop_requested = False
                        self.resume_preview_after_auto_scan()
                        self.set_auto_scan_button_state("running" if self.auto_monitoring else "idle")
                    elif kind == "moog_open":
                        self.set_moog_home_enabled(False)
                        self.set_switch_pair("moog_switch", False)
                    elif kind == "moog_close":
                        self.set_switch_pair("moog_switch", True)
                    elif kind == "status":
                        self.set_moog_home_enabled(False)
                        self.set_switch_pair("moog_switch", False)
                    elif kind == "camera_open":
                        self.timer.stop()
                        self.set_switch_pair("camera_switch", False)
                        self.camera_view.clear()
                        self.camera_view.setText("Camera closed")
                    elif kind == "camera_close":
                        self.set_switch_pair("camera_switch", True)
                    elif kind == "polarizer_open":
                        self.set_switch_pair("pol_switch", False)
                    elif kind == "polarizer_close":
                        self.set_switch_pair("pol_switch", True)
                    self.log_msg(f"{kind} failed: {error}")
                    continue

                if kind == "frame":
                    frame, center, exposure_info = result
                    self.frame_error_count = 0
                    self.camera_retry_until = 0.0
                    self.last_frame = frame
                    self.last_center = center
                    self.camera_view.set_frame(frame, center)
                    if exposure_info:
                        self.exposure_status.setText(exposure_info)
                    self.apply_status()
                elif kind == "motion":
                    message, status = result
                    self.current_status = status
                    self.log_msg(message)
                    self.apply_status()
                elif kind == "status":
                    self.current_status = result
                    self.log_msg(
                        f"Position refreshed: pan={result.pan_deg:.2f} deg, tilt={result.tilt_deg:.2f} deg"
                    )
                    self.apply_status()
                elif kind == "moog_open":
                    message, status = result
                    self.current_status = status
                    self.set_switch_pair("moog_switch", bool(status.connected))
                    self.set_moog_home_enabled(bool(status.connected))
                    self.log_msg(message if status.connected else "Moog open failed: controller did not report connected")
                    self.apply_status()
                elif kind == "moog_close":
                    message, status = result
                    self.current_status = status
                    self.set_moog_home_enabled(False)
                    self.set_switch_pair("moog_switch", bool(status.connected))
                    self.log_msg(message)
                    self.apply_status()
                elif kind == "moog_home":
                    message, status = result
                    self.current_status = status
                    self.log_msg(message)
                    self.apply_status()
                elif kind == "camera_open":
                    message, connected = result
                    self.set_switch_pair("camera_switch", bool(connected))
                    if connected:
                        self.timer.start()
                    else:
                        self.timer.stop()
                        self.camera_view.clear()
                        self.camera_view.setText("Camera closed")
                    self.log_msg(message if connected else "Camera open failed: controller did not report connected")
                    self.apply_status()
                elif kind == "camera_close":
                    message, connected = result
                    self.set_switch_pair("camera_switch", bool(connected))
                    self.log_msg(message)
                    self.apply_status()
                elif kind == "camera":
                    self.log_msg(result)
                    self.apply_status()
                elif kind == "polarizer_open":
                    message, connected = result
                    self.set_switch_pair("pol_switch", bool(connected))
                    self.log_msg(message if connected else "Polarizer open failed: controller did not report connected")
                    self.apply_status()
                elif kind == "polarizer_close":
                    message, connected = result
                    self.set_switch_pair("pol_switch", bool(connected))
                    self.log_msg(message)
                    self.apply_status()
                elif kind == "center":
                    self.last_center_result = result
                    if result.get("ok"):
                        self.pan_offset = float(result["pan_offset_deg"])
                        self.save_auto_scan_settings()
                    self.update_auto_calibration_labels()
                    if result.get("ok"):
                        self.apply_status()
                        self.log_msg(
                            "Centered sun: "
                            f"sun pan={result['sun_pan_deg']:.2f}, "
                            f"Moog pan={result['centered_pan_deg']:.2f}, "
                            f"pan_offset={result['pan_offset_deg']:.2f}, "
                            f"tilt={result['centered_tilt_deg']:.2f}, "
                            f"correction dpan={result['applied_dpan_deg']:.4f}, dtilt={result['applied_dtilt_deg']:.4f}, "
                            f"pan direction={result['pan_direction']}"
                        )
                    else:
                        self.log_msg(self.center_failure_message(result.get("info", {})))
                        self.apply_status()
                elif kind == "stop":
                    self.set_moog_home_enabled(False)
                    self.auto_job_running = False
                    self.auto_stop_requested = False
                    self.set_auto_scan_button_state("idle")
                    self.log_msg(result)
                    self.apply_status()
                elif kind == "auto_trigger":
                    self.auto_job_running = False
                    if result.get("stopped"):
                        self.queued_sun_trigger = None
                        self.auto_stop_requested = False
                        self.resume_preview_after_auto_scan()
                        self.set_auto_scan_button_state("idle")
                        self.log_msg(
                            "Auto scan stopped: "
                            f"{result['acquisitions']}/{result['planned_acquisitions']} acquisitions saved to {result['filepath']}"
                        )
                    else:
                        self.log_msg(
                            "Auto scan complete: "
                            f"{result['acquisitions']} acquisitions saved to {result['filepath']}"
                        )
                        queued_trigger = self.queued_sun_trigger
                        self.queued_sun_trigger = None
                        if self.auto_monitoring and queued_trigger is not None:
                            self.log_msg(
                                "Starting queued sun-trigger scan: "
                                f"target={queued_trigger['target_altitude']:.2f}, "
                                f"actual={queued_trigger['actual_altitude']:.2f}, "
                                f"azimuth={queued_trigger['azimuth']:.2f}"
                            )
                            self.trigger_auto_measurement(
                                queued_trigger["target_altitude"],
                                queued_trigger["actual_altitude"],
                                queued_trigger["azimuth"],
                                queued_trigger["dt"],
                            )
                        else:
                            self.auto_stop_requested = False
                            self.resume_preview_after_auto_scan()
                            self.set_auto_scan_button_state("running" if self.auto_monitoring else "idle")
                elif kind == "auto_progress":
                    self.current_status = result["status"]
                    self.apply_status()
                    self.log_msg(
                        f"Auto scan {result['group_name']} ({result['aq_num'] + 1}/{result['total']}, "
                        f"{result['acquisition_type']}), dtilt={result['dtilt']:.2f}, file={result['filepath']}"
                    )
                elif kind == "auto_status":
                    self.log_msg(result)
                elif kind == "auto_preview":
                    frames = result.get("frames", [])
                    angles = result.get("angles", [])
                    timestamp = result.get("timestamp", "")
                    sun_altitude = result.get("sun_altitude", float("nan"))
                    sun_zenith = result.get("sun_zenith", 90.0 - sun_altitude)
                    actual_pan = result.get("actual_pan", float("nan"))
                    actual_tilt = result.get("actual_tilt", float("nan"))
                    for idx, label in enumerate(self.auto_preview_labels):
                        if idx < len(frames):
                            angle_text = f"{angles[idx]:g} deg" if idx < len(angles) else f"Image {idx + 1}"
                            label.set_frame(
                                frames[idx],
                                (
                                    f"Pol {angle_text}\n"
                                    f"{timestamp}, sun zen={sun_zenith:.2f} deg\n"
                                    f"sun alt={sun_altitude:.2f} deg\n"
                                    f"actual pan={actual_pan:.2f}, tilt={actual_tilt:.2f}"
                                ),
                            )
                        else:
                            label.clear_frame()
                    self.log_msg(f"Updated acquisition 0 preview: {len(frames)} image(s)")

        def set_simulation(self, checked):
            for checkbox_name in ("sim_check", "auto_sim_check"):
                checkbox = getattr(self, checkbox_name, None)
                if checkbox is not None and checkbox.isChecked() != checked:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(checked)
                    checkbox.blockSignals(False)
            self.simulation = checked
            self.close_camera()
            self.close_moog()
            self.close_polarizer()
            self.moog = SimMoogController() if checked else RealMoogController()
            self.polarizer = SimPolarizerController() if checked else RealPolarizerController()
            self.camera = SimCameraController() if checked else RealVmbCameraController()
            if isinstance(self.camera, SimCameraController):
                self.update_sim_camera_site()
            self.current_status = self.moog.get_status()
            if checked:
                self.log_msg(
                    "Simulation mode enabled: "
                    "pan=0 points south, "
                    f"sim pan mount offset={self.camera.pan_mount_offset_deg:+.2f} deg"
                )
            else:
                self.log_msg("Hardware mode enabled")
            self.apply_status()

        def refresh_ports(self):
            ports = [DEFAULT_MOOG_PORT, DEFAULT_ZABER_PORT, "SIM"]
            if serial is not None:
                ports += [p.device for p in serial.tools.list_ports.comports()]
            ports = list(dict.fromkeys(ports))
            port_combos = (
                (self.moog_port, DEFAULT_MOOG_PORT),
                (self.pol_port, DEFAULT_ZABER_PORT),
                (self.auto_moog_port, DEFAULT_MOOG_PORT),
                (self.auto_pol_port, DEFAULT_ZABER_PORT),
            )
            for combo, default_port in port_combos:
                current = combo.currentText()
                combo.clear()
                combo.addItems(ports)
                if current in ports:
                    combo.setCurrentText(current)
                else:
                    combo.setCurrentText(default_port)

        def selected_moog_port(self):
            if self.pages.currentIndex() == 0:
                return self.auto_moog_port.currentText()
            return self.moog_port.currentText()

        def selected_pol_port(self):
            if self.pages.currentIndex() == 0:
                return self.auto_pol_port.currentText()
            return self.pol_port.currentText()

        def set_switch_pair(self, name, checked):
            for switch_name in (name, f"auto_{name}"):
                switch = getattr(self, switch_name, None)
                if switch is not None and switch.isChecked() != checked:
                    switch.blockSignals(True)
                    switch.setChecked(checked)
                    switch.blockSignals(False)

        def set_moog_home_enabled(self, enabled):
            for button_name in (
                "home_btn",
                "auto_home_btn",
                "refresh_position_btn",
                "auto_refresh_position_btn",
                "aim_sun_btn",
            ):
                button = getattr(self, button_name, None)
                if button is not None:
                    button.setEnabled(bool(enabled))

        def toggle_moog(self, checked):
            self.set_switch_pair("moog_switch", checked)
            if checked:
                self.open_moog()
            else:
                self.close_moog()

        def toggle_camera(self, checked):
            self.set_switch_pair("camera_switch", checked)
            if checked:
                self.open_camera()
            else:
                self.close_camera()

        def toggle_polarizer(self, checked):
            self.set_switch_pair("pol_switch", checked)
            if checked:
                self.open_polarizer()
            else:
                self.close_polarizer()

        def open_moog(self):
            port = self.selected_moog_port()
            self.log_msg(f"Opening Moog on {port}...")

            def task():
                with self.motion_lock:
                    status = self.moog.open(port)
                return "Moog opened", status

            self.submit_worker("moog_open", task)

        def request_moog_status(self):
            if self.status_pending:
                return
            if not self.current_status.connected:
                self.log_msg("Cannot refresh position: Moog is not open")
                return
            self.status_pending = True
            self.log_msg("Refreshing Moog position...")

            def task():
                with self.motion_lock:
                    return self.moog.get_status()

            self.submit_worker("status", task)

        def home_moog(self):
            if self.auto_job_running:
                self.log_msg("Cannot reset Moog while acquisition is running")
                return
            self.log_msg("Returning Moog to zero: pan first, then tilt...")

            def task():
                with self.motion_lock:
                    status = self.moog.home()
                    self.current_status = status
                return "Moog returned to zero: pan then tilt completed", status

            self.submit_worker("moog_home", task)

        def close_moog(self):
            self.log_msg("Closing Moog: returning pan to zero, then tilt to zero...")
            self.set_moog_home_enabled(False)

            def task():
                with self.motion_lock:
                    self.moog.close()
                    status = self.moog.get_status()
                return "Moog returned to zero in order and closed", status

            self.submit_worker("moog_close", task)

        def open_camera(self):
            exposure_us = self.exposure_us.value()
            self.log_msg("Opening camera...")

            def task():
                with self.camera_lock:
                    self.camera.set_exposure(exposure_us)
                    self.camera.open()
                    connected = bool(getattr(self.camera, "connected", False))
                return "Camera opened", connected

            self.submit_worker("camera_open", task)

        def close_camera(self):
            self.timer.stop()
            self.log_msg("Closing camera...")

            def task():
                with self.camera_lock:
                    self.camera.close()
                    connected = bool(getattr(self.camera, "connected", False))
                return "Camera closed", connected

            self.submit_worker("camera_close", task)
            self.camera_view.clear()
            self.camera_view.setText("Camera closed")

        def open_polarizer(self):
            port = self.selected_pol_port()
            self.log_msg(f"Opening polarizer on {port}...")

            def task():
                with self.motion_lock:
                    self.polarizer.open(port)
                    connected = bool(getattr(self.polarizer, "connected", False))
                return "Polarizer opened", connected

            self.submit_worker("polarizer_open", task)

        def close_polarizer(self):
            self.log_msg("Closing polarizer...")

            def task():
                with self.motion_lock:
                    self.polarizer.close()
                    connected = bool(getattr(self.polarizer, "connected", False))
                return "Polarizer closed", connected

            self.submit_worker("polarizer_close", task)

        def stop_all(self):
            self.stop_acquisition_event.set()
            self.auto_stop_requested = True
            self.resume_preview_after_scan = False
            self.set_auto_scan_button_state("stopping")
            if self.auto_monitoring:
                self.stop_sun_monitor()
            self.timer.stop()
            self.set_moog_home_enabled(False)
            self.set_switch_pair("camera_switch", False)
            self.set_switch_pair("pol_switch", False)
            self.set_switch_pair("moog_switch", False)
            def task():
                with self.camera_lock:
                    self.camera.close()
                with self.motion_lock:
                    self.polarizer.close()
                    self.moog.close()
                    self.current_status = self.moog.get_status()
                return "STOP completed: camera, Moog, and polarizer are closed"

            self.submit_worker("stop", task)
            self.camera_view.clear()
            self.camera_view.setText("Camera closed")

        def set_auto_exposure(self, checked):
            self.exposure_us.setEnabled(not checked)
            mode = "Auto" if checked else "Manual"
            self.exposure_status.setText(f"{mode}: {self.camera.get_exposure():.0f} us")
            self.apply_status()

        def set_auto_exposure_mode(self, mode):
            self.auto_exposure_mode = str(mode).strip().lower()
            if self.auto_exposure_mode not in ("median", "max"):
                self.auto_exposure_mode = "median"
            metric = getattr(self, "auto_exposure_metric", None)
            if metric is not None and metric.currentText().lower() != self.auto_exposure_mode:
                metric.blockSignals(True)
                metric.setCurrentText(self.auto_exposure_mode.title())
                metric.blockSignals(False)
            self.update_settings_summary()
            self.log_msg(f"Auto exposure metric set to {self.auto_exposure_mode}")

        def set_auto_exposure_max(self, exposure_us):
            self.auto_exposure_max_us = float(exposure_us)
            self.exposure_us.setMaximum(self.auto_exposure_max_us)
            if self.exposure_us.value() > self.auto_exposure_max_us:
                self.exposure_us.setValue(self.auto_exposure_max_us)
            max_widget = getattr(self, "max_exposure_us", None)
            if max_widget is not None and abs(max_widget.value() - self.auto_exposure_max_us) > 0.5:
                max_widget.blockSignals(True)
                max_widget.setValue(self.auto_exposure_max_us)
                max_widget.blockSignals(False)
            self.update_settings_summary()
            self.log_msg(f"Auto exposure max set to {self.auto_exposure_max_us:.0f} us")

        def set_manual_exposure(self, exposure_us):
            if self.auto_exposure_check.isChecked():
                return
            if self.auto_job_running:
                self.log_msg("Manual exposure ignored while Auto Scan is running")
                return
            def task():
                with self.camera_lock:
                    self.camera.set_exposure(exposure_us)
                    exposure = self.camera.get_exposure()
                return f"Exposure set: {exposure:.0f} us"

            self.exposure_status.setText(f"Manual: {float(exposure_us):.0f} us")
            self.submit_worker("camera", task)

        def auto_exposure_all_angles(
            self,
            angles,
            pan_deg,
            tilt_deg,
            target_median=AUTO_EXPOSURE_TARGET_MEDIAN,
            metric="median",
            max_exposure_us=None,
        ):
            max_pixel_value = 2 ** UV_BIT_DEPTH - 1
            saturation_limit = max_pixel_value * AUTO_EXPOSURE_SATURATION_FRACTION
            metric = str(metric).strip().lower()
            if metric not in ("median", "max"):
                metric = "median"
            exposure_limit = float(self.auto_exposure_max_us if max_exposure_us is None else max_exposure_us)

            with self.camera_lock:
                self.camera.set_exposure(AUTO_EXPOSURE_TEST_US)

            medians = []
            maxima = []
            saturated = []
            for angle in angles:
                with self.motion_lock:
                    self.polarizer.move_absolute(angle)
                time.sleep(0.5)

                with self.camera_lock:
                    self.update_sim_camera_site()
                    frame = self.camera.get_frame(pan_deg, tilt_deg)
                data = np.asarray(frame, dtype=np.uint16)
                med = float(np.median(data))
                max_value = float(np.max(data))
                medians.append(med)
                maxima.append(max_value)
                saturated.append(max_value >= saturation_limit)

            medians_array = np.asarray(medians, dtype=float)
            maxima_array = np.asarray(maxima, dtype=float)
            avg_median = float(np.mean(medians_array))
            avg_max = float(np.mean(maxima_array))

            if all(saturated):
                new_exposure = AUTO_EXPOSURE_MIN_US
                reason = "all test angles saturated"
            elif metric == "max":
                if avg_max == 0:
                    new_exposure = AUTO_EXPOSURE_MIN_US
                    reason = "zero max"
                else:
                    scale = saturation_limit / avg_max
                    new_exposure = float(np.clip(
                        AUTO_EXPOSURE_TEST_US * scale,
                        AUTO_EXPOSURE_MIN_US,
                        exposure_limit,
                    ))
                    reason = "adaptive max"
            elif avg_median == 0:
                new_exposure = AUTO_EXPOSURE_MIN_US
                reason = "zero median"
            else:
                scale = float(target_median) / avg_median
                new_exposure = float(np.clip(
                    AUTO_EXPOSURE_TEST_US * scale,
                    AUTO_EXPOSURE_MIN_US,
                    exposure_limit,
                ))
                reason = "adaptive median"

            with self.camera_lock:
                self.camera.set_exposure(new_exposure)

            return (
                f"Auto exposure all angles: medians={medians_array.astype(int).tolist()}, "
                f"maxima={maxima_array.astype(int).tolist()}, metric={metric}, "
                f"target median={float(target_median):.0f}, target max={saturation_limit:.0f}, "
                f"exposure={new_exposure:.0f} us, limit={exposure_limit:.0f} us, {reason}"
            )

        def calculate_auto_exposure(self, frame, target_median=None):
            image = np.asarray(frame, dtype=np.float32)
            median = float(np.median(image))
            image_max = float(np.max(image))
            max_value = float(2 ** UV_BIT_DEPTH - 1)
            target_max = max_value * AUTO_EXPOSURE_SATURATION_FRACTION
            saturated_fraction = float(np.mean(image >= max_value * AUTO_EXPOSURE_SATURATION_FRACTION))
            current = float(self.camera.get_exposure())

            if saturated_fraction > 0.01:
                proposed = current * 0.5
            elif self.auto_exposure_mode == "max":
                if image_max <= 0:
                    proposed = AUTO_EXPOSURE_MIN_US
                else:
                    proposed = current * np.clip(target_max / image_max, 0.25, 4.0)
            elif median <= 0:
                proposed = AUTO_EXPOSURE_MIN_US
            else:
                scale = (self.target_median.value() if target_median is None else target_median) / median
                proposed = current * np.clip(scale, 0.25, 4.0)

            new_exposure = float(np.clip(proposed, AUTO_EXPOSURE_MIN_US, self.auto_exposure_max_us))
            if abs(new_exposure - current) / max(current, 1.0) > 0.02:
                self.camera.set_exposure(new_exposure)

            return (
                f"Auto {self.auto_exposure_mode}: {self.camera.get_exposure():.0f} us, "
                f"median={median:.0f}, max={image_max:.0f}, sat={saturated_fraction * 100:.2f}%"
            )

        def request_frame_update(self):
            if self.auto_job_running:
                return
            if self.camera_pending:
                return
            if time.time() < self.camera_retry_until:
                return
            self.camera_pending = True
            self.preview_frame_counter += 1
            auto_exposure = self.auto_exposure_check.isChecked()
            run_auto_exposure = auto_exposure and self.preview_frame_counter % PREVIEW_AUTO_EXPOSURE_INTERVAL == 0
            run_center_detection = self.preview_frame_counter % PREVIEW_CENTER_DETECT_INTERVAL == 0
            pan_deg = self.current_status.pan_deg
            tilt_deg = self.current_status.tilt_deg
            previous_center = self.last_center

            def task():
                with self.camera_lock:
                    self.update_sim_camera_site()
                    frame = self.camera.get_frame(pan_deg, tilt_deg)
                    exposure_info = None
                    if run_auto_exposure and not self.auto_job_running:
                        exposure_info = self.calculate_auto_exposure(frame)
                if run_center_detection:
                    stride = max(1, int(math.ceil(max(frame.shape[:2]) / float(PREVIEW_CENTER_MAX_SIDE_PX))))
                    center_small, _ = detect_sun_center(frame[::stride, ::stride])
                    center = None if center_small is None else (
                        float(center_small[0]) * stride,
                        float(center_small[1]) * stride,
                    )
                else:
                    center = previous_center
                return frame, center, exposure_info

            self.submit_worker("frame", task)

        def apply_status(self):
            status = self.current_status
            self.pan_display.set_value(status.pan_deg)
            self.tilt_display.set_value(status.tilt_deg)
            center_line = "Sun center: none"
            if self.last_center_result:
                if self.last_center_result.get("ok"):
                    center_line = (
                        "Sun center: "
                        f"x={self.last_center_result['center_x']:.1f}, "
                        f"y={self.last_center_result['center_y']:.1f}, "
                        f"centered pan={self.last_center_result['centered_pan_deg']:.2f} deg, "
                        f"centered tilt={self.last_center_result['centered_tilt_deg']:.2f} deg"
                    )
                else:
                    center_line = f"Sun center: failed {self.last_center_result.get('info', {})}"
            self.current_output_text = (
                "Calibration output\n"
                f"Pan: {status.pan_deg:.2f} deg\n"
                f"Tilt: {status.tilt_deg:.2f} deg\n"
                f"Polarizer: {float(getattr(self.polarizer, 'angle_deg', 0.0)):.2f} deg\n"
                f"Exposure: {self.camera.get_exposure():.0f} us "
                f"({'auto' if self.auto_exposure_check.isChecked() else 'manual'})\n"
                f"Pixel scale: {self.deg_per_pixel.value():.6f} deg/pixel "
                f"({self.deg_per_pixel.value() * 3600.0:.2f} arcsec/pixel)\n"
                f"{center_line}"
            )

        def jog(self, dpan, dtilt):
            self.clear_center_result_for_reposition()

            def task():
                with self.motion_lock:
                    status = self.moog.move_relative(dpan, dtilt)
                    self.current_status = status
                return f"Moved pan {dpan:+.2f}, tilt {dtilt:+.2f}", status

            self.submit_worker("motion", task)

        def clear_center_result_for_reposition(self):
            if self.last_center_result and self.last_center_result.get("ok"):
                self.last_center_result = None
                self.update_auto_calibration_labels()
                self.log_msg("Previous sun-center result cleared after pointing change")

        def aim_at_calculated_sun(self):
            if not self.current_status.connected:
                self.log_msg("Cannot aim at calculated sun: Moog is not open")
                return

            self.clear_center_result_for_reposition()
            dt = datetime.now()
            sun_azimuth, sun_tilt = solar_position_deg(
                dt,
                self.latitude_input.value(),
                self.longitude_input.value(),
            )
            sun_pan = azimuth_to_moog_pan(sun_azimuth)
            target_pan, target_tilt = moog_command_pointing(
                sun_pan,
                sun_tilt,
            )
            self.log_msg(
                "Aiming at calculated sun: "
                f"sun az={sun_azimuth:.2f}, sun pan={sun_pan:.2f}, sun tilt={sun_tilt:.2f}, "
                "using no calibration offset, "
                f"target Moog pan={target_pan:.2f}, tilt={target_tilt:.2f}"
            )

            def task():
                with self.motion_lock:
                    status = self.moog.move_absolute(target_pan, target_tilt)
                    self.current_status = status
                return (
                    "Calculated sun aim complete: "
                    f"actual Moog pan={status.pan_deg:.2f}, tilt={status.tilt_deg:.2f}; "
                    "adjust manually or use Center sun in FOV",
                    status,
                )

            self.submit_worker("motion", task)

        def set_polarizer(self, angle):
            def task():
                with self.motion_lock:
                    self.polarizer.move_absolute(angle)
                return f"Polarizer angle set to {angle:g} deg"

            self.submit_worker("polarizer", task)

        def center_sun(self):
            pan_direction = (
                "+pan moves sun right"
                if self.pan_positive_moves_right_check.isChecked()
                else "+pan moves sun left"
            )
            self.log_msg(f"Center sun requested: searching current camera frame; pan direction={pan_direction}")
            self.remote_center()
            self.apply_status()

        def center_failure_message(self, info):
            reason = info.get("reason", "unknown detection failure")
            messages = {
                "camera not open": "Sun centering failed: camera is not open",
                "empty frame": "Sun centering failed: camera frame is empty",
                "low contrast": "Sun centering failed: image contrast is too low; no solar disk detected",
                "no bright component": "Sun centering failed: no sun-like bright region detected in the frame",
                "component too small": "Sun centering failed: detected bright region is too small to identify as the sun",
                "no bright pixels": "Sun centering failed: no bright solar pixels detected in the frame",
                "pan probe failed": "Sun centering failed: automatic pan direction detection did not produce a usable image shift",
            }
            message = messages.get(reason, f"Sun centering failed: {reason}")
            if "area" in info:
                message += f" (area={info['area']} px)"
            if "detail" in info:
                message += f" ({info['detail']})"
            return message

        def remote_status(self):
            status = self.current_status
            return {
                "simulation": self.simulation,
                "pan_deg": status.pan_deg,
                "tilt_deg": status.tilt_deg,
                "moog_connected": status.connected,
                "camera_connected": bool(getattr(self.camera, "connected", False)),
                "camera_exposure_us": float(self.camera.get_exposure()),
                "auto_exposure": bool(self.auto_exposure_check.isChecked()),
                "auto_exposure_mode": self.auto_exposure_mode,
                "auto_exposure_target_median": float(self.target_median.value()),
                "auto_exposure_min_us": AUTO_EXPOSURE_MIN_US,
                "auto_exposure_max_us": float(self.auto_exposure_max_us),
                "polarizer_angle_deg": float(getattr(self.polarizer, "angle_deg", 0.0)),
                "deg_per_pixel": self.deg_per_pixel.value(),
                "center_pan_positive_moves_right": bool(self.pan_positive_moves_right_check.isChecked()),
                "last_center": self.last_center_result,
            }

        def remote_move(self, dpan, dtilt):
            self.jog(dpan, dtilt)
            status = self.remote_status()
            status["accepted"] = True
            return status

        def remote_goto(self, pan, tilt):
            def task():
                with self.motion_lock:
                    status = self.moog.move_absolute(pan, tilt)
                    self.current_status = status
                return f"Moved to pan={pan:.2f}, tilt={tilt:.2f}", status

            self.submit_worker("motion", task)
            status = self.remote_status()
            status["accepted"] = True
            return status

        def remote_polarizer(self, angle):
            self.set_polarizer(angle)
            return self.remote_status()

        def remote_exposure(self, auto, exposure):
            if self.auto_job_running:
                status = self.remote_status()
                status["accepted"] = False
                status["message"] = "Exposure changes are blocked while Auto Scan is running"
                return status
            if auto is not None:
                self.auto_exposure_check.setChecked(str(auto).lower() in ("1", "true", "yes", "on"))
            if exposure is not None:
                self.exposure_us.setValue(float(exposure))
                if not self.auto_exposure_check.isChecked():
                    self.set_manual_exposure(float(exposure))
            return self.remote_status()

        def remote_center(self):
            deg_per_pixel = self.deg_per_pixel.value()
            pan_positive_moves_right = bool(self.pan_positive_moves_right_check.isChecked())
            latitude = self.latitude_input.value()
            longitude = self.longitude_input.value()

            def task():
                if not bool(getattr(self.camera, "connected", False)):
                    return {"ok": False, "info": {"reason": "camera not open"}}

                with self.motion_lock:
                    status = self.moog.get_status()
                    initial_pan_deg = float(status.pan_deg)
                    initial_tilt_deg = float(status.tilt_deg)
                with self.camera_lock:
                    self.update_sim_camera_site()
                    local_frame = self.camera.get_frame(initial_pan_deg, initial_tilt_deg)

                center, info = detect_sun_center(local_frame)
                if center is None:
                    return {"ok": False, "info": info}

                height, width = local_frame.shape[:2]
                cx, cy = center
                dx = cx - width / 2.0
                dy = cy - height / 2.0
                dtilt = -dy * deg_per_pixel
                pan_pixels_per_degree = (1.0 if pan_positive_moves_right else -1.0) / deg_per_pixel
                dpan = -dx / pan_pixels_per_degree
                pan_direction = "+pan moves sun right" if pan_positive_moves_right else "+pan moves sun left"

                with self.motion_lock:
                    centered_status = self.moog.move_absolute(
                        initial_pan_deg + dpan,
                        initial_tilt_deg + dtilt,
                    )
                    self.current_status = centered_status
                centered_dt = datetime.now()
                sun_azimuth, sun_tilt = solar_position_deg(centered_dt, latitude, longitude)
                sun_pan = azimuth_to_moog_pan(sun_azimuth)
                pan_offset = quantize_pointing(sun_pan - centered_status.pan_deg)
                with self.camera_lock:
                    self.update_sim_camera_site()
                    centered_frame = self.camera.get_frame(centered_status.pan_deg, centered_status.tilt_deg)
                centered_center, _ = detect_sun_center(centered_frame)
                self.result_queue.put(("frame", (centered_frame, centered_center, None), None))
                return {
                    "ok": True,
                    "center_x": cx,
                    "center_y": cy,
                    "pixel_error_x": dx,
                    "pixel_error_y": dy,
                    "applied_dpan_deg": dpan,
                    "applied_dtilt_deg": dtilt,
                    "pan_pixels_per_degree": pan_pixels_per_degree,
                    "pan_direction": pan_direction,
                    "centered_pan_deg": centered_status.pan_deg,
                    "centered_tilt_deg": centered_status.tilt_deg,
                    "sun_azimuth_deg": sun_azimuth,
                    "sun_pan_deg": sun_pan,
                    "sun_tilt_deg": sun_tilt,
                    "pan_offset_deg": pan_offset,
                    "centered_local_time": centered_dt.isoformat(timespec="seconds"),
                    "info": info,
                }

            self.submit_worker("center", task)
            return {"accepted": True}

        def save_calibration_h5(self):
            if h5py is None:
                self.log_msg("H5 save failed: h5py is not installed")
                return

            default_name = f"ULTRASIP_pointing_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            default_path = os.path.join(os.getcwd(), default_name)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save calibration H5",
                default_path,
                "HDF5 files (*.h5 *.hdf5)",
            )
            if not path:
                return

            status = self.current_status
            try:
                with h5py.File(path, "w") as h5:
                    meta = h5.create_group("Measurement_Metadata")
                    meta.attrs["Created Local Time"] = datetime.now().isoformat(timespec="seconds")
                    meta.attrs["Mode"] = "Simulation" if self.simulation else "Hardware"
                    meta.attrs["Purpose"] = "Manual pointing calibration"
                    meta.attrs["UV Image Width [px]"] = UV_IMAGE_WIDTH_PX
                    meta.attrs["UV Image Height [px]"] = UV_IMAGE_HEIGHT_PX
                    meta.attrs["UV Full FOV [deg]"] = UV_FULL_FOV_DEG
                    meta.attrs["UV Pixel Scale [arcsec/pixel]"] = UV_ARCSEC_PER_PIXEL
                    meta.attrs["UV Pixel Scale [deg/pixel]"] = UV_DEG_PER_PIXEL
                    meta.attrs["Moog Pan Range [deg]"] = np.array([PAN_MIN_DEG, PAN_MAX_DEG])
                    meta.attrs["Moog Tilt Range [deg]"] = np.array([TILT_MIN_DEG, TILT_MAX_DEG])
                    meta.attrs["Moog Resolution [deg]"] = MOOG_RESOLUTION_DEG
                    meta.attrs["Moog Command Resolution [deg]"] = MOOG_COMMAND_RESOLUTION_DEG

                    cal = h5.create_group("Pointing_Calibration")
                    cal.attrs["Pan [deg]"] = float(status.pan_deg)
                    cal.attrs["Tilt [deg]"] = float(status.tilt_deg)
                    cal.attrs["Pan Offset [deg]"] = float(self.pan_offset)
                    if self.last_center_result and self.last_center_result.get("ok"):
                        cal.attrs["Sun Centered Pan [deg]"] = float(self.last_center_result["centered_pan_deg"])
                        cal.attrs["Sun Centered Tilt [deg]"] = float(self.last_center_result["centered_tilt_deg"])
                        cal.attrs["Calculated Sun Pan [deg]"] = float(self.last_center_result["sun_pan_deg"])
                        cal.attrs["Calculated Sun Tilt [deg]"] = float(self.last_center_result["sun_tilt_deg"])
                        cal.attrs["Measured Pan Offset [deg]"] = float(self.last_center_result["pan_offset_deg"])
                    cal.attrs["Polarizer Angle [deg]"] = float(getattr(self.polarizer, "angle_deg", 0.0))
                    cal.attrs["Camera Exposure [us]"] = float(self.camera.get_exposure())
                    cal.attrs["Auto Exposure Enabled"] = int(self.auto_exposure_check.isChecked())
                    cal.attrs["Auto Exposure Metric"] = self.auto_exposure_mode
                    cal.attrs["Auto Exposure Target Median"] = float(self.target_median.value())
                    cal.attrs["Auto Exposure Min [us]"] = AUTO_EXPOSURE_MIN_US
                    cal.attrs["Auto Exposure Max [us]"] = float(self.auto_exposure_max_us)
                    cal.attrs["Auto Exposure Saturation Fraction"] = AUTO_EXPOSURE_SATURATION_FRACTION
                    cal.attrs["Deg Per Pixel Used"] = float(self.deg_per_pixel.value())
                    cal.attrs["Arcsec Per Pixel Used"] = float(self.deg_per_pixel.value() * 3600.0)
                    cal.attrs["Sun Center Result JSON"] = json.dumps(self.last_center_result or {})
                    cal.attrs["Remote Status JSON"] = json.dumps(self.remote_status())

                    if self.last_frame is not None:
                        img = h5.create_group("Camera_Frame")
                        img.create_dataset("Latest Frame", data=self.last_frame, compression="gzip")
                        if self.last_center:
                            img.attrs["Detected Sun Center [x,y]"] = np.array(self.last_center)

                self.log_msg(f"Saved calibration H5: {path}")
            except Exception as exc:
                self.log_msg(f"H5 save failed: {exc}")

        def toggle_remote(self, checked):
            if checked:
                try:
                    self.remote_server = RemoteControlServer(self, port=self.remote_port.value())
                    self.remote_server.start()
                    self.remote_label.setText(f"http://127.0.0.1:{self.remote_port.value()}")
                    self.log_msg("Remote HTTP API started")
                except Exception as exc:
                    self.remote_check.setChecked(False)
                    self.log_msg(f"Remote API failed: {exc}")
            else:
                if self.remote_server:
                    self.remote_server.stop()
                    self.remote_server = None
                self.remote_label.setText("Stopped")
                self.log_msg("Remote HTTP API stopped")

        def closeEvent(self, event):
            self.toggle_remote(False)
            self.timer.stop()
            self.result_timer.stop()
            self.auto_monitor_timer.stop()
            try:
                with self.camera_lock:
                    self.camera.close()
                with self.motion_lock:
                    self.polarizer.close()
                    self.moog.close()
            except Exception as exc:
                self.log_msg(f"Shutdown failed: {exc}")
            self.executor.shutdown(wait=False, cancel_futures=True)
            super().closeEvent(event)

    return MainWindow


def main():
    try:
        QtCore, QtGui, QtWidgets, qt_api = import_qt()
    except ImportError as exc:
        print(exc)
        print("Example: python3 -m pip install PySide6")
        return 2

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = build_app_classes(QtCore, QtGui, QtWidgets)
    window = MainWindow()
    window.log_msg(f"Using {qt_api}")
    window.show()
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

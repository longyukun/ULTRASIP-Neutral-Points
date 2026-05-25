# -*- coding: utf-8 -*-
"""
Track the Moon with the Moog pan/tilt and UV camera.

Workflow:
1. Operator manually points the Moon into the UV camera FOV.
2. The script fits the lunar limb, not the brightness centroid, and recenters
   the Moon in the image.
3. The script follows ephemeris Moon azimuth/altitude and records target Moog
   position, actual Moog position, ephemeris Moon observation, and image-based
   residuals in HDF5.

Skyfield is the preferred ephemeris backend:
    pip install skyfield

Astropy and python-suncalc are optional fallbacks when available.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import h5py
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import serial
except ImportError:
    serial = None

try:
    from skyfield.api import Loader, wgs84
except ImportError:
    Loader = None
    wgs84 = None

try:
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time
    import astropy.units as u
except ImportError:
    AltAz = None
    EarthLocation = None
    Time = None
    get_body = None
    u = None

try:
    from suncalc import getMoonPosition as suncalc_get_moon_position
except ImportError:
    suncalc_get_moon_position = None

import moog_functions as mf

try:
    import uv_cam_functions as uv
except ImportError:
    uv = None


PAN_MIN_DEG = -217.5
PAN_MAX_DEG = 217.5
TILT_MIN_DEG = -90.0
TILT_MAX_DEG = 90.0
MOOG_RESOLUTION_DEG = 0.01
UV_IMAGE_WIDTH_PX = 2848
UV_IMAGE_HEIGHT_PX = 2848
UV_DEG_PER_PIXEL = 7.20 / 3600.0
DEFAULT_MOON_DIAMETER_DEG = 0.518


@dataclass
class MoonObservation:
    azimuth_deg: float
    altitude_deg: float
    distance_km: float
    angular_diameter_deg: float
    backend: str


@dataclass
class LimbFit:
    ok: bool
    center_x_px: float = np.nan
    center_y_px: float = np.nan
    radius_px: float = np.nan
    inlier_count: int = 0
    total_edge_count: int = 0
    residual_px: float = np.nan
    reason: str = ""


def quantize_pointing(value: float) -> float:
    return round(float(value) / MOOG_RESOLUTION_DEG) * MOOG_RESOLUTION_DEG


def clamp_pointing(pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
    pan = min(max(quantize_pointing(pan_deg), PAN_MIN_DEG), PAN_MAX_DEG)
    tilt = min(max(quantize_pointing(tilt_deg), TILT_MIN_DEG), TILT_MAX_DEG)
    return pan, tilt


def normalize_azimuth(azimuth_deg: float) -> float:
    return float(azimuth_deg) % 360.0


def wrap_to_180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


class MoonEphemeris:
    def __init__(self, skyfield_data_dir: str = "skyfield-data", ephemeris: str = "de421.bsp"):
        self.skyfield_data_dir = skyfield_data_dir
        self.ephemeris = ephemeris
        self.backend = None
        self.ts = None
        self.planets = None

        if Loader is not None:
            load = Loader(skyfield_data_dir)
            self.ts = load.timescale()
            self.planets = load(ephemeris)
            self.backend = "skyfield"
        elif get_body is not None:
            self.backend = "astropy"
        elif suncalc_get_moon_position is not None:
            self.backend = "suncalc"
        else:
            raise ImportError("Install skyfield, astropy, or python-suncalc to compute Moon azimuth/altitude.")

    def observe(self, dt: datetime, latitude_deg: float, longitude_deg: float, elevation_m: float = 0.0) -> MoonObservation:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        if self.backend == "skyfield":
            t = self.ts.from_datetime(dt.astimezone(timezone.utc))
            observer = self.planets["earth"] + wgs84.latlon(latitude_deg, longitude_deg, elevation_m=elevation_m)
            apparent = observer.at(t).observe(self.planets["moon"]).apparent()
            altitude, azimuth, distance = apparent.altaz()
            distance_km = float(distance.km)
            angular_diameter_deg = math.degrees(2.0 * math.atan2(1737.4, distance_km))
            return MoonObservation(
                azimuth_deg=normalize_azimuth(azimuth.degrees),
                altitude_deg=float(altitude.degrees),
                distance_km=distance_km,
                angular_diameter_deg=angular_diameter_deg,
                backend="skyfield",
            )

        if self.backend == "astropy":
            location = EarthLocation(lat=latitude_deg * u.deg, lon=longitude_deg * u.deg, height=elevation_m * u.m)
            obstime = Time(dt)
            frame = AltAz(obstime=obstime, location=location)
            moon_altaz = get_body("moon", obstime, location=location).transform_to(frame)
            distance_km = float(moon_altaz.distance.to(u.km).value)
            angular_diameter_deg = math.degrees(2.0 * math.atan2(1737.4, distance_km))
            return MoonObservation(
                azimuth_deg=normalize_azimuth(moon_altaz.az.deg),
                altitude_deg=float(moon_altaz.alt.deg),
                distance_km=distance_km,
                angular_diameter_deg=angular_diameter_deg,
                backend="astropy",
            )

        pos = suncalc_get_moon_position(dt, longitude_deg, latitude_deg)
        return MoonObservation(
            azimuth_deg=normalize_azimuth(np.degrees(pos["azimuth"])),
            altitude_deg=float(np.degrees(pos["altitude"])),
            distance_km=np.nan,
            angular_diameter_deg=DEFAULT_MOON_DIAMETER_DEG,
            backend="suncalc",
        )


def frame_to_image(frame) -> np.ndarray:
    data = np.frombuffer(frame.get_buffer(), dtype=np.uint16)
    size = int(math.sqrt(data.size))
    if size * size == data.size:
        return data.reshape((size, size))
    return data.reshape((UV_IMAGE_HEIGHT_PX, UV_IMAGE_WIDTH_PX))


def largest_component(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def circle_from_three_points(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    (x1, y1), (x2, y2), (x3, y3) = points
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-6:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    radius = math.hypot(cx - x1, cy - y1)
    return cx, cy, radius


def refine_circle(points: np.ndarray) -> Tuple[float, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return float(cx), float(cy), float(radius)


def fit_lunar_limb(
    image: np.ndarray,
    expected_radius_px: Optional[float] = None,
    threshold_sigma: float = 5.0,
    min_edge_points: int = 80,
    max_trials: int = 800,
    random_seed: int = 11,
) -> LimbFit:
    """Fit the outer lunar limb.

    This deliberately avoids the brightness centroid. Crescent and gibbous Moon
    images have a phase-dependent illuminated area and uneven albedo, so a
    centroid of intensity or thresholded pixels is not the disk center.
    """
    if cv2 is None:
        return LimbFit(ok=False, reason="OpenCV is required for lunar limb detection.")

    img = np.asarray(image, dtype=np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return LimbFit(ok=False, reason="Image has no finite pixels.")

    background = float(np.median(finite))
    mad = float(np.median(np.abs(finite - background)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
    thresh = max(background + threshold_sigma * sigma, float(np.percentile(finite, 99.2)) * 0.45)
    mask = img > thresh
    mask = largest_component(mask)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    ys, xs = np.nonzero(edges)
    points = np.column_stack((xs.astype(float), ys.astype(float)))

    if points.shape[0] < min_edge_points:
        return LimbFit(ok=False, total_edge_count=int(points.shape[0]), reason="Too few lunar limb edge points.")

    rng = np.random.default_rng(random_seed)
    best_inliers = None
    best_model = None
    tolerance_px = 3.0
    radius_min = expected_radius_px * 0.65 if expected_radius_px else 20.0
    radius_max = expected_radius_px * 1.35 if expected_radius_px else max(image.shape) * 0.35

    for _ in range(max_trials):
        sample_idx = rng.choice(points.shape[0], size=3, replace=False)
        model = circle_from_three_points(points[sample_idx])
        if model is None:
            continue
        cx, cy, radius = model
        if not (radius_min <= radius <= radius_max):
            continue
        if cx < -radius or cx > image.shape[1] + radius or cy < -radius or cy > image.shape[0] + radius:
            continue
        residuals = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)
        inliers = residuals < tolerance_px
        if best_inliers is None or int(np.sum(inliers)) > int(np.sum(best_inliers)):
            best_inliers = inliers
            best_model = model

    if best_inliers is None or int(np.sum(best_inliers)) < min_edge_points:
        return LimbFit(ok=False, total_edge_count=int(points.shape[0]), reason="Could not fit a stable lunar limb.")

    cx, cy, radius = refine_circle(points[best_inliers])
    residuals = np.abs(np.hypot(points[best_inliers, 0] - cx, points[best_inliers, 1] - cy) - radius)
    return LimbFit(
        ok=True,
        center_x_px=cx,
        center_y_px=cy,
        radius_px=radius,
        inlier_count=int(np.sum(best_inliers)),
        total_edge_count=int(points.shape[0]),
        residual_px=float(np.median(residuals)),
    )


def image_offset_to_pointing_delta(
    fit: LimbFit,
    image_shape: Tuple[int, int],
    deg_per_pixel: float,
    current_altitude_deg: float,
    x_sign: float,
    y_sign: float,
) -> Tuple[float, float]:
    image_center_x = (image_shape[1] - 1) / 2.0
    image_center_y = (image_shape[0] - 1) / 2.0
    dx_px = fit.center_x_px - image_center_x
    dy_px = fit.center_y_px - image_center_y
    cos_alt = max(math.cos(math.radians(current_altitude_deg)), 0.2)
    delta_pan = x_sign * dx_px * deg_per_pixel / cos_alt
    delta_tilt = y_sign * dy_px * deg_per_pixel
    return delta_pan, delta_tilt


def move_moog(moog, target_pan_deg: float, target_tilt_deg: float):
    pan, tilt = clamp_pointing(target_pan_deg, target_tilt_deg)
    mf.mv_to_coord(moog, int(round(pan * 10.0)), int(round(tilt * 10.0)))
    return pan, tilt


def get_actual_moog_position(moog) -> Tuple[float, float]:
    status = mf.get_status_jog(moog)
    if status is None:
        return np.nan, np.nan
    return float(status.pan_coord), float(status.tilt_coord)


def write_observation_attrs(attrs, prefix: str, obs: MoonObservation):
    attrs[f"{prefix} Moon Azimuth [deg]"] = float(obs.azimuth_deg)
    attrs[f"{prefix} Moon Altitude [deg]"] = float(obs.altitude_deg)
    attrs[f"{prefix} Moon Distance [km]"] = float(obs.distance_km)
    attrs[f"{prefix} Moon Angular Diameter [deg]"] = float(obs.angular_diameter_deg)
    attrs[f"{prefix} Ephemeris Backend"] = obs.backend


def parse_args():
    parser = argparse.ArgumentParser(description="Track and center the Moon with ULTRASIP Moog hardware.")
    parser.add_argument("--latitude", type=float, default=32.23134)
    parser.add_argument("--longitude", type=float, default=-110.94712)
    parser.add_argument("--elevation-m", type=float, default=0.0)
    parser.add_argument("--location", default="MeinelRoof")
    parser.add_argument("--outpath", default="C:/Users/deleo/Documents/Data")
    parser.add_argument("--moog-port", default="COM7")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--pan-offset", type=float, default=-9.0)
    parser.add_argument("--tilt-offset", type=float, default=0.0)
    parser.add_argument("--exposure-us", type=float, default=1000.0)
    parser.add_argument("--track-seconds", type=float, default=900.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--center-iterations", type=int, default=4)
    parser.add_argument("--center-tolerance-px", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=0.4)
    parser.add_argument("--deg-per-pixel", type=float, default=UV_DEG_PER_PIXEL)
    parser.add_argument("--image-x-to-pan-sign", type=float, default=1.0)
    parser.add_argument("--image-y-to-tilt-sign", type=float, default=1.0)
    parser.add_argument("--skyfield-data-dir", default="skyfield-data")
    parser.add_argument("--ephemeris", default="de421.bsp")
    parser.add_argument("--dry-run", action="store_true", help="Compute Moon positions and create an H5 log without hardware.")
    return parser.parse_args()


def main():
    args = parse_args()
    ephemeris = MoonEphemeris(args.skyfield_data_dir, args.ephemeris)
    start_dt = datetime.now().astimezone()
    date_dir = start_dt.strftime("%Y_%m_%d")
    datapath = os.path.join(args.outpath, date_dir)
    os.makedirs(datapath, exist_ok=True)
    filename = f"{args.location}_MoonTrack_{start_dt.strftime('%Y%m%d_%H_%M_%S')}.h5"
    filepath = os.path.join(datapath, filename)

    moog = None
    uvcam = None
    vmb_context = None

    try:
        if not args.dry_run:
            if serial is None:
                raise ImportError("pyserial is required for hardware tracking.")
            if uv is None:
                raise ImportError("VmbPy/uv_cam_functions is required for camera-based Moon tracking.")
            moog = serial.Serial()
            moog.baudrate = args.baudrate
            moog.port = args.moog_port
            moog.open()
            mf.init_autobaud(moog)
            mf.get_status_jog(moog)

            cam_id = uv.parse_args()
            vmb_context = uv.VmbSystem.get_instance()
            vmb_context.__enter__()
            uvcam = uv.get_camera(cam_id)
            uvcam.__enter__()
            uv.setup_camera(uvcam, args.exposure_us)

        with h5py.File(filepath, "w") as h5:
            meta = h5.create_group("Measurement_Metadata")
            meta.attrs["Mode"] = "Dry Run" if args.dry_run else "Hardware"
            meta.attrs["Created Local Time"] = start_dt.isoformat(timespec="seconds")
            meta.attrs["Latitude"] = args.latitude
            meta.attrs["Longitude"] = args.longitude
            meta.attrs["Elevation [m]"] = args.elevation_m
            meta.attrs["Location"] = args.location
            meta.attrs["Pan_Offset"] = args.pan_offset
            meta.attrs["Tilt_Offset"] = args.tilt_offset
            meta.attrs["UV Pixel Scale [deg/pixel]"] = args.deg_per_pixel
            meta.attrs["Center Method"] = "RANSAC lunar limb circle fit; no brightness centroid"

            current_obs = ephemeris.observe(datetime.now().astimezone(), args.latitude, args.longitude, args.elevation_m)
            target_pan = current_obs.azimuth_deg - args.pan_offset
            target_tilt = current_obs.altitude_deg - args.tilt_offset

            if not args.dry_run:
                move_moog(moog, target_pan, target_tilt)
                time.sleep(args.settle_seconds)

                for iteration in range(args.center_iterations):
                    frame = uvcam.get_frame()
                    image = frame_to_image(frame)
                    current_obs = ephemeris.observe(datetime.now().astimezone(), args.latitude, args.longitude, args.elevation_m)
                    expected_radius_px = (current_obs.angular_diameter_deg / 2.0) / args.deg_per_pixel
                    fit = fit_lunar_limb(image, expected_radius_px=expected_radius_px)
                    center_group = h5.create_group(f"Centering_{iteration}")
                    center_group.create_dataset("UV Image", data=image, compression="gzip")
                    write_observation_attrs(center_group.attrs, "Observation", current_obs)
                    center_group.attrs["Fit OK"] = fit.ok
                    center_group.attrs["Fit Reason"] = fit.reason
                    center_group.attrs["Moon Center X [px]"] = fit.center_x_px
                    center_group.attrs["Moon Center Y [px]"] = fit.center_y_px
                    center_group.attrs["Moon Radius [px]"] = fit.radius_px
                    center_group.attrs["Fit Inlier Count"] = fit.inlier_count
                    center_group.attrs["Fit Total Edge Count"] = fit.total_edge_count
                    center_group.attrs["Fit Median Residual [px]"] = fit.residual_px
                    if not fit.ok:
                        break

                    image_center_x = (image.shape[1] - 1) / 2.0
                    image_center_y = (image.shape[0] - 1) / 2.0
                    residual_px = math.hypot(fit.center_x_px - image_center_x, fit.center_y_px - image_center_y)
                    if residual_px <= args.center_tolerance_px:
                        break

                    delta_pan, delta_tilt = image_offset_to_pointing_delta(
                        fit,
                        image.shape,
                        args.deg_per_pixel,
                        current_obs.altitude_deg,
                        args.image_x_to_pan_sign,
                        args.image_y_to_tilt_sign,
                    )
                    target_pan += delta_pan
                    target_tilt += delta_tilt
                    target_pan, target_tilt = move_moog(moog, target_pan, target_tilt)
                    center_group.attrs["Applied Delta Pan [deg]"] = delta_pan
                    center_group.attrs["Applied Delta Tilt [deg]"] = delta_tilt
                    center_group.attrs["Target Moog Pan [deg]"] = target_pan
                    center_group.attrs["Target Moog Tilt [deg]"] = target_tilt
                    time.sleep(args.settle_seconds)

            start = time.time()
            sample_idx = 0
            while time.time() - start <= args.track_seconds:
                dt = datetime.now().astimezone()
                obs = ephemeris.observe(dt, args.latitude, args.longitude, args.elevation_m)
                target_pan = obs.azimuth_deg - args.pan_offset
                target_tilt = obs.altitude_deg - args.tilt_offset

                if args.dry_run:
                    actual_pan = np.nan
                    actual_tilt = np.nan
                    commanded_pan, commanded_tilt = clamp_pointing(target_pan, target_tilt)
                    image = None
                    fit = LimbFit(ok=False, reason="dry run")
                else:
                    commanded_pan, commanded_tilt = move_moog(moog, target_pan, target_tilt)
                    time.sleep(args.settle_seconds)
                    actual_pan, actual_tilt = get_actual_moog_position(moog)
                    frame = uvcam.get_frame()
                    image = frame_to_image(frame)
                    expected_radius_px = (obs.angular_diameter_deg / 2.0) / args.deg_per_pixel
                    fit = fit_lunar_limb(image, expected_radius_px=expected_radius_px)

                aq = h5.create_group(f"Tracking_{sample_idx}")
                aq.attrs["Timestamp Local"] = dt.isoformat(timespec="milliseconds")
                aq.attrs["Timestamp UTC"] = dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")
                write_observation_attrs(aq.attrs, "Observation", obs)
                aq.attrs["Target Moog Pan [deg]"] = float(commanded_pan)
                aq.attrs["Target Moog Tilt [deg]"] = float(commanded_tilt)
                aq.attrs["Actual Moog Pan [deg]"] = float(actual_pan)
                aq.attrs["Actual Moog Tilt [deg]"] = float(actual_tilt)
                aq.attrs["Moog Pan Error [deg]"] = wrap_to_180(actual_pan - commanded_pan) if np.isfinite(actual_pan) else np.nan
                aq.attrs["Moog Tilt Error [deg]"] = actual_tilt - commanded_tilt if np.isfinite(actual_tilt) else np.nan
                aq.attrs["Fit OK"] = fit.ok
                aq.attrs["Fit Reason"] = fit.reason
                aq.attrs["Moon Center X [px]"] = fit.center_x_px
                aq.attrs["Moon Center Y [px]"] = fit.center_y_px
                aq.attrs["Moon Radius [px]"] = fit.radius_px
                aq.attrs["Fit Inlier Count"] = fit.inlier_count
                aq.attrs["Fit Total Edge Count"] = fit.total_edge_count
                aq.attrs["Fit Median Residual [px]"] = fit.residual_px
                if image is not None:
                    aq.create_dataset("UV Image", data=image, compression="gzip")

                print(
                    f"{dt.strftime('%H:%M:%S')} Moon az={obs.azimuth_deg:.3f}, alt={obs.altitude_deg:.3f}, "
                    f"target Moog=({commanded_pan:.3f}, {commanded_tilt:.3f}), "
                    f"actual=({actual_pan:.3f}, {actual_tilt:.3f})"
                )
                sample_idx += 1
                h5.flush()
                time.sleep(args.interval_seconds)

            meta.attrs["Samples Written"] = sample_idx
            meta.attrs["Completed Local Time"] = datetime.now().astimezone().isoformat(timespec="seconds")

    finally:
        if uvcam is not None:
            uvcam.__exit__(None, None, None)
        if vmb_context is not None:
            vmb_context.__exit__(None, None, None)
        if moog is not None and moog.is_open:
            moog.close()

    print(f"Moon tracking log written to {filepath}")


if __name__ == "__main__":
    main()

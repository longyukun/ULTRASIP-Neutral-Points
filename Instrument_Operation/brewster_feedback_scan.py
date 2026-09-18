#!/usr/bin/env python3
"""Fast, no-NUC Brewster search. Hardware is opened only by main()."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ANGLES = (0, 45, 90, 135)
SETTINGS = Path.home() / '.ultrasip_auto_scan_settings.json'


def attitude(x, y, args):
    """Provisional installation model requested by operator (not Euler conversion)."""
    return -y + args.tilt_offset, args.roll_sign * x + args.roll_offset


def rotate_stokes(q, u, roll_deg):
    c, s = np.cos(np.deg2rad(2 * roll_deg)), np.sin(np.deg2rad(2 * roll_deg))
    return c * q + s * u, -s * q + c * u


def analyze(images, roll, args):
    h, w = images[0].shape
    cx = (w - 1) / 2 if args.center_x is None else args.center_x
    cy = (h - 1) / 2 if args.center_y is None else args.center_y
    xlo, xhi = max(0, int(cx - args.roi_width / 2)), min(w, int(cx + args.roi_width / 2) + 1)
    ylo, yhi = max(0, int(cy - args.band_height / 2)), min(h, int(cy + args.band_height / 2) + 1)
    if xhi - xlo < 16 or yhi <= ylo or not (xlo <= cx < xhi and ylo <= cy < yhi):
        raise ValueError('Optical center / ROI is outside image')
    raw = np.stack([im[ylo:yhi, xlo:xhi] for im in images]).astype(float)
    intensity = raw.sum(axis=0) / 2
    valid = np.all(np.isfinite(raw) & (raw < args.saturation), axis=0) & (intensity > args.min_intensity)
    q, u = rotate_stokes(raw[0] - raw[2], raw[1] - raw[3], roll)
    q = np.divide(q, intensity, out=np.full_like(q, np.nan), where=valid)
    u = np.divide(u, intensity, out=np.full_like(u, np.nan), where=valid)
    counts = valid.sum(axis=0)
    good = counts >= max(1, math.ceil((yhi - ylo) * .8))
    x = (np.arange(xlo, xhi) - cx) * args.deg_per_pixel
    profile = np.nansum(u, axis=0) / np.maximum(counts, 1)
    base = dict(valid=False, reason='too few valid columns', root_deg=None, root_se_deg=None,
                slope=None, q_center=None, u_center=None)
    if good.sum() < 16:
        return base, x, np.where(good, profile, np.nan)
    a, b = np.polyfit(x[good], profile[good], 1)
    residual = profile[good] - (a * x[good] + b)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    center = (np.abs(x) <= args.center_width * args.deg_per_pixel / 2)[None, :] & valid
    if not center.any():
        return {**base, 'reason': 'no valid center pixels'}, x, profile
    base.update(slope=float(a), intercept=float(b), fit_rms=rms,
                q_center=float(np.mean(q[center])), u_center=float(np.mean(u[center])))
    if abs(a) < args.min_slope:
        return {**base, 'reason': 'U slope too small'}, x, profile
    root = float(-b / a)
    design = np.column_stack([x[good], np.ones(good.sum())])
    cov = (residual @ residual / (good.sum() - 2)) * np.linalg.inv(design.T @ design)
    jac = np.array([b / a ** 2, -1 / a])
    se = float(np.sqrt(max(0, jac @ cov @ jac)))
    base.update(root_deg=root, root_se_deg=se)
    reason = ('zero outside ROI' if not x[good].min() <= root <= x[good].max() else
              'nonlinear/noisy U fit' if rms > args.max_fit_rms else
              'uncertain zero' if se > args.max_root_se else '')
    return {**base, 'valid': not reason, 'reason': reason}, x, profile


class Hardware:
    def __init__(self, args):
        from Measurement_QT_GUI import RealMoogController, RealPolarizerController, RealVmbCameraController
        from inc1000 import INC1000
        self.args = args
        self.moog = RealMoogController()
        self.pol = RealPolarizerController()
        self.cam = RealVmbCameraController(args.exposure_us)
        self.inc = INC1000(args.inc_port, baudrate=args.inc_baud)

    def open(self):
        self.inc.open()
        self.moog.open(self.args.moog_port)
        self.pol.open(self.args.zaber_port)
        self.cam.open()

    def position(self):
        s = self.moog.get_status()
        return s.pan_deg, s.tilt_deg

    def read(self):
        r = self.inc.read_measurement()
        if not all(math.isfinite(v) for v in (r.x_deg, r.y_deg, r.temperature_c)):
            raise RuntimeError('INC1000 returned non-finite data')
        return r.x_deg, r.y_deg, r.temperature_c

    def move(self, pan, tilt):
        if not (-217.5 <= pan <= 217.5 and -90 <= tilt <= 90):
            raise RuntimeError('Requested motion exceeds Moog limits')
        self.moog.move_absolute(pan, tilt)
        time.sleep(self.args.settle)

    def frame(self, angle):
        actual = self.pol.move_absolute(angle)
        if abs(actual - angle) > .05:
            raise RuntimeError('Polarizer did not reach requested angle')
        time.sleep(self.args.polarizer_settle)
        return self.cam.get_frame().copy()

    def close(self):
        # Existing Moog.close() homes the mount; do not invoke it on abort.
        for device in (self.cam, self.pol, self.inc):
            try:
                device.close()
            except Exception as exc:
                print(f'Close warning: {exc}', file=sys.stderr)
        port = self.moog.serial_port
        if port and port.is_open:
            try:
                self.moog.mf.get_status_jog(port, stop=1, verbose=False)
            finally:
                port.close()


class Simulation:
    """Synthetic tilted mount, displaced U line and a Q zero at zenith 55 deg."""
    def __init__(self, args):
        self.args, self.pan, self.tilt = args, 0., 20.
        self.rng = np.random.default_rng(7)
    def open(self): pass
    def close(self): pass
    def position(self): return self.pan, self.tilt
    def move(self, pan, tilt): self.pan, self.tilt = pan, tilt
    def read(self):
        return 2., -(self.tilt + .3 * math.sin(math.radians(self.pan))), 25.
    def frame(self, angle):
        altitude, roll = attitude(*self.read()[:2], self.args)
        x = (np.arange(512) - 255.5) * self.args.deg_per_pixel
        root = .12 + .01 * (altitude - 20) - self.pan * math.cos(math.radians(altitude))
        u = np.broadcast_to(.04 * (x - root), (128, 512))
        q = np.full_like(u, .004 * (altitude - 35))
        qcam, ucam = rotate_stokes(q, u, -roll)
        phi = math.radians(2 * angle)
        im = 1200 * (1 + qcam * math.cos(phi) + ucam * math.sin(phi))
        return np.round(im + self.rng.normal(0, .3, im.shape)).astype(np.uint16)


class Scanner:
    def __init__(self, args, device, output):
        self.a, self.dev, self.output = args, device, output
        self.index = 0
        self.previous = None
        self.brackets = []
        self.plot = None
        if args.plot:
            import matplotlib.pyplot as plt
            plt.ion()
            self.plot = plt
            self.fig, self.ax = plt.subplots()

    def sun(self):
        if self.a.simulate:
            return 180., 45.
        from Measurement_QT_GUI import solar_position_deg
        return solar_position_deg(datetime.now(timezone.utc), self.a.latitude, self.a.longitude)

    def set_altitude(self, target):
        for _ in range(self.a.max_tilt_iterations):
            x, y, _ = self.dev.read()
            altitude, _ = attitude(x, y, self.a)
            error = target - altitude
            if abs(error) <= self.a.tilt_tolerance:
                return
            pan, tilt = self.dev.position()
            self.dev.move(pan, tilt + float(np.clip(error, -self.a.max_tilt_step, self.a.max_tilt_step)))
        raise RuntimeError('INC1000 altitude feedback did not converge')

    def capture(self, target, phase):
        folder = self.output / f'acquisition_{self.index:04d}'
        folder.mkdir()
        self.index += 1
        readings, images, timestamps = [], [], []
        start_position = self.dev.position()
        # Save each frame immediately; interrupted quartets remain recoverable.
        for angle in ANGLES:
            before = self.dev.read()
            frame = self.dev.frame(angle)
            after = self.dev.read()
            np.save(folder / f'raw_{angle:03d}.npy', frame)
            images.append(frame)
            readings.extend((before, after))
            timestamps.append(datetime.now(timezone.utc).isoformat())
            (folder / 'partial.json').write_text(json.dumps(dict(angles=list(ANGLES[:len(images)]),
                readings=readings, timestamps=timestamps)), encoding='utf-8')
        readings = np.asarray(readings)
        x, y, temp = readings.mean(axis=0)
        altitude, roll = attitude(x, y, self.a)
        result, degrees, profile = analyze(images, roll, self.a)
        if np.ptp(readings[:, :2], axis=0).max() > self.a.max_pose_drift:
            result.update(valid=False, reason='pose changed during quartet')
        pan, tilt = self.dev.position()
        if max(abs(pan - start_position[0]), abs(tilt - start_position[1])) > self.a.max_pose_drift:
            result.update(valid=False, reason='Moog moved during quartet')
        record = dict(index=self.index - 1, phase=phase, timestamp=timestamps[-1],
                      target_zenith=90 - target, zenith=90 - altitude, roll=roll,
                      inc_x=float(x), inc_y=float(y), temperature=float(temp),
                      pan=pan, moog_tilt=tilt, exposure_us=self.a.exposure_us,
                      readings=readings.tolist(), frame_timestamps=timestamps, **result)
        (folder / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        np.savez(folder / 'u_fit.npz', degree=degrees, u=profile)
        with (self.output / 'measurements.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        root_text = f"{result['root_deg']:+.4f}" if result['root_deg'] is not None else 'n/a'
        print(f"{phase}: zenith={90-altitude:.3f}, roll={roll:+.3f}, U zero={root_text} deg, "
              f"Q/I={result['q_center']}, {result['reason'] or 'fit OK'}", flush=True)
        if self.plot:
            self.ax.clear()
            self.ax.plot(degrees, profile, '.', markersize=2)
            if result['slope'] is not None:
                self.ax.plot(degrees, result['slope'] * degrees + result['intercept'])
            self.ax.axhline(0, color='gray'); self.ax.axvline(0, color='gray')
            self.ax.set(xlabel='Image horizontal angle relative to optical axis [deg]',
                        ylabel="U'/I (no NUC)", title=f'Zenith {90-altitude:.2f} | {phase}')
            self.plot.pause(.01)
        return record

    def require_fit(self, record):
        if not record['valid']:
            raise RuntimeError('Feedback stopped: ' + record['reason'])

    def track(self, target):
        self.set_altitude(target)
        record = self.capture(target, 'track')
        self.require_fit(record)
        if abs(record['root_deg']) <= self.a.pan_tolerance:
            return record
        # Measure signed image-zero displacement per actual motor degree.
        start_pan, tilt = self.dev.position()
        self.dev.move(start_pan + self.a.pan_probe, tilt)
        self.set_altitude(target)
        probe = self.capture(target, 'pan_probe')
        self.require_fit(probe)
        dp = probe['pan'] - start_pan
        if abs(dp) < .05:
            raise RuntimeError('Pan probe did not move sufficiently')
        response = (probe['root_deg'] - record['root_deg']) / dp
        uncertainty = math.hypot(probe['root_se_deg'], record['root_se_deg'])
        if abs(response) < .05 or abs(probe['root_deg'] - record['root_deg']) < 3 * uncertainty:
            raise RuntimeError('Pan probe response too small/noisy')
        record = probe
        for _ in range(self.a.max_pan_iterations):
            if abs(record['root_deg']) <= self.a.pan_tolerance:
                return record
            delta = float(np.clip(-self.a.gain * record['root_deg'] / response,
                                  -self.a.max_pan_step, self.a.max_pan_step))
            # Controller commands resolve 0.1 deg, so do not request vanishing steps.
            if abs(delta) < .1:
                delta = math.copysign(.1, delta)
            pan, tilt = self.dev.position()
            if abs(pan + delta - start_pan) > self.a.max_pan_excursion:
                raise RuntimeError('Pan tracking excursion exceeded')
            self.dev.move(pan + delta, tilt)
            self.set_altitude(target)
            new = self.capture(target, 'track')
            self.require_fit(new)
            if abs(new['root_deg']) > abs(record['root_deg']) + self.a.pan_tolerance:
                raise RuntimeError('U feedback diverged; check roll sign / selected U branch')
            record = new
        raise RuntimeError('Pan tracking did not converge')

    def run(self):
        azimuth, sun_alt = self.sun()
        target = 90 - self.a.start_zenith
        if target >= sun_alt - self.a.sun_margin:
            raise ValueError('Sun must be above the starting altitude plus sun margin')
        pan = ((azimuth - 180 + 180) % 360 - 180) - self.a.pan_offset
        _, tilt = self.dev.position()
        self.dev.move(pan, tilt)
        while True:
            _, sun_alt = self.sun()
            limit = sun_alt - self.a.sun_margin
            if target > limit + 1e-6:
                break
            record = self.track(target)
            record['aligned'] = True
            with (self.output / 'aligned.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            if self.previous is not None and self.previous['q_center'] * record['q_center'] <= 0:
                bracket = dict(zenith_a=self.previous['zenith'], zenith_b=record['zenith'],
                               acquisition_a=self.previous['index'], acquisition_b=record['index'],
                               label='Candidate only: no NUC, optical offsets / roll sign provisional')
                self.brackets.append(bracket)
                (self.output / 'candidates.json').write_text(json.dumps(self.brackets, indent=2), encoding='utf-8')
                print(f"Q sign-change candidate: zenith {bracket['zenith_a']:.3f} to {bracket['zenith_b']:.3f}", flush=True)
            self.previous = record
            if abs(target - limit) < 1e-6:
                break
            target = min(target + self.a.zenith_step, limit)
        return self.brackets


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulate', action='store_true', help='No hardware; synthetic full scan')
    p.add_argument('--calibrate-sun', action='store_true', help='Open original calibration GUI first; finish calibration and close GUI to continue')
    p.add_argument('--settings', type=Path, default=SETTINGS)
    p.add_argument('--output', type=Path, default=Path('brewster_scans'))
    p.add_argument('--plot', action='store_true')
    p.add_argument('--roll-sign', type=int, choices=(-1, 1), help='Required for hardware; test both signs if unknown')
    for name, default in [('tilt-offset', 0.), ('roll-offset', 0.), ('start-zenith', 70.),
                          ('zenith-step', 2.), ('sun-margin', 0.), ('exposure-us', 1000.),
                          ('deg-per-pixel', .002), ('saturation', 4090.), ('min-intensity', 100.),
                          ('min-slope', .001), ('max-fit-rms', .01), ('max-root-se', .05),
                          ('pan-tolerance', .08), ('tilt-tolerance', .15), ('pan-probe', .2),
                          ('gain', .7), ('max-pan-step', 1.), ('max-pan-excursion', 3.),
                          ('max-tilt-step', 5.), ('max-pose-drift', .15), ('settle', .5),
                          ('polarizer-settle', .5)]:
        p.add_argument('--' + name, type=float, default=default)
    for name in ('center-x', 'center-y', 'pan-offset', 'latitude', 'longitude'):
        p.add_argument('--' + name, type=float)
    for name, default in [('roi-width', 800), ('band-height', 40), ('center-width', 20),
                          ('max-pan-iterations', 8), ('max-tilt-iterations', 30), ('inc-baud', 115200)]:
        p.add_argument('--' + name, type=int, default=default)
    for name, default in [('inc-port', 'COM1'), ('moog-port', 'COM7'), ('zaber-port', 'COM6')]:
        p.add_argument('--' + name, default=default)
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.calibrate_sun:
        if a.simulate:
            p.error('--calibrate-sun cannot be used with --simulate')
        subprocess.run([sys.executable, str(Path(__file__).with_name('Measurement_QT_GUI.py'))], check=True)
    settings = json.loads(a.settings.read_text(encoding='utf-8')) if a.settings.exists() else {}
    for key, default in [('pan_offset', None), ('latitude', 32.23134), ('longitude', -110.94712)]:
        if getattr(a, key) is None:
            setattr(a, key, settings.get(key, default))
    if a.simulate:
        a.roll_sign = a.roll_sign or 1
        a.pan_offset = 0.
    if a.roll_sign is None:
        p.error('Choose --roll-sign 1 or --roll-sign -1; INC1000 X sign is not yet calibrated')
    if a.pan_offset is None:
        p.error('No solar azimuth calibration: use --calibrate-sun or --pan-offset')
    for key, value in vars(a).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            p.error(f'{key} must be finite')
    positive = ('zenith_step', 'deg_per_pixel', 'min_slope', 'max_fit_rms', 'max_root_se',
                'pan_tolerance', 'tilt_tolerance', 'max_pan_step', 'max_pan_excursion',
                'max_tilt_step', 'max_pose_drift', 'roi_width', 'band_height', 'center_width',
                'max_pan_iterations', 'max_tilt_iterations', 'min_intensity', 'saturation')
    if any(getattr(a, k) <= 0 for k in positive) or not 0 < a.gain <= 1 or abs(a.pan_probe) < .1:
        p.error('Feedback scales must be positive, gain in (0,1], and |pan-probe| >= 0.1')
    if not 0 < a.start_zenith < 90 or a.sun_margin < 0 or min(a.settle, a.polarizer_settle) < 0:
        p.error('Invalid scan range / settle time')
    if not 100 <= a.exposure_us <= 1e6:
        p.error('Exposure must be 100..1000000 us')
    output = a.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(vars(a), default=str, indent=2), encoding='utf-8')
    print(f'Output: {output}\nAltitude=-Y+{a.tilt_offset}; roll={a.roll_sign}*X+{a.roll_offset}; no NUC', flush=True)
    device = Simulation(a) if a.simulate else Hardware(a)
    state = {'status': 'failed'}
    try:
        device.open()
        scanner = Scanner(a, device, output)
        candidates = scanner.run()
        state = dict(status='complete', candidate_brackets=candidates)
        return 0
    except KeyboardInterrupt:
        state = dict(status='interrupted', reason='Operator interrupted')
        print('Stopped by operator; acquired files retained.', file=sys.stderr)
        return 130
    except Exception as exc:
        state = dict(status='failed', reason=str(exc))
        print(f'Stopped: {exc}', file=sys.stderr)
        return 1
    finally:
        try:
            device.close()
        finally:
            (output / 'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())

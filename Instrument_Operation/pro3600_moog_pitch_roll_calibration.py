#!/usr/bin/env python3
"""Calibrate Moog pitch/roll from a PRO3600 level while sweeping pan angle.

The scan path is 0 -> -180 -> 0 -> 180 -> 0 in 5 degree increments,
repeated three times by default.  At each pan angle the Moog is held still,
the PRO3600 angle is sampled, and the final data are fit to:

    level_angle = offset + pitch * cos(pan) + roll * sin(pan)

Sign convention depends on the physical mounting direction of the level.
Use the output CSV/plot to verify whether pitch or roll signs need to be
flipped for your instrument coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Missing dependency: pyserial", file=sys.stderr)
    print("Install it with: py -m pip install pyserial", file=sys.stderr)
    raise SystemExit(1)


DEFAULT_BAUD = 9600
DEFAULT_MOOG_PORT = "COM7"
DEFAULT_TRIGGER = "tx-break-pulse"
DEFAULT_REPEAT_TRIGGER = 1.0
DEFAULT_PULSE_WIDTH = 0.2

TRIGGER_CHOICES = (
    "rts",
    "rts-pulse",
    "dtr",
    "dtr-pulse",
    "tx-break",
    "tx-break-pulse",
    "none",
)

PARITY_CHOICES = {
    "none": serial.PARITY_NONE,
    "even": serial.PARITY_EVEN,
    "odd": serial.PARITY_ODD,
}

STOPBITS_CHOICES = {
    "1": serial.STOPBITS_ONE,
    "1.5": serial.STOPBITS_ONE_POINT_FIVE,
    "2": serial.STOPBITS_TWO,
}

NUMBER_RE = re.compile(r"[-+]?\s*\d+(?:\.\d+)?")


@dataclass
class Sample:
    cycle: int
    index: int
    pan_deg: float
    angle_deg: float
    angle_std_deg: float | None
    n_readings: int
    timestamp: str


def list_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}  {port.description or 'unknown device'}")


def apply_trigger(ser: serial.Serial, trigger: str, pulse_width: float) -> None:
    if trigger == "rts":
        ser.rts = True
    elif trigger == "rts-pulse":
        ser.rts = False
        time.sleep(pulse_width)
        ser.rts = True
    elif trigger == "dtr":
        ser.dtr = True
    elif trigger == "dtr-pulse":
        ser.dtr = False
        time.sleep(pulse_width)
        ser.dtr = True
    elif trigger == "tx-break":
        ser.break_condition = True
    elif trigger == "tx-break-pulse":
        ser.break_condition = True
        time.sleep(pulse_width)
        ser.break_condition = False


def release_trigger(ser: serial.Serial, trigger: str) -> None:
    if trigger in ("rts", "rts-pulse"):
        ser.rts = False
    elif trigger in ("dtr", "dtr-pulse"):
        ser.dtr = False
    elif trigger in ("tx-break", "tx-break-pulse"):
        ser.break_condition = False


def parse_angle(line: str) -> float | None:
    """Parse a PRO3600 ASCII angle line such as '+ 12.34' or '-  4.32'."""
    match = NUMBER_RE.search(line)
    if not match:
        return None
    return float(match.group(0).replace(" ", ""))


class PRO3600Reader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ser = serial.Serial(
            port=args.level_port,
            baudrate=args.baud,
            bytesize=args.bytesize,
            parity=PARITY_CHOICES[args.parity],
            stopbits=STOPBITS_CHOICES[args.stopbits],
            timeout=args.timeout,
        )
        self.buffer = bytearray()
        self.last_trigger = time.monotonic()
        apply_trigger(self.ser, args.trigger, args.pulse_width)
        if args.trigger_delay > 0:
            time.sleep(args.trigger_delay)

    def close(self) -> None:
        release_trigger(self.ser, self.args.trigger)
        self.ser.close()

    def _maybe_trigger(self) -> None:
        if self.args.repeat_trigger <= 0:
            return
        now = time.monotonic()
        if now - self.last_trigger >= self.args.repeat_trigger:
            apply_trigger(self.ser, self.args.trigger, self.args.pulse_width)
            self.last_trigger = now

    def read_available_angles(self) -> list[float]:
        self._maybe_trigger()
        raw = self.ser.read(self.ser.in_waiting or 1)
        values: list[float] = []
        if not raw:
            return values

        self.buffer.extend(raw)
        while b"\n" in self.buffer or b"\r" in self.buffer:
            positions = [pos for pos in (self.buffer.find(b"\n"), self.buffer.find(b"\r")) if pos >= 0]
            split_at = min(positions)
            line_bytes = bytes(self.buffer[:split_at])
            del self.buffer[: split_at + 1]
            line = line_bytes.decode("ascii", errors="ignore").strip()
            value = parse_angle(line)
            if value is not None:
                values.append(value)

        if len(self.buffer) > 80:
            line = bytes(self.buffer).decode("ascii", errors="ignore").strip()
            self.buffer.clear()
            value = parse_angle(line)
            if value is not None:
                values.append(value)
        return values

    def sample(self, duration: float, min_readings: int, max_wait: float) -> tuple[float, float | None, int]:
        readings: list[float] = []
        start = time.monotonic()
        deadline = start + duration
        hard_deadline = start + max(duration, max_wait)
        while time.monotonic() < hard_deadline:
            readings.extend(self.read_available_angles())
            if len(readings) >= min_readings and time.monotonic() >= deadline:
                break
            time.sleep(self.args.interval)

        if not readings:
            raise RuntimeError("No PRO3600 angle readings received during sample window.")

        angle = statistics.median(readings)
        stdev = statistics.stdev(readings) if len(readings) > 1 else None
        return angle, stdev, len(readings)


class MoogBackend:
    def move_pan(self, pan_deg: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class DryRunMoog(MoogBackend):
    def move_pan(self, pan_deg: float) -> None:
        print(f"[dry-run] move Moog pan to {pan_deg:.1f} deg")


class CommandMoog(MoogBackend):
    def __init__(self, command_template: str) -> None:
        self.command_template = command_template

    def move_pan(self, pan_deg: float) -> None:
        command = self.command_template.format(pan=pan_deg)
        subprocess.run(command, shell=True, check=True)


class MoogFunctionsMoog(MoogBackend):
    """Moog backend using Instrument_Operation/moog_functions.py directly."""

    def __init__(self, port: str, tilt_deg: float, move_timeout: float) -> None:
        import moog_functions as mf

        self.mf = mf
        self.port = port
        self.tilt_deg = tilt_deg
        self.move_timeout = move_timeout
        self.serial_port = serial.Serial()
        self.serial_port.baudrate = 9600
        self.serial_port.port = port
        self.serial_port.timeout = 1.0
        self.serial_port.write_timeout = 1.0
        self.serial_port.open()
        self.mf.init_autobaud(self.serial_port)
        status = self.mf.get_status_jog(self.serial_port, verbose=False)
        print(f"Moog opened on {port}: pan={status.pan_coord:.2f}, tilt={status.tilt_coord:.2f}")

    def move_pan(self, pan_deg: float) -> None:
        target_pan = int(round(pan_deg * 10.0))
        target_tilt = int(round(self.tilt_deg * 10.0))
        status = self.mf.move_to_coord_and_wait(
            self.serial_port,
            target_pan,
            target_tilt,
            timeout=self.move_timeout,
            verbose=False,
        )
        print(f"    Moog actual pan={status.pan_coord:.2f}, tilt={status.tilt_coord:.2f}")

    def close(self) -> None:
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


def build_moog_backend(args: argparse.Namespace) -> MoogBackend:
    if args.moog_mode == "dry-run":
        return DryRunMoog()
    if args.moog_mode == "command":
        if not args.moog_command:
            raise ValueError("--moog-command is required when --moog-mode command")
        return CommandMoog(args.moog_command)
    return MoogFunctionsMoog(
        port=args.moog_port,
        tilt_deg=args.tilt,
        move_timeout=args.moog_move_timeout,
    )


def path_segment(start: int, stop: int, step_abs: int) -> list[int]:
    if start == stop:
        return [start]
    step = step_abs if stop > start else -step_abs
    return list(range(start, stop + step, step))


def scan_sequence(step_deg: int, repeats: int) -> list[tuple[int, float]]:
    waypoints = [0, -180, 0, 180, 0]
    one_cycle: list[int] = []
    for start, stop in zip(waypoints, waypoints[1:]):
        segment = path_segment(start, stop, step_deg)
        if one_cycle:
            segment = segment[1:]
        one_cycle.extend(segment)

    sequence: list[tuple[int, float]] = []
    for cycle in range(1, repeats + 1):
        sequence.extend((cycle, float(pan)) for pan in one_cycle)
    return sequence


def fit_pitch_roll(samples: Iterable[Sample]) -> dict[str, float]:
    rows = []
    y = []
    for sample in samples:
        theta = math.radians(sample.pan_deg)
        rows.append([1.0, math.cos(theta), math.sin(theta)])
        y.append(sample.angle_deg)

    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for row, value in zip(rows, y):
        for i in range(3):
            xty[i] += row[i] * value
            for j in range(3):
                xtx[i][j] += row[i] * row[j]

    offset, pitch, roll = solve_3x3(xtx, xty)
    residuals = []
    for row, value in zip(rows, y):
        fitted = offset + pitch * row[1] + roll * row[2]
        residuals.append(value - fitted)
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return {
        "offset_deg": offset,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "amplitude_deg": math.hypot(pitch, roll),
        "phase_deg": math.degrees(math.atan2(roll, pitch)),
        "rms_residual_deg": rms,
        "n_samples": float(len(y)),
    }


def solve_3x3(a: list[list[float]], b: list[float]) -> tuple[float, float, float]:
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular fit matrix.")
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        for j in range(col, 4):
            m[col][j] /= scale
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]
    return m[0][3], m[1][3], m[2][3]


def write_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cycle", "index", "pan_deg", "angle_deg", "angle_std_deg", "n_readings", "timestamp"]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.cycle,
                    sample.index,
                    f"{sample.pan_deg:.6f}",
                    f"{sample.angle_deg:.6f}",
                    "" if sample.angle_std_deg is None else f"{sample.angle_std_deg:.6f}",
                    sample.n_readings,
                    sample.timestamp,
                ]
            )


def write_plot(path: Path, samples: list[Sample], fit: dict[str, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot output.")
        return

    pans = [s.pan_deg for s in samples]
    angles = [s.angle_deg for s in samples]
    grid = [x for x in range(-180, 181)]
    fitted = [
        fit["offset_deg"]
        + fit["pitch_deg"] * math.cos(math.radians(x))
        + fit["roll_deg"] * math.sin(math.radians(x))
        for x in grid
    ]

    plt.figure(figsize=(9, 5))
    plt.scatter(pans, angles, s=12, alpha=0.65, label="PRO3600 readings")
    plt.plot(grid, fitted, color="black", linewidth=2, label="sinusoidal fit")
    plt.xlabel("Moog pan angle (deg)")
    plt.ylabel("PRO3600 angle (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def timestamp_for_file() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Moog pan and fit pitch/roll from PRO3600 level readings."
    )
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit.")
    parser.add_argument(
        "-p",
        "--port",
        "--level-port",
        dest="level_port",
        default="COM3",
        help="PRO3600 Windows COM port. Default: COM3.",
    )
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD, help="Baud rate. Default: 9600.")
    parser.add_argument("--bytesize", type=int, choices=(5, 6, 7, 8), default=8)
    parser.add_argument("--parity", choices=tuple(PARITY_CHOICES), default="none")
    parser.add_argument("--stopbits", choices=tuple(STOPBITS_CHOICES), default="1")
    parser.add_argument("--trigger", choices=TRIGGER_CHOICES, default=DEFAULT_TRIGGER)
    parser.add_argument("--trigger-delay", type=float, default=0.2)
    parser.add_argument("--pulse-width", type=float, default=DEFAULT_PULSE_WIDTH)
    parser.add_argument("--repeat-trigger", type=float, default=DEFAULT_REPEAT_TRIGGER)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=5, help="Pan step in degrees. Default: 5.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of scan repeats. Default: 3.")
    parser.add_argument("--settle", type=float, default=2.0, help="Seconds to wait after each move. Default: 2.")
    parser.add_argument("--sample-seconds", type=float, default=0.4, help="Reading window after settling. Default: 0.4.")
    parser.add_argument("--sample-timeout", type=float, default=5.0, help="Max seconds to wait for readings. Default: 5.")
    parser.add_argument("--min-readings", type=int, default=1, help="Minimum PRO3600 lines per pan point.")
    parser.add_argument("--tilt", type=float, default=0.0, help="Moog tilt/elevation held during pan sweep.")
    parser.add_argument(
        "--moog-mode",
        choices=("moog-functions", "command", "dry-run"),
        default="moog-functions",
        help="Moog control backend. Default: moog-functions.",
    )
    parser.add_argument("--moog-port", default=DEFAULT_MOOG_PORT, help=f"Moog Windows COM port. Default: {DEFAULT_MOOG_PORT}.")
    parser.add_argument("--moog-move-timeout", type=float, default=30.0, help="Max seconds to wait for each Moog move. Default: 30.")
    parser.add_argument(
        "--moog-command",
        help="Command template for --moog-mode command, for example: py move_moog.py --pan {pan}",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for CSV/JSON/PNG outputs.")
    parser.add_argument("--no-plot", action="store_true", help="Skip diagnostic plot.")
    parser.add_argument("--plan-only", action="store_true", help="Print scan sequence and exit without hardware.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_ports:
        list_serial_ports()
        return

    if args.step <= 0 or 180 % args.step != 0:
        raise SystemExit("--step must be positive and divide 180 exactly.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_for_file()
    csv_path = output_dir / f"pro3600_moog_pitch_roll_{stamp}.csv"
    json_path = output_dir / f"pro3600_moog_pitch_roll_{stamp}.json"
    plot_path = output_dir / f"pro3600_moog_pitch_roll_{stamp}.png"

    sequence = scan_sequence(args.step, args.repeats)
    if args.plan_only:
        for index, (cycle, pan_deg) in enumerate(sequence, start=1):
            print(f"{index:03d}: cycle={cycle} pan={pan_deg:.1f}")
        print(f"Total points: {len(sequence)}")
        return

    print(f"Scan points: {len(sequence)}")
    print(f"PRO3600 port: {args.level_port}")
    print(f"Moog mode: {args.moog_mode}")
    if args.moog_mode == "moog-functions":
        print(f"Moog port: {args.moog_port}")
    print(f"Output CSV: {csv_path}")

    moog = build_moog_backend(args)
    reader: PRO3600Reader | None = None
    samples: list[Sample] = []
    try:
        reader = PRO3600Reader(args)
        for index, (cycle, pan_deg) in enumerate(sequence, start=1):
            print(f"[{index}/{len(sequence)}] cycle={cycle} pan={pan_deg:.1f}")
            moog.move_pan(pan_deg)
            time.sleep(args.settle)
            angle, stdev, n_readings = reader.sample(
                args.sample_seconds,
                args.min_readings,
                args.sample_timeout,
            )
            sample = Sample(
                cycle=cycle,
                index=index,
                pan_deg=pan_deg,
                angle_deg=angle,
                angle_std_deg=stdev,
                n_readings=n_readings,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            samples.append(sample)
            print(
                f"    level={angle:.4f} deg"
                + ("" if stdev is None else f" std={stdev:.4f}")
                + f" n={n_readings}"
            )
            write_csv(csv_path, samples)
    finally:
        if reader is not None:
            reader.close()
        moog.close()

    fit = fit_pitch_roll(samples)
    result = {
        "fit_model": "level_angle_deg = offset_deg + pitch_deg*cos(pan_deg) + roll_deg*sin(pan_deg)",
        "sign_note": "Pitch/roll signs depend on PRO3600 mounting direction.",
        "scan": {
            "step_deg": args.step,
            "repeats": args.repeats,
            "settle_seconds": args.settle,
            "sample_seconds": args.sample_seconds,
            "tilt_deg": args.tilt,
            "level_port": args.level_port,
            "moog_mode": args.moog_mode,
            "moog_port": args.moog_port if args.moog_mode == "moog-functions" else None,
        },
        "fit": fit,
        "files": {
            "csv": os.fspath(csv_path),
            "json": os.fspath(json_path),
            "plot": None if args.no_plot else os.fspath(plot_path),
        },
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.no_plot:
        write_plot(plot_path, samples, fit)

    print("\nFit result:")
    print(json.dumps(result["fit"], indent=2))
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

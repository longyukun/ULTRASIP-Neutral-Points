#!/usr/bin/env python3
"""Read PRO3600 angle output from a serial port on Windows."""

from __future__ import annotations

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Missing dependency: pyserial", file=sys.stderr)
    print("Install it with: py -m pip install pyserial", file=sys.stderr)
    raise SystemExit(1)


DEFAULT_BAUD = 9600
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


def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")
    for port in ports:
        description = port.description or "unknown device"
        print(f"  {port.device}  {description}")


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


def print_trigger_message(trigger: str) -> None:
    if trigger == "tx-break-pulse":
        print("REQ trigger: TXD break pulse")
        print("Default Windows mode: repeat one pulse every 1 second.")
    elif trigger == "tx-break":
        print("REQ trigger: TXD held in break state")
    elif trigger == "rts":
        print("REQ trigger: RTS high")
    elif trigger == "rts-pulse":
        print("REQ trigger: RTS low-to-high pulse")
    elif trigger == "dtr":
        print("REQ trigger: DTR high")
    elif trigger == "dtr-pulse":
        print("REQ trigger: DTR low-to-high pulse")
    else:
        print("REQ trigger: none controlled by this script")


def read_angles(args: argparse.Namespace) -> None:
    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=args.bytesize,
        parity=PARITY_CHOICES[args.parity],
        stopbits=STOPBITS_CHOICES[args.stopbits],
        timeout=args.timeout,
    )

    try:
        print_trigger_message(args.trigger)
        apply_trigger(ser, args.trigger, args.pulse_width)

        if args.trigger_delay > 0:
            time.sleep(args.trigger_delay)

        print(
            f"Start reading PRO3600 on {args.port} at {args.baud} baud, "
            f"{args.bytesize}{args.parity[0].upper()}{args.stopbits}. Ctrl+C to stop."
        )

        buffer = bytearray()
        last_status = time.monotonic()
        last_trigger = time.monotonic()

        while True:
            if args.repeat_trigger > 0 and time.monotonic() - last_trigger >= args.repeat_trigger:
                apply_trigger(ser, args.trigger, args.pulse_width)
                last_trigger = time.monotonic()

            raw = ser.read(ser.in_waiting or 1)
            if raw:
                if args.raw:
                    text = raw.decode("ascii", errors="replace")
                    print(f"RAW ascii={text!r} hex={raw.hex(' ')}")
                else:
                    buffer.extend(raw)
                    while b"\n" in buffer or b"\r" in buffer:
                        positions = [pos for pos in (buffer.find(b"\n"), buffer.find(b"\r")) if pos >= 0]
                        split_at = min(positions)
                        line_bytes = bytes(buffer[:split_at])
                        del buffer[: split_at + 1]
                        line = line_bytes.decode("ascii", errors="ignore").strip()
                        if line:
                            print(f"Angle: {line}")
                    if len(buffer) > 80:
                        line = bytes(buffer).decode("ascii", errors="ignore").strip()
                        buffer.clear()
                        if line:
                            print(f"Angle: {line}")
                last_status = time.monotonic()
            elif time.monotonic() - last_status >= 2.0:
                print("Waiting for data...")
                last_status = time.monotonic()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        release_trigger(ser, args.trigger)
        ser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read PRO3600 serial angle output on Windows.")
    parser.add_argument("-p", "--port", default="COM3", help="Windows COM port. Default: COM3.")
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD, help="Baud rate. Default: 9600.")
    parser.add_argument("--bytesize", type=int, choices=(5, 6, 7, 8), default=8, help="Data bits. Default: 8.")
    parser.add_argument("--parity", choices=tuple(PARITY_CHOICES), default="none", help="Parity. Default: none.")
    parser.add_argument("--stopbits", choices=tuple(STOPBITS_CHOICES), default="1", help="Stop bits. Default: 1.")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit.")
    parser.add_argument("--trigger", choices=TRIGGER_CHOICES, default=DEFAULT_TRIGGER, help="REQ trigger mode.")
    parser.add_argument("--trigger-delay", type=float, default=0.2, help="Delay after first trigger. Default: 0.2.")
    parser.add_argument("--pulse-width", type=float, default=DEFAULT_PULSE_WIDTH, help="Pulse width. Default: 0.2.")
    parser.add_argument(
        "--repeat-trigger",
        type=float,
        default=DEFAULT_REPEAT_TRIGGER,
        help="Repeat selected trigger every N seconds. Default: 1.",
    )
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout. Default: 1.0.")
    parser.add_argument("--interval", type=float, default=0.05, help="Loop sleep interval. Default: 0.05.")
    parser.add_argument("--raw", action="store_true", help="Print raw bytes as ASCII and HEX.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print_ports()
        return
    read_angles(args)


if __name__ == "__main__":
    main()

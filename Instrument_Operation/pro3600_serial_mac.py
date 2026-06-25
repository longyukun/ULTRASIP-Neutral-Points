#!/usr/bin/env python3
"""Read PRO3600 angle output from a serial port on macOS."""

from __future__ import annotations

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Missing dependency: pyserial", file=sys.stderr)
    print("Install it with: python3 -m pip install pyserial", file=sys.stderr)
    raise SystemExit(1)


DEFAULT_BAUD = 9600
TRIGGER_CHOICES = (
    "rts",
    "rts-pulse",
    "dtr",
    "dtr-pulse",
    "tx",
    "tx-pulse",
    "tx-break",
    "tx-break-pulse",
    "none",
)
PARITY_CHOICES = {
    "none": serial.PARITY_NONE,
    "even": serial.PARITY_EVEN,
    "odd": serial.PARITY_ODD,
    "mark": serial.PARITY_MARK,
    "space": serial.PARITY_SPACE,
}
STOPBITS_CHOICES = {
    "1": serial.STOPBITS_ONE,
    "1.5": serial.STOPBITS_ONE_POINT_FIVE,
    "2": serial.STOPBITS_TWO,
}


def available_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")
    for port in ports:
        description = port.description or "unknown device"
        print(f"  {port.device}  {description}")


def choose_port(port_arg: str | None) -> str:
    if port_arg:
        return port_arg

    ports = available_ports()
    cu_ports = [port for port in ports if port.startswith("/dev/cu.")]

    if len(cu_ports) == 1:
        return cu_ports[0]

    if not ports:
        print("No serial ports found. Connect the USB serial adapter and try again.", file=sys.stderr)
    else:
        print("Please choose one port with --port. Current ports:", file=sys.stderr)
        for port in ports:
            print(f"  {port}", file=sys.stderr)

    raise SystemExit(2)


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
    elif trigger == "tx-pulse":
        ser.break_condition = True
        time.sleep(pulse_width)
        ser.break_condition = False
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
    elif trigger in ("tx-pulse", "tx-break", "tx-break-pulse"):
        ser.break_condition = False


def print_trigger_message(trigger: str) -> None:
    if trigger == "rts":
        print("REQ trigger: RTS high")
    elif trigger == "rts-pulse":
        print("REQ trigger: RTS low-to-high pulse")
    elif trigger == "dtr":
        print("REQ trigger: DTR high")
    elif trigger == "dtr-pulse":
        print("REQ trigger: DTR low-to-high pulse")
    elif trigger == "tx":
        print("REQ trigger: TXD idle high")
        print("Do not send data on TXD while it is connected to PRO3600 Pin 5 REQ.")
    elif trigger == "tx-pulse":
        print("REQ trigger: TXD break pulse")
        print("For RS232 TXD, break is usually positive voltage and can trigger REQ.")
        print("Do not use this if TXD is connected to PRO3600 Pin 2 TD.")
    elif trigger == "tx-break":
        print("REQ trigger: TXD held in break state")
        print("For RS232 TXD, this usually holds REQ high for continuous output.")
        print("Do not use this if TXD is connected to PRO3600 Pin 2 TD.")
    elif trigger == "tx-break-pulse":
        print("REQ trigger: TXD break pulse")
        print("For RS232 TXD, this gives REQ a positive pulse.")
        print("Do not use this if TXD is connected to PRO3600 Pin 2 TD.")
    else:
        print("REQ trigger: none controlled by this script")


def read_angles(
    port: str,
    baud: int,
    bytesize: int,
    parity: str,
    stopbits: str,
    trigger: str,
    trigger_delay: float,
    pulse_width: float,
    repeat_trigger: float,
    timeout: float,
    interval: float,
    raw_output: bool,
) -> None:
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=bytesize,
        parity=PARITY_CHOICES[parity],
        stopbits=STOPBITS_CHOICES[stopbits],
        timeout=timeout,
    )

    try:
        print_trigger_message(trigger)
        apply_trigger(ser, trigger, pulse_width)

        if trigger_delay > 0:
            time.sleep(trigger_delay)

        print(
            f"Start reading PRO3600 on {port} at {baud} baud, "
            f"{bytesize}{parity[0].upper()}{stopbits}. Ctrl+C to stop."
        )

        buffer = bytearray()
        last_status = time.monotonic()
        last_trigger = time.monotonic()

        while True:
            if repeat_trigger > 0 and time.monotonic() - last_trigger >= repeat_trigger:
                apply_trigger(ser, trigger, pulse_width)
                last_trigger = time.monotonic()

            waiting = ser.in_waiting
            raw = ser.read(waiting or 1)
            if raw:
                if raw_output:
                    text = raw.decode("ascii", errors="replace")
                    hex_text = raw.hex(" ")
                    print(f"RAW ascii={text!r} hex={hex_text}")
                else:
                    buffer.extend(raw)
                    while b"\n" in buffer or b"\r" in buffer:
                        newline_positions = [
                            pos for pos in (buffer.find(b"\n"), buffer.find(b"\r")) if pos >= 0
                        ]
                        split_at = min(newline_positions)
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
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        release_trigger(ser, trigger)
        ser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read PRO3600 serial angle output on macOS."
    )
    parser.add_argument(
        "-p",
        "--port",
        help="Serial port, for example /dev/cu.usbserial-0001 or /dev/cu.SLAB_USBtoUART.",
    )
    parser.add_argument(
        "-b",
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate. Default: {DEFAULT_BAUD}.",
    )
    parser.add_argument(
        "--bytesize",
        type=int,
        choices=(5, 6, 7, 8),
        default=8,
        help="Data bits. Default: 8.",
    )
    parser.add_argument(
        "--parity",
        choices=tuple(PARITY_CHOICES),
        default="none",
        help="Parity. Default: none.",
    )
    parser.add_argument(
        "--stopbits",
        choices=tuple(STOPBITS_CHOICES),
        default="1",
        help="Stop bits. Default: 1.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List serial ports and exit.",
    )
    parser.add_argument(
        "--trigger",
        choices=TRIGGER_CHOICES,
        default="rts",
        help=(
            "How PRO3600 Pin 5 REQ is pulled high. "
            "Use rts/rts-pulse if REQ is wired to USB-TTL RTS, "
            "dtr/dtr-pulse if wired to DTR, tx if wired to TXD idle high, "
            "tx-break if RS232 TXD should hold REQ high, tx-break-pulse for RS232 TXD pulses, "
            "or none if REQ is tied high externally. "
            "Default: rts."
        ),
    )
    parser.add_argument(
        "--trigger-delay",
        type=float,
        default=0.2,
        help="Seconds to wait after applying the REQ trigger. Default: 0.2.",
    )
    parser.add_argument(
        "--pulse-width",
        type=float,
        default=0.1,
        help="Low time before the rising edge for pulse triggers, in seconds. Default: 0.1.",
    )
    parser.add_argument(
        "--repeat-trigger",
        type=float,
        default=0.0,
        help="Repeat the selected trigger every N seconds. Default: 0 means trigger once.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Serial read timeout in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Loop sleep interval in seconds. Default: 0.05.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw bytes as ASCII and HEX instead of parsing angle lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print_ports()
        return

    port = choose_port(args.port)
    read_angles(
        port=port,
        baud=args.baud,
        bytesize=args.bytesize,
        parity=args.parity,
        stopbits=args.stopbits,
        trigger=args.trigger,
        trigger_delay=args.trigger_delay,
        pulse_width=args.pulse_width,
        repeat_trigger=args.repeat_trigger,
        timeout=args.timeout,
        interval=args.interval,
        raw_output=args.raw,
    )


if __name__ == "__main__":
    main()

# Running the PRO3600 Level on Windows

This guide explains how to read angle data from a PRO3600 digital level on Windows using Python.

## Hardware Connection

Use a serial adapter that matches your wiring. The PRO3600 serial output is RS-232 compatible.

For the current TXD-trigger setup:

```text
PRO3600 Pin 1 GND  -> Serial adapter GND
PRO3600 Pin 2 TD   -> Serial adapter RXD
PRO3600 Pin 5 REQ  -> Serial adapter TXD
```

If the PRO3600 is not powered by its internal battery, provide external power:

```text
PRO3600 Pin 9 BATT+ -> 4.25V to 10V positive supply, nominal 9V
PRO3600 Pin 1 GND   -> Power supply ground
```

Make sure all grounds are connected together.

## Install Python Dependency

Open Command Prompt or PowerShell and install `pyserial`:

```bat
py -m pip install pyserial
```

## Find the COM Port

From the folder containing `pro3600_serial_windows.py`, run:

```bat
py pro3600_serial_windows.py --list
```

Look for the COM port assigned to the serial adapter, for example:

```text
COM3
COM4
COM5
```

## Run the Reader

Replace `COM3` with your actual COM port:

```bat
py pro3600_serial_windows.py --port COM3 --raw
```

The Windows script defaults to this trigger behavior:

```text
--trigger tx-break-pulse --repeat-trigger 1 --pulse-width 0.2
```

That means it sends a TXD break pulse every 1 second to trigger the PRO3600 REQ line.

The full equivalent command is:

```bat
py pro3600_serial_windows.py --port COM3 --trigger tx-break-pulse --repeat-trigger 1 --pulse-width 0.2 --raw
```

If the raw output shows readable ASCII angle values, run without `--raw`:

```bat
py pro3600_serial_windows.py --port COM3
```

Expected output format:

```text
Angle: + 12.34
Angle: -  4.32
```

Stop the program with:

```text
Ctrl+C
```

## Serial Settings

The PRO3600 manual specifies:

```text
Baud rate: 9600
Data bits: 8
Parity: none
Stop bits: 1
Format: ASCII angle followed by carriage return and line feed
```

These are the script defaults.

## Troubleshooting

If no COM port appears:

```text
Check the USB cable, adapter driver, and Windows Device Manager.
```

If the program prints `Waiting for data...`:

```text
Check PRO3600 power.
Check GND connection.
Check PRO3600 Pin 2 TD goes to adapter RXD.
Check PRO3600 Pin 5 REQ goes to adapter TXD for this script mode.
```

If output is garbled:

```text
Confirm the adapter is RS-232 compatible, not TTL-only.
Confirm the serial settings are 9600 8N1.
Confirm PRO3600 TD is connected to RXD, not TXD.
```

If TXD triggering does not work:

```text
Use a serial adapter with RTS or DTR control lines, or tie PRO3600 Pin 5 REQ to a valid high level.
```

For continuous output with REQ tied high, run:

```bat
py pro3600_serial_windows.py --port COM3 --trigger none --raw
```

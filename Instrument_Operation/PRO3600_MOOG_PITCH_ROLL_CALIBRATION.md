# PRO3600 + Moog Pitch/Roll Calibration

This procedure rotates Moog pan while a PRO3600 level sits on the Moog.  The
script records the level angle at each pan angle and fits:

```text
level_angle = offset + pitch*cos(pan) + roll*sin(pan)
```

The default scan is:

```text
0 -> -180 -> 0 -> 180 -> 0
```

with 5 degree pan steps, 2 seconds settle time at each point, repeated 3 times.

## 1. Install dependency on the Windows Moog computer

```bat
py -m pip install pyserial
```

`matplotlib` is optional.  If installed, the script also saves a diagnostic PNG.

```bat
py -m pip install matplotlib
```

## 2. Find and test the PRO3600 COM port

```bat
cd Instrument_Operation
py pro3600_moog_pitch_roll_calibration.py --list-ports
py pro3600_serial_windows.py --port COM3
```

Replace `COM3` with the actual PRO3600 serial port.

## 3. Run with Moog control

Run from the same folder as `moog_functions.py` so Python can import it.  The
script uses the same Moog path as `Measurement_QT_GUI.py`: it opens the Moog
serial port, calls `moog_functions.init_autobaud()`, then moves with
`moog_functions.move_to_coord_and_wait(serial_port, int(pan*10), int(tilt*10))`.

```bat
cd Instrument_Operation
py pro3600_moog_pitch_roll_calibration.py --level-port COM3 --moog-port COM7 --output-dir calibration_outputs
```

`COM3` is the PRO3600 level port.  `COM7` is the Moog port used by the GUI
defaults; change both to match Windows Device Manager.

## 4. Run through a command adapter

If Moog control is easier as a separate command, use:

```bat
py pro3600_moog_pitch_roll_calibration.py --port COM3 --moog-mode command --moog-command "py move_moog.py --pan {pan}" --output-dir calibration_outputs
```

`{pan}` is replaced by the current pan angle.

## 5. Check the scan path without hardware

This prints the target points without opening the serial port or moving Moog:

```bat
py pro3600_moog_pitch_roll_calibration.py --plan-only
```

## Outputs

The script writes files like:

```text
pro3600_moog_pitch_roll_YYYYMMDD_HHMMSS.csv
pro3600_moog_pitch_roll_YYYYMMDD_HHMMSS.json
pro3600_moog_pitch_roll_YYYYMMDD_HHMMSS.png
```

The JSON contains:

```text
offset_deg
pitch_deg
roll_deg
amplitude_deg
phase_deg
rms_residual_deg
```

The pitch/roll sign depends on the physical direction of the PRO3600 on the
Moog.  Use the CSV and PNG to confirm whether either sign should be flipped for
the instrument coordinate convention.

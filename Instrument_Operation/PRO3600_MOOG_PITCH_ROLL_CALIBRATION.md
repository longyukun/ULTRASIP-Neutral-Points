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

## 3. Run with Moog Python control

Run from the same folder as `moog_functions.py` so Python can import it:

```bat
cd Instrument_Operation
py pro3600_moog_pitch_roll_calibration.py --port COM3 --output-dir calibration_outputs
```

If automatic Moog function detection does not find the right move function,
pass the exact function names:

```bat
py pro3600_moog_pitch_roll_calibration.py --port COM3 --moog-function move_moog --output-dir calibration_outputs
```

If `moog_functions.py` needs explicit open/close calls:

```bat
py pro3600_moog_pitch_roll_calibration.py --port COM3 --moog-init connect_moog --moog-function move_moog --moog-close close_moog --output-dir calibration_outputs
```

The move function can accept any of these common forms:

```python
move(pan)
move(pan, tilt)
move(pan=pan, tilt=tilt)
move(azimuth=pan, elevation=tilt)
```

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

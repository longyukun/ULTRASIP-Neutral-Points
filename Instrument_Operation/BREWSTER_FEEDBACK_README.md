# INC1000 Brewster Point Feedback Scan

`brewster_feedback_scan.py` is a standalone acquisition script for rapid field
validation. It reuses the existing solar-alignment GUI and does not apply
non-uniformity correction (NUC).

## Field workflow

1. Use `Measurement_QT_GUI.py` to aim at and center the Sun. Finish calibration
   to save the pan offset, then close the GUI to release the camera and ports.
2. Start the feedback scan at **zenith 70° (altitude 20°)**. The default altitude
   increment is **2°**, moving upward toward the Sun.
3. At each level: use INC1000 altitude feedback, wait for settling, acquire four
   polarizer orientations, rotate Q/U using camera roll, fit U/I versus horizontal
   image angle, and adjust pan to center the U zero line.
4. Recheck altitude after each pan move. Record Q/I sign changes as candidate
   neutral-point brackets, but continue toward the solar altitude. Stop and
   record the reason if the fit is invalid or feedback does not converge.

## Running the script

Run from the repository root in the existing camera-acquisition environment.
Dependencies are numpy, pyserial, zaber-motion, and VmbPy. Live plots require
matplotlib; solar calibration requires the existing Qt environment.

Test without hardware:

```powershell
py Instrument_Operation/brewster_feedback_scan.py --simulate --plot
```

Open the original solar-calibration GUI first. Complete calibration and close
that window to continue into the scan:

```powershell
py Instrument_Operation/brewster_feedback_scan.py --calibrate-sun --roll-sign 1 --plot --output D:/Data/Brewster
```

If solar alignment has already been calibrated for the current installation:

```powershell
py Instrument_Operation/brewster_feedback_scan.py --roll-sign 1 --plot --output D:/Data/Brewster
```

Use `--roll-sign -1` for the opposite INC1000 X convention. The script does not
infer the correct sign from fit quality. Verify it using a known camera rotation
or an independent image-level reference.

Default connections are INC1000 on COM1 at 115200 baud, Moog on COM7, and Zaber
on COM6. Override them with `--inc-port`, `--inc-baud`, `--moog-port`, and
`--zaber-port`.

## Attitude and solar alignment

- `altitude = -INC1000_Y + tilt_offset`; `zenith = 90 - altitude`.
- `roll = roll_sign * INC1000_X + roll_offset`.
- `--tilt-offset` corrects the sensor-to-optical-axis altitude offset. It is
  separate from the original GUI's Moog tilt offset.
- `--roll-offset` includes the fixed polarization-reference offset, in degrees.
- This is a provisional model for the current mounting convention, not a full
  conversion from the inclinometer's two axis readings to Euler angles.
- The script reads `pan_offset`, `latitude`, and `longitude` from the original
  GUI's `~/.ultrasip_auto_scan_settings.json`. Use `--settings` for another file,
  or override values with `--pan-offset`, `--latitude`, and `--longitude`.
  Repeat solar alignment after moving the base.
- Initial pan follows the existing convention: solar Moog/SunCalc azimuth minus
  `pan_offset`. Subsequent tracking uses the image feedback; the Moog pan reading
  is not treated as an exact absolute azimuth.
- The scan refuses to start if the Sun is not above the starting altitude plus
  the requested margin. Solar altitude is recalculated at each level. The
  default endpoint is solar altitude; `--sun-margin 2` ends 2° below it.
- The original calibration GUI homes the mount when it closes. That behavior
  is retained. The feedback scan itself stops and releases connections on
  completion or interruption without an additional homing move.

## Fast processing and feedback

Acquire polarizer angles 0/45/90/135° with the same exposure for all four frames:

```text
I = (I0 + I45 + I90 + I135) / 2
Q = I0 - I90
U = I45 - I135
Q' = Q*cos(2*roll) + U*sin(2*roll)
U' = -Q*sin(2*roll) + U*cos(2*roll)
```

The default central horizontal band is approximately 800 pixels wide and 40
pixels high. Average valid U'/I samples in each column, fit `u = a*degree + b`,
and locate the zero at `-b/a`. Positive image angle follows increasing column
index. The default scale, inherited from the acquisition GUI, is 0.002°/pixel;
change it with `--deg-per-pixel`. Set `--center-x` and `--center-y` to the optical
axis pixel; otherwise the geometric image center is used.

A 0.2° pan probe measures the direction and magnitude of the actual response.
Image angle is not directly equated to motor pan. Each correction applies 70%
of the predicted adjustment, capped at **1° per move**, with a minimum requested
correction of 0.1° to account for Moog command resolution. The default centering
tolerance is 0.08° in image angle, and the altitude tolerance is 0.15°. Both are
configurable using their corresponding tolerance options.

Exposure defaults to 1000 µs; change it with `--exposure-us`. This version does
not adjust exposure automatically. Saturated or dark pixels are excluded; the
scan stops if too few valid samples remain, the slope is too small, the zero is
outside the ROI, the fit is too noisy, the pose drifts, or motion feedback fails
to converge. Saturation near the Sun may end a scan early; inspect the log and
reduce exposure before repeating it.

`root_se_deg` is only the formal standard error from the column-average linear
fit. It excludes NUC effects, correlated noise, optical-axis mounting errors,
and roll-sign errors. Candidate brackets are not calibrated neutral-point
measurements. This version does not automatically refine a bracket; repeat
with a smaller `--zenith-step` if needed.

## Saved data

Each run creates a separate timestamped directory containing:

- `config.json`: effective parameters and solar pan offset.
- `acquisition_XXXX/raw_000.npy`, etc.: four full raw frames, saved without
  compression to reduce processing overhead.
- `partial.json`: completed-frame timestamps and INC1000 readings before and
  after acquisition, retained even if the quartet is interrupted.
- `result.json` and `u_fit.npz`: attitude, fit results, and the fitted profile.
- `measurements.jsonl`: all acquisitions, including pan probes.
- `aligned.jsonl`: successfully centered results at each altitude level.
- `candidates.json`: candidate zenith-angle brackets, created when Q changes sign.
- `status.json`: completion, interruption, or failure status and its reason.

This standalone validation format is not a direct input to the existing H5
analysis pipeline. Saved raw frames can be corrected with NUC and analyzed more
precisely offline. Four 2848×2848 uint16 frames occupy approximately 65 MB per
quartet, so disk throughput also affects acquisition speed.

Press Ctrl+C to stop. The script does not automatically modify INC1000 zero,
baud-rate, address, or Flash settings.

## Hardware-free tests

```powershell
py -m unittest discover -s Instrument_Operation/tests -p "test_brewster_feedback_scan.py" -v
```

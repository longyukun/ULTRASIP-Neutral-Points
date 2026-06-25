# Measurement_QT_GUI.py

Qt GUI for operating the ULTRASIP instrument: controls the Moog pan/tilt mount, Zaber polarizer rotation stage, and Allied Vision UV camera to perform calibration and automated polarimetric measurements.

**Version:** 2.1.0

---

## Overview

The GUI has two pages:

- **Auto Measurement page** — the main operational view. Configure location, scan parameters, and exposure settings, then run an automated scan triggered by the sun reaching target altitudes.
- **Calibration page** — manually drive the mount and center the sun in the camera FOV to establish a pan/tilt offset before running a scan.

Scan data are written to HDF5 files (`.h5`), one per acquisition sequence.

---

## Requirements

### Python packages

| Package | Purpose | Required? |
|---|---|---|
| PySide6 / PyQt6 / PyQt5 | Qt GUI (tries each in order) | Yes (one of these) |
| numpy | Array math | Yes |
| h5py | HDF5 file output | Yes (for scans) |
| cv2 (OpenCV) | Sun center detection | Recommended |
| pyserial | Moog serial communication | Yes (for Moog) |
| zaber_motion | Zaber polarizer stage | Yes (for polarizer) |
| suncalc | Sun position (unused; internal implementation used instead) | Optional |

### Local modules (must be on Python path)

- `moog_functions` — serial protocol for the Moog pan/tilt controller
- `uv_cam_functions` — Allied Vision Vimba SDK wrapper for the UV camera

---

## Hardware

| Device | Default port | Notes |
|---|---|---|
| Moog pan/tilt | COM7 | RS-232, 9600 baud; pan ±217.5°, tilt ±90°, resolution 0.01° |
| Zaber polarizer stage | COM6 | ASCII protocol via `zaber_motion`; homes on connect |
| Allied Vision UV camera | (auto-detected) | 2848×2848 px, 12-bit, 5.78° FOV, 7.20 arcsec/px |

---

## Running

```bash
python Measurement_QT_GUI.py
```

The GUI tries PySide6, then PyQt6, then PyQt5. Install whichever you have.

---

## Workflow

### 1. Connect devices

On the Auto Measurement page, select COM ports for the Moog and Polarizer, then toggle each device switch on. The camera connects automatically (no port selection needed).

### 2. Calibrate (first time or when mount position is uncertain)

Click **Calibration** to open the calibration page. Use the pan/tilt jog controls to aim the mount at the sun, then click **Center sun in FOV** to compute and apply a pan offset that centers the sun. Click **Done Calibration** to record the offset and return to the Auto Measurement page.

### 3. Configure scan settings

Click **Acquisition Settings…** to set:

- Latitude/longitude of the site
- Sun altitude range and step size (target altitudes to trigger measurements at)
- Sun altitude tolerance for triggering (degrees)
- Tilt scan parameters (start offset and step between tilt positions)
- Polarizer angles (comma-separated, e.g. `0,45,90,135`)
- Auto-exposure mode and limits
- Output folder and location label

Settings are persisted to `~/.ultrasip_auto_scan_settings.json` automatically.

### 4. Run

Click **Start Auto Scan**. If sun-trigger mode is enabled (default), the GUI monitors the sun's altitude at the configured check interval and fires a measurement sequence each time the sun crosses a target altitude. If sun-trigger is disabled, the scan runs immediately.

Each measurement sequence:
1. Moves to each tilt offset in the scan plan
2. For each tilt position, cycles through all configured polarizer angles and captures one UV image per angle
3. Writes everything to an HDF5 file in `<output_dir>/<YYYY_MM_DD>/`

### 5. Pitch/Roll calibration scan (optional)

Used to characterize mount pitch/roll errors. Rasters a configurable rectangle around the sun, capturing polarizer-0° images into a `*_calibration.h5` file. Configure the grid extent and step on the Auto Measurement page, then click **Start Pitch/Roll Calibration Scan**.

---

## Output files

HDF5 files are saved to `<output_dir>/<YYYY_MM_DD>/` with names like:

```
<Location>_<YYYYMMDD>_<HH_MM_SS>.h5              ← normal scan
<Location>_<YYYYMMDD>_<HH_MM_SS>_calibration.h5  ← pitch/roll raster
```

### File structure

```
<file>.h5
├── Measurement_Metadata/        ← scan-level metadata (attrs only, no datasets)
├── Aquistion_0/                 ← normal acquisition, dtilt nearest sun  [note: typo in code]
│   └── UV Image Data/
│       └── UV Raw Images        ← dataset: uint16 (n_angles, 2848, 2848)
├── Aquistion_1/
│   └── UV Image Data/ ...
├── ...
├── calibration_acqui_up_0p5/    ← calibration acquisition +0.5 deg tilt
│   └── UV Image Data/ ...
└── ...
```

For pitch/roll calibration files the acquisition groups are named `Calibration_NNN_rRR_cCC`.

---

### `Measurement_Metadata` attributes (normal scan)

| Attribute | Type | Description |
|---|---|---|
| `Software Version` | str | App version string |
| `Latitude` | str | Site latitude (degrees) |
| `Longitude` | str | Site longitude (degrees) |
| `Pan_Offset` | str | Pan calibration offset applied (degrees) |
| `Tilt_Offset` | str | Tilt calibration offset applied (degrees) |
| `Location` | str | Site label |
| `Trigger Target Altitude [deg]` | float64 | Sun altitude that triggered this scan (NaN if triggered manually) |
| `Trigger Actual Altitude [deg]` | float64 | Actual sun altitude at trigger time (NaN if manual) |
| `Trigger Azimuth [deg]` | float64 | Sun azimuth at trigger time (NaN if manual) |
| `Trigger Timestamp` | str | ISO-8601 timestamp of trigger |
| `Scan Initial Sun Altitude [deg]` | float64 | Sun altitude when scan started |
| `Scan Initial SZA [deg]` | float64 | Solar zenith angle when scan started |
| `Scan Initial Tilt Base [deg]` | float64 | Base tilt = initial sun altitude |
| `Scan Tilt Mode` | str | `"initial_sun_altitude_plus_delta_tilt"` |
| `Scan Order` | str | `"zenith_to_sun"` or `"sun_to_zenith"` |
| `Acquisition Numbering` | str | `"small_near_sun_large_near_zenith"` |
| `Normal Acquisition Execution Order` | int64 array | Execution order of normal acquisition indices |
| `Calibration Acquisition Order` | str | When calibration acquisitions run relative to normal |
| `Scan Start Tilt [deg]` | float64 | Minimum dtilt (GUI "Scan start tilt offset") |
| `Scan Start Delta Tilt [deg]` | float64 | Same as above |
| `Scan Computed End Tilt [deg]` | float64 | Maximum absolute tilt reached |
| `Scan Computed End Delta Tilt [deg]` | float64 | Maximum dtilt reached |
| `Scan Step Tilt [deg]` | float64 | Step between tilt positions |
| `Calibration Delta Tilts [deg]` | float64 array | Fixed calibration offsets `[−0.5, +0.5, −1.0, +1.0, −1.5, +1.5]` |
| `Auto Exposure Enabled` | int (0/1) | Whether auto-exposure was active |
| `Auto Exposure Metric` | str | `"median"` or `"max"` |
| `Auto Exposure Target Median` | float64 | Target pixel median for median mode |
| `Auto Exposure Initial Test [us]` | float64 | Test exposure used during metering (100 000 µs) |
| `Auto Exposure Fixed Near Sun [us]` | float64 | Fixed exposure for dtilt ≤ 3° (800 µs) |
| `Auto Exposure Fixed Near Sun Enabled` | int (0/1) | Always 1 |
| `Auto Exposure Metering Angles [deg]` | float64 array | Polarizer angles used for metering (0° and 90°) |
| `Auto Exposure Metering Frames` | int | Frames averaged per angle during metering (5) |
| `Auto Exposure Min [us]` | float64 | Clamp lower bound (100 µs) |
| `Auto Exposure Max [us]` | float64 | Clamp upper bound |
| `UV Image Width [px]` | int | 2848 |
| `UV Image Height [px]` | int | 2848 |
| `UV Pixel Scale [deg/pixel]` | float64 | 0.002 °/px (7.20 arcsec/px) |
| `Moog Command Resolution [deg]` | float64 | 0.1° (integer command units × 0.1) |
| `Moog Settle Before Capture [s]` | float64 | 2.0 s wait after move |
| `Polarizer Settle After Idle [s]` | float64 | 0.5 s wait after polarizer move |
| `Zenith Wait Between Groups Enabled` | int (0/1) | 1 when scan order is zenith-to-sun |
| `Zenith Wait Delta Tilt [deg]` | float64 | dtilt of the zenith-wait position |
| `Zenith Wait Tilt [deg]` | float64 | Absolute tilt of the zenith-wait position |
| `Total Measurement Time` | float64 | Total scan duration (minutes) |
| `Acquisition Stopped` | int (0/1) | 1 if scan was interrupted |
| `Completed Acquisitions` | int | Total groups written |
| `Completed Normal Acquisitions` | int | Normal groups written |
| `Completed Calibration Acquisitions` | int | Calibration groups written |

---

### Per-acquisition group attributes

These appear on every `Aquistion_N` and `calibration_acqui_*` group.

#### Identity

| Attribute | Type | Description |
|---|---|---|
| `Acquisition Type` | str | `"normal"` or `"calibration"` |
| `Normal Acquisition Index` | int | Index within normal plan (normal only) |
| `Calibration Acquisition Index` | int | Index within calibration plan (calibration only) |
| `Calibration Direction` | str | `"up"`, `"down"`, or `"center"` (calibration only) |
| `Calibration Step Magnitude [deg]` | float64 | Abs value of calibration dtilt (calibration only) |

#### Timestamps

| Attribute | Type | Description |
|---|---|---|
| `Timestamp Local` | str | `HH_MM_SS` at command time |
| `Timestamp Local New` | str | ISO-8601 with microseconds, local TZ |
| `Timestamp UTC New` | str | ISO-8601 with microseconds, UTC (`…Z`) |
| `Timestamp Local Time Zone New` | str | TZ name (e.g. `"MST"`) |
| `Timestamp Local UTC Offset New` | str | UTC offset (e.g. `"-0700"`) |
| `Sun Pre Capture Timestamp Local` | str | ISO-8601 local, just before capture |
| `Sun Pre Capture Timestamp UTC` | str | ISO-8601 UTC, just before capture |
| `Sun Post Capture Timestamp Local` | str | ISO-8601 local, just after capture |
| `Sun Post Capture Timestamp UTC` | str | ISO-8601 UTC, just after capture |

#### Solar position

| Attribute | Type | Description |
|---|---|---|
| `Pointing Calculation Sun Position Azimuth` | float64 | Azimuth used to compute command (degrees, N=0) |
| `Pointing Calculation Sun Position Altitude` | float64 | Altitude used to compute command (degrees) |
| `Pointing Calculation Sun Position SZA` | float64 | SZA used to compute command (degrees) |
| `Sun Position Azimuth` | float64 | Azimuth at pre-capture time |
| `Sun Position Altitude` | float64 | Altitude at pre-capture time |
| `Sun Position SZA` | float64 | SZA at pre-capture time |
| `Sun Pre Capture Azimuth [deg]` | float64 | Same as `Sun Position Azimuth` |
| `Sun Pre Capture Altitude [deg]` | float64 | Same as `Sun Position Altitude` |
| `Sun Pre Capture SZA [deg]` | float64 | Same as `Sun Position SZA` |
| `Sun Post Capture Azimuth [deg]` | float64 | Azimuth at post-capture time |
| `Sun Post Capture Altitude [deg]` | float64 | Altitude at post-capture time |
| `Sun Post Capture SZA [deg]` | float64 | SZA at post-capture time |
| `Initial Sun Altitude [deg]` | float64 | Sun altitude when scan sequence started |
| `Initial SZA [deg]` | float64 | SZA when scan sequence started |
| `Initial Tilt Base [deg]` | float64 | Base tilt = initial sun altitude |

#### Pointing

| Attribute | Type | Description |
|---|---|---|
| `Pan` | float64 | Commanded pan (degrees) |
| `Tilt` | float64 | Commanded tilt = tilt_base + dtilt (degrees) |
| `Moog Command Tilt [deg]` | float64 | Same as `Tilt` |
| `Pan Offset` | float64 | Pan calibration offset applied |
| `Tilt Offset` | float64 | Tilt calibration offset applied |
| `Delta Tilt From Sun [deg]` | float64 | dtilt for this acquisition |
| `Delta Tilt From Initial Sun Altitude [deg]` | float64 | Same as above |
| `Moog Requested Pan [deg]` | float64 | Pan sent to Moog (= Pan − Pan Offset) |
| `Moog Requested Tilt [deg]` | float64 | Tilt sent to Moog (= Tilt − Tilt Offset) |
| `Moog Target Pan [deg]` | float64 | Quantized command pan |
| `Moog Target Tilt [deg]` | float64 | Quantized command tilt |
| `Moog Actual Pan [deg]` | float64 | Reported pan at arrival |
| `Moog Actual Tilt [deg]` | float64 | Reported tilt at arrival |
| `Moog Pan Error [deg]` | float64 | Actual − Target pan |
| `Moog Tilt Error [deg]` | float64 | Actual − Target tilt |
| `Moog Move Complete` | int (0/1) | Move-complete bit at arrival |
| `Moog Status Raw Bytes` | bytes / empty list | Raw serial response bytes |
| `Moog EXEC Bit` | int (0/1) | Executing bit from status |
| `Moog Moving Bits` | str (JSON) | `{"cw","ccw","up","down"}` motion bits |
| `Moog Soft Limit Bits` | str (JSON) | `{"pan_cw","pan_ccw","tilt_up","tilt_down"}` soft limits |
| `Moog Hard Limit Bits` | str (JSON) | Same keys, hard limits |
| `Moog Post Capture Actual Pan [deg]` | float64 | Reported pan after capture |
| `Moog Post Capture Actual Tilt [deg]` | float64 | Reported tilt after capture |
| `Moog Post Capture Pan Error [deg]` | float64 | Actual − Target pan after capture |
| `Moog Post Capture Tilt Error [deg]` | float64 | Actual − Target tilt after capture |
| `Moog Post Capture Move Complete` | int (0/1) | Move-complete bit after capture |
| `Moog Post Capture Status Raw Bytes` | bytes / empty list | Raw serial response after capture |
| `Moog Post Capture EXEC Bit` | int (0/1) | Executing bit after capture |
| `Moog Post Capture Moving Bits` | str (JSON) | Motion bits after capture |
| `Moog Post Capture Soft Limit Bits` | str (JSON) | Soft limits after capture |
| `Moog Post Capture Hard Limit Bits` | str (JSON) | Hard limits after capture |

---

### `UV Image Data` sub-group

#### Dataset

| Name | dtype | Shape | Compression | Description |
|---|---|---|---|---|
| `UV Raw Images` | uint16 | `(n_angles, 2848, 2848)` | gzip | Raw 12-bit camera frames, one per polarizer angle |

#### Attributes

| Attribute | Type | Description |
|---|---|---|
| `UV Exposure Time` | float64 | Actual exposure used for capture (µs) |
| `UV Auto Exposure Enabled` | int (0/1) | Whether auto-exposure was active |
| `UV Auto Exposure Metric` | str | `"median"` or `"max"` |
| `UV Auto Exposure Info` | str | Human-readable metering summary |
| `UV Auto Exposure Min [us]` | float64 | Lower clamp (100 µs) |
| `UV Auto Exposure Max [us]` | float64 | Upper clamp |
| `UV Bandpass` | str | `"355 FWHM 10nm"` |
| `UV Image Capture Time` | float64 | Wall time for the full polarizer sweep (seconds) |
| `UV Polarizer Angles` | str | String repr of requested angle list |
| `UV Polarizer Requested Angles [deg]` | float64 array | Commanded polarizer angles |
| `UV Polarizer Actual Angles [deg]` | float64 array | Measured polarizer positions |
| `UV Polarizer Angle Errors [deg]` | float64 array | Actual − Requested per angle |
| `UV Polarizer Position Tolerance [deg]` | float64 | Tolerance threshold (0.05°) |
| `UV Polarizer Position In Tolerance` | int (0/1) | 1 if all angle errors ≤ tolerance |
| `UV Image Shape` | str | String repr of array shape, e.g. `"(4, 2848, 2848)"` |

---

### Additional attributes for pitch/roll calibration files

`Measurement_Metadata` gains these extra attributes:

| Attribute | Type | Description |
|---|---|---|
| `Purpose` | str | `"Pitch/Roll pointing calibration raster scan around the sun"` |
| `Scan Pattern` | str | `"row_by_row_top_to_bottom_left_to_right"` |
| `Scan Up Range [deg]` | float64 | Tilt range above sun |
| `Scan Down Range [deg]` | float64 | Tilt range below sun |
| `Scan Left Range [deg]` | float64 | Pan range left of sun |
| `Scan Right Range [deg]` | float64 | Pan range right of sun |
| `Scan Step [deg]` | float64 | Grid step size |
| `Scan Grid Rows` | int | Number of tilt rows |
| `Scan Grid Cols` | int | Number of pan columns |
| `Scan Grid Delta Tilts [deg]` | float64 array | Tilt offsets for each row |
| `Scan Grid Delta Pans [deg]` | float64 array | Pan offsets for each column |
| `Scan Sun Tracking` | str | `"sun_position_recomputed_at_each_grid_point"` |
| `Polarizer Angle [deg]` | float64 | Fixed angle used (0°) |
| `Capture Exposure [us]` | float64 | Exposure used for all frames |
| `Planned Acquisitions` | int | Total grid points planned |

Each `Calibration_NNN_rRR_cCC` group gains:

| Attribute | Type | Description |
|---|---|---|
| `Grid Index` | int | Sequential index across the full grid |
| `Grid Row` | int | Row index (tilt axis) |
| `Grid Col` | int | Column index (pan axis) |
| `Grid Delta Pan [deg]` | float64 | Pan offset from sun for this point |
| `Grid Delta Tilt [deg]` | float64 | Tilt offset from sun for this point |
| `Detected Sun Center X [px]` | float64 | X centroid in image (if detected) |
| `Detected Sun Center Y [px]` | float64 | Y centroid in image (if detected) |
| `Detected Sun Center OK` | int (0/1) | 1 if centroid detection succeeded |
| `Detected Sun Center Info` | str (JSON) | Detection diagnostics |

---

## HTTP remote control API

Enable via the **HTTP API** toggle on the calibration page (default port 8765, localhost only). All endpoints use GET.

| Endpoint | Parameters | Description |
|---|---|---|
| `/status` | — | Returns current pan, tilt, polarizer, and camera state |
| `/move` | `dpan`, `dtilt` | Relative jog (degrees) |
| `/goto` | `pan`, `tilt` | Absolute move (degrees) |
| `/center` | — | Center sun in FOV |
| `/polarizer` | `angle` | Move polarizer to angle (degrees) |
| `/exposure` | `us` or `auto` | Set exposure time or trigger auto-exposure |

---

## Key constants

| Constant | Value | Meaning |
|---|---|---|
| `UV_FULL_FOV_DEG` | 5.78° | Camera full field of view |
| `UV_ARCSEC_PER_PIXEL` | 7.20 | Camera plate scale |
| `MOOG_SETTLE_BEFORE_CAPTURE_SEC` | 2.0 s | Wait after move before capture |
| `AUTO_EXPOSURE_TARGET_MEDIAN` | 2600 | Default median target (12-bit scale) |
| `CALIBRATION_DTILTS_DEG` | ±0.5, ±1.0, ±1.5 | Tilt offsets used in calibration scans |
| `DEFAULT_MOOG_PORT` | COM7 | Default serial port for Moog |
| `DEFAULT_ZABER_PORT` | COM6 | Default serial port for Zaber |

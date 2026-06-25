# README: `process_lacie_0602_csv_h5_batch.py`

This document describes the batch processing flow driven by:

```bash
process_lacie_0602_csv_h5_batch.py
```

The script is a batch wrapper for processing many ULTRASIP H5 files. For each selected H5 file it runs:

1. `process_single_level0_1_2.py`
2. `robust_np_all_acquisitions.py`

The first script writes Level 0, Level 1, and Level 2 products into the H5 file. The second script reads those products and writes a robust neutral-point CSV.

---

## Quick Start

```bash
cd Data_Analysis_Visualization

python process_lacie_0602_csv_h5_batch.py \
  "/Volumes/LaCie/Level2 data/2026_06_02" \
  --mode with-csv \
  --start-at 1 \
  --nuc NUC_0813.npz \
  --wmatrix ULTRASIP_AvgWmatrix_15.npy \
  --workers 4 \
  --save-diagnostics 0 \
  --min-sun-zen-separation 1.0
```

---

## Command Line Arguments

```bash
python process_lacie_0602_csv_h5_batch.py [directory] [options]
```

### Positional

`directory` — directory containing H5 files. Default: `/Volumes/LaCie/Level2 data/2026_06_02`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {all,with-csv,without-csv}` | `with-csv` | `all`: every H5; `with-csv`: only those that already have a sibling robust NP CSV; `without-csv`: only those that don't. |
| `--start-at INT` | `1` | One-based index into the selected file list. Useful for resuming after a failure. |
| `--nuc PATH` | `NUC_0813.npz` | NUC calibration file (flat-field + background). |
| `--wmatrix PATH` | `ULTRASIP_AvgWmatrix_15.npy` | W-matrix file for Stokes inversion. |
| `--workers INT` | `4` | Worker threads for `robust_np_all_acquisitions.py` (via `NP_WORKERS`). |
| `--save-diagnostics {0,1}` | `0` | Save diagnostic figures from robust NP (via `NP_SAVE_DIAGNOSTICS`). |
| `--min-sun-zen-separation FLOAT` | `1.0` | Reject NP candidates within this many degrees of the sun zenith (via `NP_MIN_SUN_ZEN_SEPARATION_DEG`). |

The sibling CSV path is `<h5_stem>_robust_np_methods.csv`.

---

## Batch Script Flow

1. Read the target directory.
2. Select real H5 files, skipping AppleDouble files (`._file.h5`).
3. Filter files according to `--mode`.
4. Set processing environment variables.
5. For each selected file run:

```bash
python process_single_level0_1_2.py <h5_path> <nuc_path> <wmatrix_path>
python robust_np_all_acquisitions.py <h5_path>
```

If a file fails, the batch continues. At the end it reports all failed files and exits with code `1`.

---

## Level 0 to Level 2 Processing (`process_single_level0_1_2.py`)

Single-file usage:

```bash
python process_single_level0_1_2.py \
  /path/to/file.h5 \
  /path/to/NUC_0813.npz \
  /path/to/ULTRASIP_AvgWmatrix_15.npy
```

The H5 file is opened in `r+` mode. Existing datasets with the same output names are overwritten.

### Level 0: Raw UV Images → Corrected Stokes

Function: `process_level0(handle, nuc_path, wmatrix_path)`

The raw UV image stack is reshaped to `(4, 2848, 2848)` and interpreted as `p0, p45, p90, p135`. Flat-field and background correction is applied per channel:

```python
corrected = (mean(rij) / rij) * (pij - bij) + mean(bij)
```

The corrected channel stack is multiplied by the pseudo-inverse W matrix (SVD-based):

```python
[I_corrected, Q_corrected, U_corrected] = pinv(W) @ [C0, C90, C45, C135]
```

If no NUC/W-matrix paths are supplied, the script falls back to an ideal W matrix. The batch script always supplies both.

**Outputs written per acquisition/calibration group:**

```text
I_corrected
Q_corrected
U_corrected
```

**Metadata written:**

```text
Measurement_Metadata/Processed Level = "Level 0"
Measurement_Metadata/Level 0 Calibration Note = <path info>
```

### Level 1: Corrected Stokes → Viewing Geometry

Function: `process_level1(handle)`

#### Geometry reference: image centre

Level 1 uses the **image centre** as the fixed geometry anchor for every frame:

```python
x_center = IMG_X / 2.0   # = 1424.0
y_center = IMG_Y / 2.0   # = 1424.0
```

The geometry grid is computed per pixel as:

```python
view_zen[row, col] = 90 - Tilt + (row - y_center) * VFOV
view_az [row, col] = Pan        + (col - x_center) * HFOV
```

At the centre pixel the offset terms vanish, so:

```text
view_zen[centre] = 90 - Moog_Tilt   [deg]
view_az [centre] = Moog_Pan          [deg, Moog coordinates]
```

This design keeps Level 1 geometry clean (pure Moog-coordinate mapping) so that pointing corrections v1–v4 each make a single uniform shift without double-counting.

#### Sun centroid (metadata only)

The sun centroid is detected from `Aquistion_0`'s `I_corrected` using a 0.95 × max threshold, but is **not** used as the geometry anchor. It is stored as metadata for reference and used by `compute_pointing_calibration_v1.py`:

```text
Measurement_Metadata/Sun Center Pixel aq0 (x,y)
```

#### Coordinate conventions

The following convention attributes are written to `Measurement_Metadata`:

```text
Geometry Reference Origin = "Image centre (IMG_X/2, IMG_Y/2)"

View Zenith Convention = "Sky zenith angle [deg]; 0=overhead; 90=horizon;
                          value at image centre pixel = 90 - Moog Tilt"

View Azimuth Convention = "Moog Pan coordinates [deg]; Pan=0 points due South;
                           North-based meteorological azimuth = (view_az + 180) % 360"
```

#### Per-frame attributes

Each acquisition/calibration group receives:

```text
Geometry Pan Used [deg]        = Moog Pan for this frame
Geometry Tilt Used [deg]       = Moog Tilt for this frame
Geometry Zenith Center [deg]   = 90 - Tilt  (view_zen at image centre)
Geometry Az Center [deg]       = Pan         (view_az  at image centre)
Geometry Pan Source            = attribute name used for Pan
Geometry Tilt Source           = attribute name used for Tilt
Geometry Sun Altitude Drift Correction = "disabled"
```

Tilt priority: `Moog Post Capture Actual Tilt [deg]` → `Moog Actual Tilt [deg]` → `Tilt`

Pan priority: `Moog Actual Pan [deg]` → `Pan`

#### Solar drift correction

`pixel_geometry()` receives `sza0 = Sun_Position_Altitude` of the current frame, so `delta_zen = 0` for every frame. No solar altitude drift correction is applied between acquisitions.

#### Outputs written per acquisition/calibration group

```text
view_az    [deg, Moog Pan coordinates]
view_zen   [deg, sky zenith angle]
sun_az     [deg]
sun_zen    [deg]
```

### Level 2: Polarization Products

Functions: `process_level2`, `save_level2_figure`

```python
q    = Q_corrected / I_corrected
u    = U_corrected / I_corrected
DoLP = sqrt(q**2 + u**2) * 100          [%]
AoLP = mod(degrees(0.5 * atan2(U, Q)), 180)  [deg]
```

**Outputs written per acquisition/calibration group:**

```text
q
u
DoLP
AoLP
```

Diagnostic figures are saved under `level2_figures/`.

**Metadata:**

```text
Measurement_Metadata/Processed Level = "Level 2"
```

---

## Pointing Corrections (v1–v4)

Level 1 geometry is anchored to the image centre and contains no boresight correction. Pointing corrections are applied as a post-processing step — either via the GUI (`robust_np_qt_app.py`) or by running the relevant script directly. Each correction writes per-frame attributes that the GUI reads to shift `view_zen` and `view_az` uniformly.

The GUI applies:

```python
view_zen_corrected = view_zen - zen_shift
view_az_corrected  = view_az  + (Pan_vN - Pan_raw)
```

### v1 — per-frame sun centroid (`compute_pointing_calibration_v1.py`)

Detects the sun centroid `(cx, cy)` per frame from `I_corrected` and computes:

```python
zen_shift = (sun_alt - tilt) + (cy - IMG_Y/2) * VFOV
pan_err   = (pan_raw - sun_az) + (cx - IMG_X/2) * HFOV
pan_v1    = pan_raw - pan_err
```

Works on both regular acquisitions and calibration frames. Requires Level 0 (`I_corrected`). Takes a single H5 file path:

```bash
python compute_pointing_calibration_v1.py /path/to/file.h5
python compute_pointing_calibration_v1.py /path/to/file.h5 --dry-run
```

Attributes written per frame:

```text
Tilt Error v1 Pred [deg]   — zenith correction (subtract from view_zen)
Zenith_v1 [deg]            — corrected zenith at image centre (= 90 - Tilt - zen_shift)
Az_v1 [deg]                — corrected azimuth at image centre (= Pan_v1)
Pan_v1 [deg]               — corrected pan; GUI uses (Pan_v1 - Pan_raw) as azimuth shift
Camera Roll v1 [deg]       = 0 (v1 does not estimate roll)
Sun Center v1 cx [px]
Sun Center v1 cy [px]
```

### v2 — folder-wide sinusoidal fit (`compute_pointing_calibration_v2.py`)

Fits a sinusoid `A·sin(pan + φ) + C` to tilt errors measured across all calibration frames in the folder. Also estimates camera roll. Takes a directory path.

Attributes written per frame:

```text
Tilt Error v2 Pred [deg]   — zenith correction (subtract from view_zen)
Zenith_v2 [deg]            — corrected zenith at image centre (= 90 - Tilt - zen_shift)
Az_v2 [deg]                — corrected azimuth at image centre (= Pan_v2)
Pan_v2 [deg]               — corrected pan; GUI uses (Pan_v2 - Pan_raw) as azimuth shift
Camera Roll v2 [deg]       — estimated camera roll
```

Azimuth is corrected via `Pan_v2`: the GUI applies `view_az += (Pan_v2 - Pan_raw)` to the entire grid. The pan correction is roll-induced and varies with pan angle: `Pan_v2 = Pan_raw + delta_roll · cos(pan) / cos(tilt)`.

### v2b — folder-wide fit from raw `Aquistion_0` (`compute_pointing_calibration_v2b.py`)

Uses the raw sun centre from `Aquistion_0` only to build a sinusoidal correction across the folder. With image-centre Level 1, v2b does not double-count. Takes a directory path. Attributes: `Tilt Error v2b Pred [deg]`, `Zenith_v2b [deg]`, `Az_v2b [deg]`, `Pan_v2b [deg]`, `Camera Roll v2b [deg]`.

### v3 — per-file calibration, shared roll (`compute_pointing_calibration_v2.py --only-file`)

Uses this file's calibration frames for the tilt sinusoid, but reuses the folder-wide pan/roll from v2. Takes a directory path with `--only-file`. Attributes: `Tilt Error v3 Pred [deg]`, `Zenith_v3 [deg]`, `Az_v3 [deg]`, `Pan_v3 [deg]`, `Camera Roll v3 [deg]`.

### v4 — raster calibration grid rotation (`apply_calibration_v4.py`)

Rotates the geometry grid to align with a raster calibration measurement. Takes a directory path.

### Why no double-counting

v2/v3 measure tilt error from `cy0 = IMG_Y/2` (image centre), independent of Level 1's reference choice. v1 measures the full per-frame boresight error directly from the sun centroid. Because Level 1 itself uses the image centre, there is no overlap: each vN correction accounts for exactly one source of pointing error.

---

## Robust Neutral-Point Processing (`robust_np_all_acquisitions.py`)

Run after Level 0/1/2:

```bash
python robust_np_all_acquisitions.py <h5_path>
```

Reads `I_corrected`, `Q_corrected`, `U_corrected`, `view_zen`, `view_az` (recomputing q/u from the Stokes datasets directly, not from the saved `q`/`u` datasets). Writes a robust NP CSV.

**Main output:** `<h5_stem>_robust_np_methods.csv`

**Optional diagnostic figures:** `robust_np_figures/`

### Cropping

A circular crop is applied before the NP search:

```text
radius_fraction  = 0.82
radius_shrink_px = 100.0
```

### Method 1: Robust Minimum DoLP (`robust_dolp_min`)

1. Smooth I, Q, U.
2. Compute DoLP from smoothed Stokes.
3. Mask low-intensity and invalid pixels.
4. Select the low-DoLP percentile region.
5. Compute a weighted centre of that region.

Output columns: `dolp_np_zen`, `dolp_np_az`, `dolp_min`, `dolp_confidence`.

### Method 2: Q/U Zero-Line Intersection (`robust_qu_zero_intersection`)

Smooths Q/I and U/I, finds zero-crossing regions, and estimates the best overlap point.

Output columns: `zero_np_zen`, `zero_np_az`, `zero_residual`, `zero_confidence`, `q_zero_in_fov`, `u_zero_in_fov`.

### Confidence and Selection

Both methods produce confidence scores. The CSV also includes:

```text
agreement_confidence
total_confidence
selection    = BEST | POSSIBLE | not selected
```

### Calibration Tilt Correction Columns

`robust_np_all_acquisitions.py` also reports a calibration tilt correction computed from calibration frames (using `detect_sun_center_raw` on raw images, independent of Level 1):

```text
calib_tilt_correction_deg
calib_tilt_applied_deg
calib_tilt_n_used
calib_tilt_n_total
calib_tilt_details_json
```

This is diagnostic information in the CSV. It is not the same as the GUI's v1–v4 pointing corrections, and it is not written back into Level 1 geometry.

---

## Data Flow Summary

```
UV Raw Images
  └─ Level 0 ──→ I_corrected, Q_corrected, U_corrected
       └─ Level 1 ──→ view_zen, view_az, sun_zen, sun_az
            │             (image-centre reference; pure Moog coordinates)
            └─ Level 2 ──→ q, u, DoLP, AoLP, level2_figures/
                 └─ robust_np ──→ <stem>_robust_np_methods.csv

                 [optional, separate step]
            └─ Pointing correction (v1–v4)
                 └─ writes per-frame vN attrs into H5
                      └─ GUI applies uniform shift to view_zen / view_az
```

### H5 Dataset Reference

| Dataset | Written by | Content |
|---------|-----------|---------|
| `I_corrected` | Level 0 | Stokes I [counts] |
| `Q_corrected` | Level 0 | Stokes Q [counts] |
| `U_corrected` | Level 0 | Stokes U [counts] |
| `view_az` | Level 1 | Moog azimuth per pixel [deg]; 0 = South |
| `view_zen` | Level 1 | Sky zenith per pixel [deg]; 0 = overhead |
| `sun_az` | Level 1 | Sun azimuth [deg] |
| `sun_zen` | Level 1 | Sun zenith [deg] |
| `q` | Level 2 | Q/I |
| `u` | Level 2 | U/I |
| `DoLP` | Level 2 | Degree of linear polarisation [%] |
| `AoLP` | Level 2 | Angle of linear polarisation [deg] |

---

## Practical Notes

- The H5 spelling `Aquistion_*` is intentional; it matches the existing data files.
- The batch script modifies H5 files in place. Existing datasets are overwritten.
- `--mode with-csv` does not mean "already processed correctly"; it means a sibling robust CSV exists.
- `--mode without-csv` is useful for filling in missing CSV files without reprocessing everything.
- Level 1 `view_az` is in Moog Pan coordinates (0 = South). To convert to standard meteorological azimuth (North = 0): `az_met = (view_az + 180) % 360`.
- Pointing corrections v1–v4 are separate from the batch pipeline. Run them via the GUI or their scripts after Level 0/1/2 have been written.
- `robust_np_all_acquisitions.py` recomputes q/u from `I/Q/U_corrected` at runtime; the saved `q`/`u` datasets from Level 2 are not used.

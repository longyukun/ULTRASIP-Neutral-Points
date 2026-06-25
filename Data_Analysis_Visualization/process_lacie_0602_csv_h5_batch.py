import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_H5_DIR = Path("/Volumes/LaCie/Level2 data/2026_06_02")
DEFAULT_NUC = SCRIPT_DIR / "NUC_0813.npz"
DEFAULT_WMATRIX = SCRIPT_DIR / "ULTRASIP_AvgWmatrix_15.npy"

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs, flush=True)


def robust_csv_path(h5_path):
    return h5_path.with_name(f"{h5_path.stem}_robust_np_methods.csv")


def is_real_h5(path):
    return (
        path.is_file()
        and path.suffix.lower() in (".h5", ".hdf5")
        and not path.name.startswith("._")
    )


def selected_h5_files(directory, mode):
    files = sorted(path for path in directory.iterdir() if is_real_h5(path))
    if mode == "all":
        return files
    if mode == "with-csv":
        return [path for path in files if robust_csv_path(path).exists()]
    if mode == "without-csv":
        return [path for path in files if not robust_csv_path(path).exists()]
    raise ValueError(f"Unsupported mode: {mode}")


def run(command, env, label=""):
    tprint(f"{label}Running: {' '.join(map(str, command))}")
    result = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            tprint(f"{label}{line}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def process_one(index, total, path, nuc, wmatrix, env):
    label = f"[{index}/{total}] {path.name} | "
    tprint(f"\n{label}START")
    try:
        run([
            sys.executable,
            str(SCRIPT_DIR / "process_single_level0_1_2.py"),
            str(path),
            str(nuc),
            str(wmatrix),
        ], env, label=label)
        run([
            sys.executable,
            str(SCRIPT_DIR / "robust_np_all_acquisitions.py"),
            str(path),
        ], env, label=label)
        tprint(f"{label}DONE")
        return None
    except subprocess.CalledProcessError as exc:
        tprint(f"{label}FAILED: exit {exc.returncode}")
        return (index, path.name, exc.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch reprocess ULTRASIP H5 files through Level0-2 and robust NP. "
            "Files can be selected by whether a sibling *_robust_np_methods.csv exists."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_H5_DIR,
        help=f"Directory containing H5 files. Default: {DEFAULT_H5_DIR}",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "with-csv", "without-csv"),
        default="with-csv",
        help="Which H5 files to process. Default: with-csv.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="1-based index in the selected file list to resume from. Default: 1.",
    )
    parser.add_argument(
        "--nuc",
        type=Path,
        default=DEFAULT_NUC,
        help=f"NUC calibration file. Default: {DEFAULT_NUC}",
    )
    parser.add_argument(
        "--wmatrix",
        type=Path,
        default=DEFAULT_WMATRIX,
        help=f"W matrix file. Default: {DEFAULT_WMATRIX}",
    )
    parser.add_argument(
        "--workers",
        default="4",
        help="NP_WORKERS value passed to robust NP processing. Default: 4.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of H5 files to process in parallel. Default: 4.",
    )
    parser.add_argument(
        "--save-diagnostics",
        choices=("0", "1"),
        default="0",
        help="Set NP_SAVE_DIAGNOSTICS. Default: 0.",
    )
    parser.add_argument(
        "--min-sun-zen-separation",
        default="1.0",
        help="NP_MIN_SUN_ZEN_SEPARATION_DEG value. Default: 1.0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    directory = args.directory.expanduser()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"H5 directory does not exist or is not a directory: {directory}")

    files = selected_h5_files(directory, args.mode)
    if not files:
        raise SystemExit(f"No H5 files matched mode={args.mode} in {directory}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/ultrasip_matplotlib")
    env["NP_SAVE_DIAGNOSTICS"] = str(args.save_diagnostics)
    env["NP_WORKERS"] = str(args.workers)
    env["NP_MIN_SUN_ZEN_SEPARATION_DEG"] = str(args.min_sun_zen_separation)

    print(f"Directory: {directory}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Parallel: {args.parallel} files at a time", flush=True)
    print(f"Selected {len(files)} H5 file(s):", flush=True)
    for index, path in enumerate(files, start=1):
        csv_state = "has CSV" if robust_csv_path(path).exists() else "no CSV"
        marker = " <-- start" if index == args.start_at else ""
        print(f"  [{index}] {path.name} ({csv_state}){marker}", flush=True)

    work = [
        (index, path)
        for index, path in enumerate(files, start=1)
        if index >= args.start_at
    ]

    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(process_one, index, len(files), path, args.nuc, args.wmatrix, env): (index, path.name)
            for index, path in work
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                failures.append(result)

    if failures:
        print("\nFailures:", flush=True)
        for index, name, code in sorted(failures):
            print(f"  [{index}] {name}: exit {code}", flush=True)
        raise SystemExit(1)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

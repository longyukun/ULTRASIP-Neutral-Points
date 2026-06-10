import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CRUCIAL_DIR = Path("/Volumes/Crucial")
NUC = ROOT / "ULTRASIP-Neutral-Points" / "Data_Analysis_Visualization" / "NUC_0813.npz"
WMATRIX = ROOT / "ULTRASIP-Neutral-Points" / "Data_Analysis_Visualization" / "ULTRASIP_AvgWmatrix_15.npy"


def h5_files():
    return sorted(
        path for path in CRUCIAL_DIR.glob("*.h5")
        if path.is_file() and not path.name.startswith("._")
    )


def run(command, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    files = h5_files()
    if not files:
        raise SystemExit(f"No H5 files found in {CRUCIAL_DIR}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/ultrasip_matplotlib")
    env.setdefault("NP_SAVE_DIAGNOSTICS", "0")
    env.setdefault("NP_WORKERS", "4")
    env.setdefault("NP_MIN_SUN_ZEN_SEPARATION_DEG", "1.0")

    print("Files:")
    for path in files:
        print(f"  {path}", flush=True)

    for index, path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {path.name}", flush=True)
        run([
            sys.executable,
            str(SCRIPT_DIR / "process_single_level0_1_2.py"),
            str(path),
            str(NUC),
            str(WMATRIX),
        ], env)
        run([
            sys.executable,
            str(SCRIPT_DIR / "robust_np_all_acquisitions.py"),
            str(path),
        ], env)


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
RUN_DIR = ROOT / "runs" / "attention_unet_fair20_fixed_postprocess_seed42"
PYTHON = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "-m",
        "attention_unet.train",
        "--data-dir",
        str(REPOSITORY_ROOT / "Dataset"),
        "--epochs",
        "20",
        "--batch-size",
        "4",
        "--lr",
        "0.001",
        "--base-channels",
        "16",
        "--image-size",
        "256,256",
        "--augment",
        "basic",
        "--loss",
        "combo",
        "--loss-dice-weight",
        "1.0",
        "--loss-bce-weight",
        "1.0",
        "--loss-focal-weight",
        "0.5",
        "--scheduler",
        "plateau",
        "--threshold-min",
        "0.12",
        "--threshold-max",
        "0.50",
        "--threshold-steps",
        "20",
        "--early-stopping-patience",
        "8",
        "--postprocess",
        "--post-open-iters",
        "1",
        "--output-dir",
        str(RUN_DIR),
    ]
    out_path = RUN_DIR / "train.out.log"
    err_path = RUN_DIR / "train.err.log"
    with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_BREAKAWAY_FROM_JOB
            ),
        )
    (RUN_DIR / "train.pid").write_text(str(process.pid), encoding="utf-8")
    print(f"Started Attention U-Net fair training PID={process.pid}")
    print(f"stdout: {out_path}")
    print(f"stderr: {err_path}")


if __name__ == "__main__":
    main()

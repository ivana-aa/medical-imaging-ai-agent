import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

import train_unet as tu


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "Dataset"
BASE_CHECKPOINT = ROOT / "runs" / "unet_agent_combo_focal_5ep_wide_thr" / "best_unet.pt"
OUTPUT_ROOT = ROOT / "runs" / "unet_agent_seed_stability"
SEEDS = [11, 22, 33, 44, 55]
TARGET_EPOCH = 20


def evaluate_checkpoint(checkpoint_path: Path, output_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**checkpoint.get("args", {}))
    image_size = getattr(args, "image_size", None)
    if image_size is not None:
        image_size = (int(image_size[0]), int(image_size[1]))
    base_channels = int(getattr(args, "base_channels", 16))
    threshold = float(checkpoint.get("best_threshold", 0.5))
    postprocess = tu.postprocess_from_dict(checkpoint.get("best_postprocess"))

    model = tu.UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    criterion = tu.build_criterion(args)

    test_loader = tu.make_loader(
        DATA_DIR,
        "test",
        image_size,
        batch_size=4,
        num_workers=0,
        augment="off",
        shuffle=False,
    )
    metrics = tu.evaluate(
        model,
        test_loader,
        criterion,
        device,
        split_name=f"test-{output_dir.name}",
        threshold=threshold,
        postprocess=postprocess,
        tta="off",
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "val_best_dice": checkpoint.get("best_dice"),
        "threshold": threshold,
        "postprocess": checkpoint.get("best_postprocess"),
        "test_loss": metrics.loss,
        "test_dice": metrics.dice,
        "test_iou": metrics.iou,
        "test_samples": tu.loader_sample_count(test_loader),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (output_dir / "test_eval.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def write_summary(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {"runs": rows}
    if rows:
        summary.update(
            {
                "mean_val_best_dice": statistics.mean(row["val_best_dice"] for row in rows),
                "std_val_best_dice": statistics.pstdev(row["val_best_dice"] for row in rows),
                "mean_test_dice": statistics.mean(row["test_dice"] for row in rows),
                "std_test_dice": statistics.pstdev(row["test_dice"] for row in rows),
                "mean_test_iou": statistics.mean(row["test_iou"] for row in rows),
                "std_test_iou": statistics.pstdev(row["test_iou"] for row in rows),
            }
        )
    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def read_existing_rows() -> list[dict]:
    csv_path = OUTPUT_ROOT / "summary.csv"
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in (
                "seed",
                "returncode",
                "checkpoint_epoch",
            ):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = int(parsed[key])
            for key in (
                "val_best_dice",
                "threshold",
                "test_loss",
                "test_dice",
                "test_iou",
            ):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_existing_rows()
    completed_seeds = {int(row["seed"]) for row in rows if row.get("status") in ("ok", "no_improvement")}
    status_path = OUTPUT_ROOT / "status.txt"

    for index, seed in enumerate(SEEDS, start=1):
        if seed in completed_seeds:
            continue
        run_dir = OUTPUT_ROOT / f"seed_{seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "history.csv").exists():
            checkpoint_for_eval = run_dir / "best_unet.pt"
            status = "ok"
            if not checkpoint_for_eval.exists():
                checkpoint_for_eval = BASE_CHECKPOINT
                status = "no_improvement"
            test_result = evaluate_checkpoint(checkpoint_for_eval, run_dir)
            rows.append(
                {
                    "seed": seed,
                    "status": status,
                    "returncode": 0,
                    "run_dir": str(run_dir),
                    "checkpoint_epoch": test_result["checkpoint_epoch"],
                    "val_best_dice": test_result["val_best_dice"],
                    "threshold": test_result["threshold"],
                    "test_loss": test_result["test_loss"],
                    "test_dice": test_result["test_dice"],
                    "test_iou": test_result["test_iou"],
                }
            )
            write_summary(rows)
            completed_seeds.add(seed)
            continue
        with status_path.open("w", encoding="utf-8") as f:
            f.write(f"running {index}/{len(SEEDS)} seed={seed} started={datetime.now().isoformat(timespec='seconds')}\n")

        command = [
            sys.executable,
            "-u",
            str(ROOT / "train_unet.py"),
            "--data-dir",
            str(DATA_DIR),
            "--checkpoint",
            str(BASE_CHECKPOINT),
            "--reset-optimizer",
            "--epochs",
            str(TARGET_EPOCH),
            "--output-dir",
            str(run_dir),
            "--batch-size",
            "4",
            "--lr",
            "0.00005",
            "--loss",
            "combo",
            "--loss-dice-weight",
            "1",
            "--loss-bce-weight",
            "1",
            "--loss-focal-weight",
            "0.5",
            "--scheduler",
            "plateau",
            "--scheduler-patience",
            "2",
            "--scheduler-factor",
            "0.5",
            "--early-stopping-patience",
            "2",
            "--early-stopping-min-delta",
            "0.0002",
            "--threshold-min",
            "0.12",
            "--threshold-max",
            "0.50",
            "--threshold-steps",
            "20",
            "--postprocess",
            "--post-open-iters",
            "1",
            "--augment",
            "basic",
            "--tta",
            "off",
            "--no-visuals",
            "--seed",
            str(seed),
        ]
        with (run_dir / "train.out.log").open("w", encoding="utf-8") as stdout, (
            run_dir / "train.err.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=str(ROOT), stdout=stdout, stderr=stderr)

        if completed.returncode != 0:
            row = {
                "seed": seed,
                "status": "failed",
                "returncode": completed.returncode,
                "run_dir": str(run_dir),
                "val_best_dice": 0.0,
                "test_dice": 0.0,
                "test_iou": 0.0,
            }
            rows.append(row)
            write_summary(rows)
            continue

        checkpoint_for_eval = run_dir / "best_unet.pt"
        status = "ok"
        if not checkpoint_for_eval.exists():
            checkpoint_for_eval = BASE_CHECKPOINT
            status = "no_improvement"
        test_result = evaluate_checkpoint(checkpoint_for_eval, run_dir)
        row = {
            "seed": seed,
            "status": status,
            "returncode": completed.returncode,
            "run_dir": str(run_dir),
            "checkpoint_epoch": test_result["checkpoint_epoch"],
            "val_best_dice": test_result["val_best_dice"],
            "threshold": test_result["threshold"],
            "test_loss": test_result["test_loss"],
            "test_dice": test_result["test_dice"],
            "test_iou": test_result["test_iou"],
        }
        rows.append(row)
        write_summary(rows)

    with status_path.open("w", encoding="utf-8") as f:
        f.write(f"done {len(rows)}/{len(SEEDS)} finished={datetime.now().isoformat(timespec='seconds')}\n")
    write_summary(rows)


if __name__ == "__main__":
    main()

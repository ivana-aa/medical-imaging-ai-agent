"""Training entry point for the U-Net segmentation model.

This file keeps model development separate from the web backend. The heavy
lifting lives in train_unet.py; this wrapper gives the project a clear
"train.py" command for resumes, reports, and interviews.
"""

from __future__ import annotations

from typing import Optional

from train_unet import DEFAULT_RUNS_DIR, build_arg_parser, run


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    parser.description = "Train the binary U-Net medical image segmentation model."

    # Defaults aligned with the current lightweight deployed model family.
    parser.set_defaults(
        base_channels=16,
        image_size=(256, 256),
        output_dir=DEFAULT_RUNS_DIR / "unet_experiment",
        loss="combo",
        loss_dice_weight=1.0,
        loss_bce_weight=1.0,
        loss_focal_weight=0.5,
        augment="basic",
        scheduler="plateau",
        threshold_min=0.20,
        threshold_max=0.60,
        threshold_steps=17,
        early_stopping_patience=8,
        postprocess_search=True,
    )

    args = parser.parse_args(argv)
    if args.eval_only:
        parser.error("Use evaluate.py for evaluation-only runs.")
    run(args)


if __name__ == "__main__":
    main()

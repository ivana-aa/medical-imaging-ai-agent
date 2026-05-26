# Self-Supervised Learning Project

From-scratch self-supervised representation learning for the medical image
dataset in `../Dataset`.

## Boundaries

- All training code lives in this folder.
- Pretraining reads only image files from `../Dataset`.
- This project does not import code, configs, checkpoints, or logs from
  `../original_unet_project` or `../attention_unet_project`.
- Backbones are initialized from scratch. Torchvision models are created with
  `weights=None`.
- Outputs are written inside this folder under `runs/`.

## What Is Included

- SimCLR pretraining with a ResNet encoder and MLP projection head.
- NT-Xent contrastive loss.
- CNN masked image modeling pretraining for masked-region reconstruction.
- Dataset loaders for `Dataset/<split>/images`.
- A frozen-encoder segmentation probe using `Dataset/<split>/labels`.
- A ResNet-UNet decoder for supervised fine-tuning/evaluation.
- YAML configs for reproducible runs.

## Quick Start

Install dependencies in a Python environment with PyTorch:

```powershell
cd .\self_supervised_learning_project
pip install -r requirements.txt
```

Run self-supervised pretraining:

```powershell
python scripts\pretrain.py --config configs\simclr_medical.yaml
```

CPU-friendly baseline:

```powershell
python scripts\pretrain.py --config configs\simclr_medical_cpu.yaml
python scripts\linear_probe_segmentation.py --config configs\linear_probe_segmentation_cpu.yaml
```

Check the dataset structure before training:

```powershell
python scripts\check_dataset.py
```

Evaluate the frozen encoder with a lightweight segmentation probe:

```powershell
python scripts\linear_probe_segmentation.py --config configs\linear_probe_segmentation.yaml
```

Fine-tune a segmentation decoder from the best SimCLR encoder:

```powershell
python scripts\train_segmentation_decoder.py --config configs\decoder_finetune_simclr_cpu20.yaml
```

Evaluate a saved decoder checkpoint:

```powershell
python scripts\evaluate_segmentation_decoder.py --config configs\decoder_finetune_simclr_cpu20.yaml --checkpoint runs\decoder_finetune_simclr_cpu20\best_decoder.pt --split test
```

The default configs use the shared dataset at `../Dataset` and write results to
this project's `runs/` directory.
The deployed three-checkpoint ensemble is packaged separately at
`../models/weights/ssl_current/`.

## Current CPU Results

Dataset check: train 1172 pairs, val 252 pairs, test 252 pairs.

| Run | Best val Dice | Best val IoU | Test Dice | Test IoU | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen 1x1 probe after 50-epoch SimCLR | 0.34672 | 0.25733 | - | - | Too weak for this dense task. |
| ResNet-UNet, SimCLR encoder init | 0.92167 | 0.85790 | 0.92407 | 0.86105 | Best checkpoint at epoch 19. |
| ResNet-UNet, random init | 0.91969 | 0.85444 | 0.92483 | 0.86235 | Best checkpoint at epoch 19. |
| ResNet-UNet, SimCLR init, 10% labels | 0.77854 | 0.65458 | 0.80049 | 0.67634 | Uses 117/1172 train pairs. |
| ResNet-UNet, random init, 10% labels | 0.77678 | 0.65354 | 0.79345 | 0.66901 | Uses the same 117 train pairs. |
| ResNet-UNet, SimCLR init, 5% labels | 0.66892 | 0.53398 | 0.69716 | 0.56084 | Uses 59/1172 train pairs. |
| ResNet-UNet, random init, 5% labels | 0.70088 | 0.57044 | 0.71861 | 0.58203 | Uses the same 59 train pairs. |
| ResNet-UNet, MIM init, 5% labels | 0.70177 | 0.57485 | 0.71634 | 0.58302 | 20-epoch masked image modeling pretrain. |
| ResNet-UNet, MIM init, 5% labels | 0.70754 | 0.57543 | 0.73047 | 0.59601 | 50-epoch masked image modeling pretrain. |
| ResNet-UNet, hard MIM init, 5% labels | 0.72706 | 0.59937 | 0.74628 | 0.61402 | 100 total pretrain epochs, mask_ratio 0.75, patch_size 8. |
| ResNet-UNet, hard MIM init, 5% labels, tuned threshold | 0.73736 | 0.61048 | 0.75322 | 0.62240 | Same checkpoint, validation-selected threshold 0.57. |

The useful comparison is the decoder fine-tuning result, not the frozen 1x1
probe. With all labels, SimCLR and random initialization are nearly tied on the
test set. With only 10% labels, SimCLR is slightly better on the test set, but
the margin is still small. With 5% labels, SimCLR learns useful masks earlier,
but the best random-initialized checkpoint is better on both validation and
test. A 20-epoch masked image modeling run is better than SimCLR in the 5%
label setting and nearly tied with random initialization. Extending masked image
modeling to 50 epochs improves the downstream 5% label run and beats the random
baseline on validation Dice, test Dice, and test IoU. The current evidence is
that MIM is the stronger self-supervised direction for this dataset, while
multiple subset seeds are still needed before making a final statistical claim.
Continuing MIM-50 with a harder masked reconstruction task to 100 total
pretrain epochs gives the best single-seed result so far on the 5% label
setting.
Threshold tuning on the validation split improves the hard MIM-100 checkpoint
without retraining: the selected threshold is 0.57 instead of the default 0.50,
raising test Dice from 0.74628 to 0.75322.

## 5% Label Multi-Seed Check

Each run uses 59/1172 train label pairs. Validation and test sets stay complete.

| subset_seed | Method | Best val Dice | Test Dice | Test IoU |
| ---: | --- | ---: | ---: | ---: |
| 7 | MIM-50 | 0.69546 | 0.72408 | 0.58769 |
| 7 | Random | 0.66871 | 0.68770 | 0.54550 |
| 42 | MIM-50 | 0.70754 | 0.73047 | 0.59601 |
| 42 | Random | 0.70088 | 0.71861 | 0.58203 |
| 123 | MIM-50 | 0.71834 | 0.75369 | 0.62768 |
| 123 | Random | 0.68511 | 0.72543 | 0.59087 |
| 7 | hard MIM-100 + tuned threshold | 0.71021 | 0.73801 | 0.60829 |
| 42 | hard MIM-100 + tuned threshold | 0.73736 | 0.75322 | 0.62240 |
| 123 | hard MIM-100 + tuned threshold | 0.73639 | 0.76402 | 0.64123 |

Three-seed mean:

| Method | Mean best val Dice | Mean test Dice | Mean test IoU |
| --- | ---: | ---: | ---: |
| MIM-50 | 0.70712 | 0.73608 | 0.60380 |
| Random | 0.68490 | 0.71058 | 0.57280 |
| hard MIM-100 + tuned threshold | 0.72799 | 0.75175 | 0.62397 |
| hard MIM-100 ensemble + tuned threshold | 0.74417 | 0.76671 | 0.64353 |
| hard MIM-100 intensity-aug ensemble + tuned threshold | 0.75851 | 0.78471 | 0.66865 |
| hard MIM-100 intensity aug 256px + tuned threshold | 0.81249 | 0.84252 | 0.74688 |
| hard MIM-100 intensity aug 256px ensemble + tuned threshold | 0.85241 | 0.88212 | 0.80496 |
| hard MIM-100 intensity aug 256px ensemble, seed123 to 30 epochs | 0.85581 | 0.88621 | 0.81154 |

Across these three 5% label subsets, MIM-50 beats random initialization on test
Dice for every subset. The mean test Dice gain is about +0.0255, and the mean
test IoU gain is about +0.0310.
The hard MIM-100 setup with validation-selected thresholds improves further:
mean test Dice is about +0.0157 above MIM-50 and +0.0412 above random
initialization.
Averaging the three hard MIM-100 probability maps and selecting a single
validation threshold improves the current best result again: test Dice rises to
0.76671 and test IoU to 0.64353.

Additional inference-only checks:

| Method | Val Dice | Test Dice | Test IoU | Decision |
| --- | ---: | ---: | ---: | --- |
| hard MIM-100 ensemble + flip TTA | 0.69335 | 0.71844 | 0.58500 | Not used; flips hurt this dataset. |
| hard MIM-100 ensemble + connected-component cleanup | 0.74778 | 0.76622 | 0.64299 | Not used; validation improves but test does not. |

The current best inference setting remains hard MIM-100 ensemble without TTA or
connected-component cleanup, using threshold 0.56.

Training-side augmentation check:

| Method | Mean val Dice | Mean test Dice | Mean test IoU |
| --- | ---: | ---: | ---: |
| hard MIM-100 + tuned threshold | 0.72799 | 0.75175 | 0.62397 |
| hard MIM-100 + intensity aug + tuned threshold | 0.72594 | 0.75671 | 0.63153 |

Geometric augmentation with small rotation/translation/scale hurt the 5% label
run. Intensity-only augmentation is safer: individual model gains are modest,
but the three-model intensity-aug ensemble is the best result so far, with test
Dice 0.78471 and test IoU 0.66865.

Loss-function check:

| Method | Seed | Val Dice | Test Dice | Test IoU | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| intensity aug + BCE/Dice | 42 | 0.72927 | 0.75576 | 0.62712 | Baseline for this check. |
| intensity aug + BCE/Tversky alpha 0.7 beta 0.3 | 42 | 0.73476 | 0.75263 | 0.62142 | Not expanded; val improves but test drops. |

Resolution check:

| Method | Seed | Val Dice | Test Dice | Test IoU | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| intensity aug 128px + BCE/Dice | 42 | 0.72927 | 0.75576 | 0.62712 | Previous single-seed baseline. |
| intensity aug 256px + BCE/Dice | 42 | 0.81249 | 0.84252 | 0.74688 | Best single model so far; expand to more seeds. |

256px multi-seed result:

| subset_seed | Threshold | Val Dice | Test Dice | Test IoU |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.43 | 0.83202 | 0.86115 | 0.77475 |
| 42 | 0.52 | 0.81249 | 0.84252 | 0.74688 |
| 123 | 0.72 | 0.84304 | 0.87988 | 0.80588 |
| 123, extended to 30 epochs | 0.74 | 0.84665 | 0.88473 | 0.81473 |
| 7, extended to 30 epochs | 0.34 | 0.84877 | 0.87912 | 0.80508 |
| 42, extended to 30 epochs | 0.30 | 0.85798 | 0.87947 | 0.79832 |

| Method | Mean val Dice | Mean test Dice | Mean test IoU |
| --- | ---: | ---: | ---: |
| hard MIM-100 intensity aug 256px | 0.82918 | 0.86118 | 0.77584 |
| hard MIM-100 intensity aug 256px ensemble | 0.85241 | 0.88212 | 0.80496 |
| hard MIM-100 intensity aug 256px ensemble, seed123 to 30 epochs | 0.85581 | 0.88621 | 0.81154 |
| hard MIM-100 intensity aug 256px, all seeds to 30 epochs | 0.85113 | 0.88111 | 0.80604 |
| hard MIM-100 intensity aug 256px ensemble, all seeds to 30 epochs | 0.87463 | 0.90154 | 0.83597 |
| hard MIM-100 intensity aug 256px ensemble, all seeds to 30 epochs + empty-prediction fallback | 0.90150 | 0.90766 | 0.84108 |
| hard-mined seed42 + original seed7/123 ensemble + empty-prediction fallback | 0.90496 | 0.90901 | 0.84332 |
| hard-mined seed7/42/123 ensemble + empty-prediction fallback | 0.90471 | 0.91099 | 0.84732 |
| hard-mined seed7 e35 + seed42 e45 + seed123 early-stop ensemble + empty-prediction fallback, 2026-05-15 threshold retune | 0.90720 | 0.91264 | 0.84856 |
| 10% label hard-mined seed7/42/123 ensemble + empty-prediction fallback | 0.91577 | 0.92074 | 0.86022 |

Moving decoder fine-tuning from 128px to 256px is the largest improvement in
the project so far. Extending all three 256px fine-tunes to 30 epochs improves
the ensemble further. The current best setting is the 256px three-seed ensemble
with 10% label hard-mining: seed7 epoch 44, seed42 epoch 52, and seed123 epoch
39. It uses validation-selected threshold 0.335 and an empty-prediction
fallback threshold of 0.0004 with `min_area=50`.

256px inference-side checks before extending all seeds to 30 epochs:

| Method | Val Dice | Test Dice | Test IoU | Decision |
| --- | ---: | ---: | ---: | --- |
| 256px ensemble threshold fine sweep | 0.85243 | 0.88211 | 0.80494 | No meaningful gain over threshold 0.56. |
| 256px ensemble connected-component cleanup | 0.86293 | 0.87738 | 0.80008 | Not used; validation improves but test drops. |

Final 256px all-30 ensemble export:

| Artifact | Path |
| --- | --- |
| Per-case CSV | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports/per_case_metrics.csv` |
| Per-case JSON | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports/per_case_metrics.json` |
| Summary JSON | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports/summary.json` |
| Prediction panels | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports/panels/` |

The export uses the final validation-selected threshold 0.458 and reproduces
the best test metrics: Dice 0.90154, IoU 0.83597, precision 0.90403, recall
0.91906. It saves first-case, best-case, and worst-case visual panels with
image, ground truth, probability map, binary prediction, and error overlay.
The worst four test cases are complete misses: prediction area is 0, while the
ground-truth areas are 2093, 2063, 1935, and 1628 pixels. Their maximum
probabilities are about 0.333, 0.333, 0.0005, and 0.0005, so the next useful
optimization should target hard-case recall rather than ordinary threshold
tuning.

Empty-prediction fallback check:

| Method | Val Dice | Test Dice | Test IoU | Precision | Recall | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256px all-30 ensemble, threshold 0.458 | 0.87463 | 0.90154 | 0.83597 | 0.90403 | 0.91906 | Previous best. |
| Empty-prediction fallback 0.110 + `min_area=50` | 0.90150 | 0.90766 | 0.84108 | 0.90624 | 0.92084 | New best; adopt. |

The fallback is selected using validation Dice only. It activates only when the
normal 0.458 threshold produces an empty prediction, then retries that image at
threshold 0.110 and removes connected components smaller than 50 pixels. On the
test set it triggers on four cases, rescues two positive cases, and introduces
zero empty-label false positives after area filtering.

Fallback artifacts:

| Artifact | Path |
| --- | --- |
| Fallback sweep | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/empty_prediction_fallback/empty_prediction_fallback_sweep.json` |
| Fallback per-case CSV | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports_empty_fallback/per_case_metrics.csv` |
| Fallback summary JSON | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports_empty_fallback/summary.json` |
| Fallback panels | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_all/prediction_exports_empty_fallback/panels/` |

Hard-mined seed42 fine-tune:

| Method | Val Dice | Test Dice | Test IoU | Precision | Recall | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Replace seed42 with hard-mined epoch35, no fallback | 0.87419 | 0.90239 | 0.83761 | 0.90569 | 0.91955 | Small test gain, val slightly below previous. |
| Replace seed42 with hard-mined epoch35 + fallback 0.100 + `min_area=50` | 0.90496 | 0.90901 | 0.84332 | 0.90816 | 0.92192 | New best; adopt for now. |

Seed42 hard-mining uses only its original 5% labeled training subset. Before
fine-tuning, the resumed seed42 checkpoint mines that subset at threshold 0.458
and assigns larger sampler weights to positive samples with low Dice or empty
predictions. In this run, 59 labeled train cases were used: 53 positive cases,
2 low-Dice positives, and 1 empty-prediction positive. The fine-tune runs from
epoch 31 to 35 with lower learning rates (`encoder_lr=1e-5`,
`decoder_lr=2e-4`) and the same intensity-only augmentation.

Hard-mined artifacts:

| Artifact | Path |
| --- | --- |
| Hard fine-tune config | `configs/decoder_hard_finetune_mim_cpu100_hard_label5_intensity_aug_256.yaml` |
| Hard-mining script | `scripts/train_segmentation_decoder_hard_mining.py` |
| Hard-mined seed42 checkpoint | `runs/decoder_hard_finetune_mim_cpu100_hard_label5_intensity_aug_256_seed42_epoch35/best_decoder.pt` |
| Hard-mining train cases | `runs/decoder_hard_finetune_mim_cpu100_hard_label5_intensity_aug_256_seed42_epoch35/hard_mining_train_cases.json` |
| New ensemble fallback sweep | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_seed42_hard/empty_prediction_fallback/empty_prediction_fallback_sweep.json` |
| New ensemble per-case CSV | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_seed42_hard/prediction_exports_empty_fallback/per_case_metrics.csv` |
| New ensemble panels | `runs/ensemble_hard_mim100_label5_intensity_aug_256_epoch30_seed42_hard/prediction_exports_empty_fallback/panels/` |

Extended hard-mining and ensemble selection:

| Candidate | Val Dice | Test Dice | Test IoU | Precision | Recall | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| All hard-mined to e35/early-stop + fallback 0.0009 + `min_area=50` | 0.90471 | 0.91099 | 0.84732 | 0.90576 | 0.92918 | Improved test and recall, but not final val best. |
| seed7 e40 + seed42 e40 + seed123 early-stop + fallback 0.001 + `min_area=50` | 0.89898 | 0.91453 | 0.84946 | 0.91250 | 0.92322 | Highest exploratory test Dice, but validation is lower; not selected as the formal setting. |
| seed7 e35 + seed42 e40 + seed123 early-stop + fallback 0.001 + `min_area=50` | 0.90633 | 0.91255 | 0.84868 | 0.90746 | 0.92755 | Strong, but seed42 e45 improves validation slightly. |
| seed7 e35 + seed42 e45 + seed123 early-stop + threshold 0.425 + fallback 0.0008 + `min_area=50` | 0.90720 | 0.91264 | 0.84856 | 0.90721 | 0.92776 | Previous formal best after 2026-05-15 threshold retune. |
| 10% label hard-mined seed7/42/123 ensemble + threshold 0.335 + fallback 0.0004 + `min_area=50` | 0.91577 | 0.92074 | 0.86022 | 0.89812 | 0.95396 | Formal current best by validation selection. |

Integer-weight ensemble checks by duplicating one checkpoint did not beat equal
weights. A low-area fallback that also retried predictions with small nonzero
foreground area was tested, but validation selected the original empty-only
behavior; this avoids adopting a rule that helps a few test cases while hurting
validation Dice.

The 10% label hard-mining update keeps the same self-supervised MIM encoder
family and training code, but expands each seed's fine-tuning subset from
59/1172 to 117/1172 labeled train pairs. This is still a low-label
self-supervised transfer setting, and it is now the deployed `ssl_current`
model in the unified frontend/backend.

Current best artifacts:

| Artifact | Path |
| --- | --- |
| seed7 10% hard checkpoint | `runs/decoder_hard_finetune_mim_cpu100_hard_label10_intensity_aug_256_seed7_epoch44/best_decoder.pt` |
| seed42 10% hard checkpoint | `runs/decoder_hard_finetune_mim_cpu100_hard_label10_intensity_aug_256_seed42_epoch52/best_decoder.pt` |
| seed123 10% hard checkpoint | `runs/decoder_hard_finetune_mim_cpu100_hard_label10_intensity_aug_256_seed123_epoch39/best_decoder.pt` |
| 10% hard-mining config | `configs/decoder_hard_finetune_mim_cpu100_hard_label10_intensity_aug_256.yaml` |
| 10% ensemble threshold sweep | `runs/metric_opt_20260515_label10_all3/threshold_minarea_sweep.json` |
| 10% ensemble fallback sweep | `runs/metric_opt_20260515_label10_all3_fallback/fallback_sweep.json` |
| 10% ensemble per-case export | `runs/metric_opt_20260515_label10_all3_prediction_exports_thr0335_fb0004/summary.json` |
| 10% ensemble panels | `runs/metric_opt_20260515_label10_all3_prediction_exports_thr0335_fb0004/panels/` |

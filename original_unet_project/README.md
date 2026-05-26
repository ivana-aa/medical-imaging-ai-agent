# Original U-Net Project

This folder preserves the first U-Net segmentation implementation and its
training and evaluation code.

## Boundaries

- Source code and new run outputs live inside this folder.
- The shared dataset remains outside the project at `../Dataset`.
- This project is independent from `../attention_unet_project`.
- The self-supervised project does not import training code from this folder.
- The released application checkpoint is stored at
  `../models/weights/original_unet/best_unet.pt`.

## Common Commands

```powershell
cd .\original_unet_project
..\.venv\Scripts\python.exe train.py
..\.venv\Scripts\python.exe evaluate.py --checkpoint ..\models\weights\original_unet\best_unet.pt
..\.venv\Scripts\python.exe predict.py --checkpoint ..\models\weights\original_unet\best_unet.pt
```

Published releases omit experiment outputs and the dataset; see
`../Dataset/README.md` for the expected local data layout.

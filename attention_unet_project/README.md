# Attention U-Net Project

This directory preserves the independent Attention U-Net training and
evaluation code. It shares the optional local dataset folder with the other
models:

```text
../Dataset
```

The released inference checkpoint used by the web application is stored at:

```text
../models/weights/attention_unet/best_attention_unet.pt
```

Training command:

```powershell
cd .\attention_unet_project
.\run_fair_training.ps1
```

Equivalent batch launcher:

```cmd
cd attention_unet_project
run_fair_training.cmd
```

Evaluate the released checkpoint:

```powershell
..\.venv\Scripts\python.exe -m attention_unet.evaluate --checkpoint ..\models\weights\attention_unet\best_attention_unet.pt --data-dir ..\Dataset --split test
```

Experiment outputs are written below this project's `runs/` directory and are
not included in the published inference package.

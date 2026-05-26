@echo off
setlocal

set "ROOT=%~dp0"
for %%I in ("%ROOT%..") do set "REPO_ROOT=%%~fI"
set "RUN_DIR=%ROOT%runs\attention_unet_fair20_fixed_postprocess_seed42"
set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
set "DATASET=%REPO_ROOT%\Dataset"

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
cd /d "%ROOT%"

echo Starting Attention U-Net training...
echo Run dir: %RUN_DIR%
echo Stdout log: %RUN_DIR%\train.out.log
echo Stderr/progress log: %RUN_DIR%\train.err.log

"%PYTHON%" -m attention_unet.train ^
  --data-dir "%DATASET%" ^
  --epochs 20 ^
  --batch-size 4 ^
  --lr 0.001 ^
  --base-channels 16 ^
  --image-size "256,256" ^
  --augment basic ^
  --loss combo ^
  --loss-dice-weight 1.0 ^
  --loss-bce-weight 1.0 ^
  --loss-focal-weight 0.5 ^
  --scheduler plateau ^
  --threshold-min 0.12 ^
  --threshold-max 0.50 ^
  --threshold-steps 20 ^
  --early-stopping-patience 8 ^
  --postprocess ^
  --post-open-iters 1 ^
  --output-dir "%RUN_DIR%" ^
  1> "%RUN_DIR%\train.out.log" ^
  2> "%RUN_DIR%\train.err.log"

if errorlevel 1 (
  echo Training failed. Check "%RUN_DIR%\train.out.log" and "%RUN_DIR%\train.err.log".
  exit /b %errorlevel%
)

echo Training completed successfully.

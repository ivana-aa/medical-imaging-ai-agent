@echo off
chcp 65001 >nul
setlocal

title Medical Imaging AI Platform - Setup
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "PYTHON_EXE="

where py >nul 2>nul
if "%ERRORLEVEL%"=="0" set "PYTHON_EXE=py -3"
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if "%ERRORLEVEL%"=="0" set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] Python 3 was not found in PATH.
    echo Install Python 3.10 or later, then rerun setup.bat.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist "%VENV_DIR%\Scripts\python.exe" %PYTHON_EXE% -m venv "%VENV_DIR%"
if errorlevel 1 exit /b %errorlevel%

echo [2/4] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

echo [3/4] Installing dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 exit /b %errorlevel%

echo [4/4] Verifying bundled model weights...
"%VENV_DIR%\Scripts\python.exe" -c "from pathlib import Path; root=Path(r'%PROJECT_DIR%'); paths=[root/'models/weights/original_unet/best_unet.pt', root/'models/weights/attention_unet/best_attention_unet.pt', root/'models/weights/ssl_current/seed7_best_decoder.pt', root/'models/weights/ssl_current/seed42_best_decoder.pt', root/'models/weights/ssl_current/seed123_best_decoder.pt']; missing=[str(p) for p in paths if not p.exists()]; raise SystemExit('Missing Git LFS weights. Run git lfs pull first:\n' + '\n'.join(missing)) if missing else print('All deployment weights are present.')"
if errorlevel 1 (
    echo If this repository was cloned without model weights, install Git LFS and run:
    echo   git lfs install
    echo   git lfs pull
    pause
    exit /b 1
)

echo.
echo Setup complete. Run start.bat to open the local application.
pause

@echo off
chcp 65001 >nul
setlocal

title Medical Imaging Three-Model AI Platform

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"
set "VENV_PY=%PROJECT_DIR%.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON_EXE=%VENV_PY%"
) else (
    echo [ERROR] Virtual environment was not found.
    echo Run setup.bat once, then run start.bat again.
    pause
    exit /b 1
)

set "PYTHONPATH=%BACKEND_DIR%;%PROJECT_DIR%self_supervised_learning_project\src"

echo ============================================
echo   Medical Imaging Three-Model AI Platform
echo ============================================
echo.
echo Project: %PROJECT_DIR%
echo Backend: %BACKEND_DIR%
echo Python:  %PYTHON_EXE%
echo URL:     http://localhost:8000/
echo.

if not exist "%BACKEND_DIR%\main.py" (
    echo [ERROR] backend\main.py was not found.
    echo Please run this file from the medical-imaging-ai-agent folder.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"

if "%ERRORLEVEL%"=="0" (
    echo [OK] Server is already running on http://localhost:8000/
    start "" "http://localhost:8000/"
    echo.
    echo You can close this window.
    pause
    exit /b 0
)

echo [1/3] Preparing folders...
if not exist "%BACKEND_DIR%\uploads" mkdir "%BACKEND_DIR%\uploads"
if not exist "%BACKEND_DIR%\results" mkdir "%BACKEND_DIR%\results"
if not exist "%BACKEND_DIR%\watched_folders" mkdir "%BACKEND_DIR%\watched_folders"

echo [2/3] Opening browser...
start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8000/'"

echo [3/3] Starting backend server...
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C in this window to stop the server.
echo.

cd /d "%BACKEND_DIR%"
"%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause

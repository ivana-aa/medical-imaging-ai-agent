@echo off
chcp 65001 >nul
title 医影智诊 - 医疗影像AI分析系统 v3.0 (智能体版)

echo ============================================
echo    医影智诊 - 医疗影像AI分析平台
echo    智能体版 v3.0
echo ============================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

:: 进入后端目录
cd /d "%~dp0backend"

:: 检查依赖
echo [1/4] 检查并安装依赖...
pip install -r requirements.txt -q --no-warn-script-location

:: 创建必要目录
echo [2/4] 创建必要目录...
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "watched_folders" mkdir watched_folders

:: 启动服务
echo.
echo [3/4] 启动后端服务...
echo.
echo ============================================
echo   医影智诊 v3.0 智能体系统
echo ============================================
echo   前端界面:    http://localhost:8000
echo   API文档:     http://localhost:8000/docs
echo   WebSocket:   ws://localhost:8000/ws
echo   智能体:      FileWatcher | Dialogue | Planner
echo ============================================
echo.
echo   智能体功能：
echo   - 文件夹自动监控分析
echo   - AI多轮对话问答
echo   - 任务规划执行
echo.
echo   按 Ctrl+C 停止服务
echo ============================================
echo.

start "" http://localhost:8000

python main.py

pause

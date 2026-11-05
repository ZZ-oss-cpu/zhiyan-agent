@echo off
chcp 65001 >nul
setlocal
title 智研 Agent 一键启动
cd /d "%~dp0"

echo ============================================================
echo   智研 Agent - 学术论文智能分析系统
echo ============================================================
echo.

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PYEXE="

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import fastapi, uvicorn, qdrant_client, pymupdf" >nul 2>nul
    if not errorlevel 1 set "PYEXE=%VENV_PY%"
)

if not defined PYEXE (
    python -c "import fastapi, uvicorn, qdrant_client, pymupdf" >nul 2>nul
    if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE (
    echo [错误] 未找到已安装依赖的 Python 环境。
    echo.
    echo   请先安装依赖（推荐使用项目内虚拟环境）：
    echo     "%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo   若还没有虚拟环境：
    echo     C:\Python314\python.exe -m venv "%~dp0.venv"
    echo.
    pause
    exit /b 1
)

echo [1/3] 解释器：%PYEXE%
"%PYEXE%" -V

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 (
    echo [2/3] 端口 8000 已有服务在运行，直接打开浏览器。
    echo       注意：如果刚改过 app.py / index.html 等代码，旧进程不会加载新代码，
    echo             请先关掉原来的服务窗口（Ctrl+C 或关闭黑窗口）再运行本脚本，
    echo             否则页面上的新按钮会提示 404 或"测试接口不存在"。
    start "" "http://127.0.0.1:8000"
    timeout /t 2 >nul
    exit /b 0
)

echo [2/3] 正在启动服务（首次启动需加载向量库，请稍候）...
start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:8000'"

cd /d "%~dp0front_window"
echo [3/3] 服务地址：http://127.0.0.1:8000    按 Ctrl+C 可停止
echo.
"%PYEXE%" -m uvicorn app:app --host 127.0.0.1 --port 8000

echo.
echo 服务已停止。
pause

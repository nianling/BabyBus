@echo off
chcp 65001 >nul

:: -------------------------------
:: 自我提权为管理员
:: -------------------------------
:: 检查是否有管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 需要管理员权限，请确认弹出的 UAC 窗口。
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo 环境设置...
:: -------------------------------
:: 设置环境变量和路径
:: -------------------------------
setlocal
:: 获取当前 bat 所在目录（末尾有 \）
set BASE_DIR=%~dp0

:: 设置 PYTHONPATH
set PYTHONPATH=%BASE_DIR%

echo 正在启动.....
:: 使用虚拟环境的 python 运行 main.py
"%BASE_DIR%.venv\Scripts\python.exe" "%BASE_DIR%dnf\abyss\main.py"

endlocal
pause

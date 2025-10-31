@echo off
chcp 65001 >nul

:: -------------------------------
:: 自我提权为管理员
:: -------------------------------
:: 检查是否有管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 准备启动，需要管理员权限，请确认弹出的 UAC 窗口。
    timeout /t 1 /nobreak >nul
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
echo BASE_DIR=%BASE_DIR%

:: 设置 PYTHONPATH
set PYTHONPATH=%BASE_DIR%
echo PYTHONPATH=%PYTHONPATH%

:: 检查虚拟环境的 Python 是否存在
if not exist "%BASE_DIR%.venv\Scripts\python.exe" (
    echo 错误: 虚拟环境中的 Python 可执行文件不存在，请确认虚拟环境是否已创建。
    pause
    exit /b
)

:: 检查 main.py 是否存在
if not exist "%BASE_DIR%dnf\stronger\main.py" (
    echo 错误: main.py 文件不存在，请检查路径是否正确。
    pause
    exit /b
)

echo 正在启动.....
:: 使用虚拟环境的 python 运行 main.py
"%BASE_DIR%.venv\Scripts\python.exe" "%BASE_DIR%dnf\stronger\main.py"

endlocal
pause

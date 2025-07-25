@echo off
echo Starting Simple LoRA Training...
echo ================================

REM 尝试多种Python命令
echo Searching for Python...

REM 尝试python命令
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python command
    python train_lora_simple.py
    goto :end
)

REM 尝试py命令
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found py command
    py train_lora_simple.py
    goto :end
)

REM 尝试python3命令
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python3 command
    python3 train_lora_simple.py
    goto :end
)

REM 尝试常见的Python安装路径
set PYTHON_PATHS=^
"C:\Python39\python.exe" ^
"C:\Python310\python.exe" ^
"C:\Python311\python.exe" ^
"C:\Python312\python.exe" ^
"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe" ^
"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" ^
"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" ^
"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" ^
"C:\Users\%USERNAME%\Anaconda3\python.exe" ^
"C:\Users\%USERNAME%\miniconda3\python.exe"

for %%P in (%PYTHON_PATHS%) do (
    if exist %%P (
        echo Found Python at %%P
        %%P train_lora_simple.py
        goto :end
    )
)

echo ERROR: No Python installation found!
echo Please install Python from https://www.python.org/
echo Or make sure Python is in your PATH
pause

:end
echo Training completed or failed.
pause
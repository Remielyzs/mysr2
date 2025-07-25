@echo off
echo Starting Diffusion Model Training...

REM 尝试不同的Python命令
echo Trying python command...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python command
    python train_diffusion_improved.py --epochs 5 --batch-size 2 --device cuda
    goto :end
)

echo Trying python3 command...
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python3 command
    python3 train_diffusion_improved.py --epochs 5 --batch-size 2 --device cuda
    goto :end
)

echo Trying py command...
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found py command
    py train_diffusion_improved.py --epochs 5 --batch-size 2 --device cuda
    goto :end
)

echo Trying conda python...
conda activate base >nul 2>&1
if %errorlevel% == 0 (
    echo Activated conda base environment
    python train_diffusion_improved.py --epochs 5 --batch-size 2 --device cuda
    goto :end
)

echo No Python interpreter found. Please install Python or Anaconda.
echo You can download Python from: https://www.python.org/downloads/
echo Or Anaconda from: https://www.anaconda.com/products/distribution

:end
pause
# LoRA微调训练启动脚本

Write-Host "Starting LoRA fine-tuning for Stable Diffusion..." -ForegroundColor Green
Write-Host "Initializing GPU environment..." -ForegroundColor Yellow

# 检查GPU
Write-Host "Checking GPU availability..." -ForegroundColor Cyan

# 尝试不同的Python命令
$pythonCommands = @("python", "python3", "py")
$scriptFound = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found python command: $version" -ForegroundColor Green
            
            # 检查是否需要安装依赖
            Write-Host "Checking LoRA dependencies..." -ForegroundColor Yellow
            
            # 尝试导入关键包
            $checkScript = @"
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch not found')
    sys.exit(1)

try:
    import torchvision
    print(f'TorchVision: {torchvision.__version__}')
except ImportError:
    print('TorchVision not found')

try:
    from diffusers import StableDiffusionPipeline
    print('Diffusers: Available')
except ImportError:
    print('Diffusers: Not available (will use custom implementation)')

try:
    from peft import LoraConfig
    print('PEFT: Available')
except ImportError:
    print('PEFT: Not available (will use custom LoRA)')
"@
            
            Write-Host "Dependency check:" -ForegroundColor Cyan
            & $cmd -c $checkScript
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Starting LoRA fine-tuning training..." -ForegroundColor Green
                & $cmd train_lora_stable_diffusion.py
            } else {
                Write-Host "Dependencies missing. Please install requirements:" -ForegroundColor Red
                Write-Host "pip install -r requirements_lora.txt" -ForegroundColor Yellow
            }
            
            $scriptFound = $true
            break
        }
    }
    catch {
        continue
    }
}

if (-not $scriptFound) {
    Write-Host "Error: Python not found. Please install Python or Anaconda." -ForegroundColor Red
    Write-Host "Visit: https://www.python.org/downloads/ or https://www.anaconda.com/" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor White
    Write-Host "After installing Python, run:" -ForegroundColor Yellow
    Write-Host "pip install -r requirements_lora.txt" -ForegroundColor Cyan
    pause
}
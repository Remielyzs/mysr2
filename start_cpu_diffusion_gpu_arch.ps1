# CPU扩散模型训练启动脚本（GPU架构设计）

Write-Host "Starting CPU diffusion training with GPU architecture..." -ForegroundColor Green
Write-Host "Avoiding RTX 5090 compatibility issues..." -ForegroundColor Yellow

# 尝试不同的Python命令
$pythonCommands = @("python", "python3", "py")
$scriptFound = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found python command: $version" -ForegroundColor Green
            Write-Host "Starting CPU diffusion training..." -ForegroundColor Cyan
            & $cmd train_cpu_diffusion_gpu_arch.py
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
    pause
}
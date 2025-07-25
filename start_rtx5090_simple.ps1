# RTX 5090兼容的简化训练启动脚本

Write-Host "Starting RTX 5090 compatible simple training..." -ForegroundColor Green
Write-Host "Applying conservative GPU settings..." -ForegroundColor Yellow

# 尝试不同的Python命令
$pythonCommands = @("python", "python3", "py")
$scriptFound = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found python command: $version" -ForegroundColor Green
            Write-Host "Starting RTX 5090 compatible training..." -ForegroundColor Cyan
            & $cmd train_rtx5090_simple.py
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
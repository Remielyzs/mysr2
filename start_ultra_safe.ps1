# 超安全RTX 5090训练启动脚本

Write-Host "Starting ultra-safe RTX 5090 training..." -ForegroundColor Green
Write-Host "Using minimal model and safest operations..." -ForegroundColor Yellow

# 尝试不同的Python命令
$pythonCommands = @("python", "python3", "py")
$scriptFound = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found python command: $version" -ForegroundColor Green
            Write-Host "Starting ultra-safe training..." -ForegroundColor Cyan
            & $cmd train_ultra_safe.py
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
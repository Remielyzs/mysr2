# 简化的LoRA训练启动脚本
Write-Host "🚀 启动简化LoRA训练..." -ForegroundColor Green

# 尝试不同的Python命令
$pythonCommands = @("python", "python3", "py")
$pythonFound = $false

foreach ($cmd in $pythonCommands) {
    try {
        $null = Get-Command $cmd -ErrorAction Stop
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 找到Python: $version" -ForegroundColor Green
            Write-Host "📊 开始LoRA微调训练..." -ForegroundColor Cyan
            
            # 运行训练脚本
            & $cmd train_lora_simple.py
            $pythonFound = $true
            break
        }
    }
    catch {
        continue
    }
}

if (-not $pythonFound) {
    Write-Host "❌ 未找到Python！" -ForegroundColor Red
    Write-Host "请安装Python或确保Python在PATH中" -ForegroundColor Yellow
    Write-Host "下载地址: https://www.python.org/downloads/" -ForegroundColor Blue
    pause
}

Write-Host "训练完成或失败" -ForegroundColor Yellow
pause
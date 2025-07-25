# Start GPU-optimized diffusion model training
Write-Host "Starting GPU-optimized diffusion model training..." -ForegroundColor Green
Write-Host "Applying RTX 5090 compatibility fixes..." -ForegroundColor Yellow

# Try to find Python
$pythonCommands = @("python", "python3", "py")
$found = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found $cmd command: $version" -ForegroundColor Green
            Write-Host "Starting GPU training with compatibility fixes..." -ForegroundColor Yellow
            Write-Host ("=" * 60) -ForegroundColor Cyan
            
            # Run the GPU training script
            & $cmd train_gpu_diffusion.py
            
            $found = $true
            break
        }
    }
    catch {
        continue
    }
}

if (-not $found) {
    Write-Host "Python interpreter not found!" -ForegroundColor Red
    Write-Host "Please install Python or Anaconda." -ForegroundColor Yellow
}

Write-Host "Training completed. Press Enter to continue..." -ForegroundColor Green
Read-Host
# Start simple diffusion model training
Write-Host "Starting simple diffusion model training..." -ForegroundColor Green

# Try to find Python
$pythonCommands = @("python", "python3", "py")
$found = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found $cmd command: $version" -ForegroundColor Green
            Write-Host "Starting simple training..." -ForegroundColor Yellow
            Write-Host ("=" * 60) -ForegroundColor Cyan
            
            # Run the simple training script
            & $cmd train_simple_diffusion.py
            
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
# PowerShell script to start diffusion model training
Write-Host "Starting Diffusion Model Training..." -ForegroundColor Green

# Try different Python commands
$pythonCommands = @("python", "python3", "py")

foreach ($cmd in $pythonCommands) {
    Write-Host "Trying $cmd command..." -ForegroundColor Yellow
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found $cmd command: $version" -ForegroundColor Green
            Write-Host "Starting training with $cmd..." -ForegroundColor Green
            & $cmd train_diffusion_improved.py --epochs 5 --batch-size 2 --device cuda
            exit
        }
    }
    catch {
        Write-Host "$cmd not found" -ForegroundColor Red
    }
}

# Try conda activation
Write-Host "Trying conda activation..." -ForegroundColor Yellow
try {
    conda activate base
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Activated conda base environment" -ForegroundColor Green
        python train_diffusion_improved.py --epochs 5 --batch-size 2 --device cuda
        exit
    }
}
catch {
    Write-Host "Conda activation failed" -ForegroundColor Red
}

Write-Host "No Python interpreter found. Please install Python or Anaconda." -ForegroundColor Red
Write-Host "You can download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
Write-Host "Or Anaconda from: https://www.anaconda.com/products/distribution" -ForegroundColor Yellow

Read-Host "Press Enter to continue"
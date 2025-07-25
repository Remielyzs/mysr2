Write-Host "Starting LoRA fine-tuning process..." -ForegroundColor Green

# Find Python
$pythonCmd = $null
$commands = @("python", "python3", "py")

foreach ($cmd in $commands) {
    try {
        $null = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "Found Python: $cmd" -ForegroundColor Green
            break
        }
    }
    catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "Python not found. Please install Python." -ForegroundColor Red
    pause
    exit 1
}

# Check GPU
Write-Host "Checking GPU..." -ForegroundColor Yellow
& $pythonCmd -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Step 1: Prepare data
Write-Host "Step 1: Preparing data..." -ForegroundColor Cyan

# Check for existing images
$imageFound = $false
$dirs = @("./images", "./data/images", "./dataset")

foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        $files = Get-ChildItem -Path $dir -Recurse -Include "*.jpg", "*.png" -ErrorAction SilentlyContinue
        if ($files.Count -gt 0) {
            Write-Host "Found images in: $dir" -ForegroundColor Green
            $imageFound = $true
            $sourceDir = $dir
            break
        }
    }
}

if (-not $imageFound) {
    Write-Host "No images found. Creating sample data..." -ForegroundColor Yellow
    & $pythonCmd prepare_lora_data.py --create-samples --num-samples 50
    $sourceDir = "./data/sample_images"
}

# Process data
Write-Host "Processing images..." -ForegroundColor Cyan
& $pythonCmd prepare_lora_data.py --source $sourceDir --lr-size 64 --hr-size 256 --max-images 100

# Step 2: Start training
Write-Host "Step 2: Starting LoRA training..." -ForegroundColor Cyan
& $pythonCmd train_lora_stable_diffusion.py

Write-Host "Process completed!" -ForegroundColor Green
pause
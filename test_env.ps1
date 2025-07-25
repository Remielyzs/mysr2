# Test environment script
Write-Host "Testing Diffusion Model Environment..." -ForegroundColor Green

# Try to find and run Python
$pythonCommands = @("python", "python3", "py")
$found = $false

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found $cmd command: $version" -ForegroundColor Green
            Write-Host "Running environment test..." -ForegroundColor Yellow
            & $cmd test_environment.py
            $found = $true
            break
        }
    }
    catch {
        continue
    }
}

if (-not $found) {
    Write-Host "No Python interpreter found!" -ForegroundColor Red
    Write-Host "Please install Python or Anaconda." -ForegroundColor Yellow
}

Read-Host "Press Enter to continue"
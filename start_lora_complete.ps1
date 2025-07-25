# 完整的LoRA微调流程启动脚本

Write-Host "🚀 LoRA微调完整流程启动" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

# 检查Python环境
Write-Host "🔍 检查Python环境..." -ForegroundColor Yellow

$pythonCommands = @("python", "python3", "py")
$pythonCmd = $null

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 找到Python: $version" -ForegroundColor Green
            $pythonCmd = $cmd
            break
        }
    }
    catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "❌ 未找到Python，请先安装Python" -ForegroundColor Red
    Write-Host "访问: https://www.python.org/downloads/" -ForegroundColor Yellow
    pause
    exit 1
}

# 检查GPU
Write-Host "🎮 检查GPU环境..." -ForegroundColor Yellow
$gpuCheck = @"
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ 未检测到CUDA GPU')
"@

& $pythonCmd -c $gpuCheck

# 步骤1: 数据准备
Write-Host ""
Write-Host "📊 步骤1: 数据准备" -ForegroundColor Cyan
Write-Host "检查是否存在图像数据..." -ForegroundColor Yellow

# 检查是否有现有图像
$imageFound = $false
$imageDirs = @("./images", "./data/images", "./dataset", "./pics", "./photos")

foreach ($dir in $imageDirs) {
    if (Test-Path $dir) {
        $imageFiles = Get-ChildItem -Path $dir -Recurse -Include "*.jpg", "*.jpeg", "*.png", "*.bmp" -ErrorAction SilentlyContinue
        if ($imageFiles.Count -gt 0) {
            Write-Host "✅ 在 $dir 中找到 $($imageFiles.Count) 个图像文件" -ForegroundColor Green
            $imageFound = $true
            $sourceDir = $dir
            break
        }
    }
}

if (-not $imageFound) {
    Write-Host "⚠️ 未找到现有图像，将创建示例数据" -ForegroundColor Yellow
    Write-Host "创建示例图像数据..." -ForegroundColor Cyan
    
    & $pythonCmd prepare_lora_data.py --create-samples --num-samples 100
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 示例数据创建成功" -ForegroundColor Green
        $sourceDir = "./data/sample_images"
    } else {
        Write-Host "❌ 示例数据创建失败" -ForegroundColor Red
        pause
        exit 1
    }
} else {
    Write-Host "使用找到的图像数据: $sourceDir" -ForegroundColor Green
}

# 处理数据
Write-Host ""
Write-Host "🔄 处理图像数据..." -ForegroundColor Cyan
& $pythonCmd prepare_lora_data.py --source $sourceDir --lr-size 64 --hr-size 256 --max-images 200

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 数据处理失败" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ 数据准备完成" -ForegroundColor Green

# 步骤2: 检查依赖
Write-Host ""
Write-Host "📦 步骤2: 检查依赖包" -ForegroundColor Cyan

$depCheck = @"
missing_packages = []
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    missing_packages.append('torch')
    print('❌ PyTorch: 未安装')

try:
    import torchvision
    print(f'✅ TorchVision: {torchvision.__version__}')
except ImportError:
    missing_packages.append('torchvision')
    print('❌ TorchVision: 未安装')

try:
    from PIL import Image
    print('✅ Pillow: 已安装')
except ImportError:
    missing_packages.append('Pillow')
    print('❌ Pillow: 未安装')

try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
except ImportError:
    missing_packages.append('numpy')
    print('❌ NumPy: 未安装')

if missing_packages:
    print(f'缺少包: {", ".join(missing_packages)}')
    print('请运行: pip install -r requirements_lora.txt')
    exit(1)
else:
    print('✅ 所有依赖包已安装')
"@

& $pythonCmd -c $depCheck

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 依赖检查失败，正在安装依赖..." -ForegroundColor Yellow
    
    if (Test-Path "requirements_lora.txt") {
        & $pythonCmd -m pip install -r requirements_lora.txt
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 依赖安装失败" -ForegroundColor Red
            pause
            exit 1
        }
    } else {
        Write-Host "❌ 未找到requirements_lora.txt文件" -ForegroundColor Red
        pause
        exit 1
    }
}

# 步骤3: 开始训练
Write-Host ""
Write-Host "🎯 步骤3: 开始LoRA微调训练" -ForegroundColor Cyan
Write-Host "启动训练脚本..." -ForegroundColor Yellow

& $pythonCmd train_lora_stable_diffusion.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 LoRA微调训练完成！" -ForegroundColor Green
    Write-Host "检查生成的模型文件..." -ForegroundColor Yellow
    
    if (Test-Path "lora_model.pth") {
        Write-Host "✅ LoRA模型已保存: lora_model.pth" -ForegroundColor Green
    }
    
    if (Test-Path "base_model.pth") {
        Write-Host "✅ 基础模型已保存: base_model.pth" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "📊 训练总结:" -ForegroundColor Cyan
    Write-Host "- 数据源: $sourceDir" -ForegroundColor White
    Write-Host "- 模型类型: LoRA微调的超分辨率模型" -ForegroundColor White
    Write-Host "- 输入尺寸: 64x64" -ForegroundColor White
    Write-Host "- 输出尺寸: 256x256" -ForegroundColor White
    
} else {
    Write-Host ""
    Write-Host "❌ 训练失败" -ForegroundColor Red
    Write-Host "请检查错误信息并重试" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Gray
pause
# PowerShell脚本用于分割DIV2K数据
# 创建输出目录
$outputDir = "data\split_sample"
$trainLrDir = "$outputDir\train\lr"
$trainHrDir = "$outputDir\train\hr"
$valLrDir = "$outputDir\val\lr"
$valHrDir = "$outputDir\val\hr"

# 创建目录
New-Item -ItemType Directory -Path $trainLrDir -Force | Out-Null
New-Item -ItemType Directory -Path $trainHrDir -Force | Out-Null
New-Item -ItemType Directory -Path $valLrDir -Force | Out-Null
New-Item -ItemType Directory -Path $valHrDir -Force | Out-Null

Write-Host "创建输出目录完成"

# 获取HR图像列表
$hrDir = "data\DIV2K\DIV2K_train_HR"
$lrDir = "data\DIV2K\DIV2K_train_LR_bicubic\X4"

$hrImages = Get-ChildItem -Path $hrDir -Filter "*.png" | Sort-Object Name
$totalImages = $hrImages.Count

Write-Host "找到 $totalImages 张HR图像"

# 计算分割点 (80% 训练, 20% 验证)
$splitPoint = [math]::Floor($totalImages * 0.8)
$trainImages = $hrImages[0..($splitPoint-1)]
$valImages = $hrImages[$splitPoint..($totalImages-1)]

Write-Host "训练集: $($trainImages.Count) 张图像"
Write-Host "验证集: $($valImages.Count) 张图像"

# 复制训练数据
Write-Host "复制训练数据..."
foreach ($hrImage in $trainImages) {
    $imageId = [System.IO.Path]::GetFileNameWithoutExtension($hrImage.Name)
    $lrImageName = "${imageId}x4.png"
    $lrImagePath = Join-Path $lrDir $lrImageName
    
    if (Test-Path $lrImagePath) {
        Copy-Item $hrImage.FullName -Destination $trainHrDir
        Copy-Item $lrImagePath -Destination $trainLrDir
    } else {
        Write-Warning "未找到对应的LR图像: $lrImagePath"
    }
}

# 复制验证数据
Write-Host "复制验证数据..."
foreach ($hrImage in $valImages) {
    $imageId = [System.IO.Path]::GetFileNameWithoutExtension($hrImage.Name)
    $lrImageName = "${imageId}x4.png"
    $lrImagePath = Join-Path $lrDir $lrImageName
    
    if (Test-Path $lrImagePath) {
        Copy-Item $hrImage.FullName -Destination $valHrDir
        Copy-Item $lrImagePath -Destination $valLrDir
    } else {
        Write-Warning "未找到对应的LR图像: $lrImagePath"
    }
}

Write-Host "数据分割完成!"
Write-Host "训练数据保存在: $trainLrDir 和 $trainHrDir"
Write-Host "验证数据保存在: $valLrDir 和 $valHrDir"
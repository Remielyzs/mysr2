#!/usr/bin/env python3
"""
LoRA微调数据准备脚本
用于准备和预处理图像数据
"""

import os
import shutil
from pathlib import Path
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Tuple
import json

def create_data_structure():
    """创建数据目录结构"""
    data_dir = Path("./data")
    
    # 创建目录
    dirs_to_create = [
        data_dir / "images" / "train",
        data_dir / "images" / "val",
        data_dir / "processed" / "lr",
        data_dir / "processed" / "hr",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    return data_dir

def find_images(directory: Path) -> List[Path]:
    """查找目录中的图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    if directory.exists():
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
    
    return image_files

def process_image_for_sr(image_path: Path, lr_size: int = 64, hr_size: int = 256) -> Tuple[Image.Image, Image.Image]:
    """处理图像生成LR和HR对"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 创建HR版本（高分辨率）
        hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size), Image.LANCZOS),
        ])
        hr_image = hr_transform(image)
        
        # 创建LR版本（低分辨率）
        lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size), Image.LANCZOS),
        ])
        lr_image = lr_transform(image)
        
        return lr_image, hr_image
        
    except Exception as e:
        print(f"❌ 处理图像失败 {image_path}: {e}")
        return None, None

def prepare_dataset(source_dir: str, output_dir: str, lr_size: int = 64, hr_size: int = 256, max_images: int = None):
    """准备数据集"""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print(f"🔍 在 {source_path} 中查找图像...")
    image_files = find_images(source_path)
    
    if not image_files:
        print(f"⚠️ 在 {source_path} 中未找到图像文件")
        return False
    
    print(f"📸 找到 {len(image_files)} 个图像文件")
    
    if max_images:
        image_files = image_files[:max_images]
        print(f"📊 限制处理 {len(image_files)} 个图像")
    
    # 创建输出目录
    lr_dir = output_path / "lr"
    hr_dir = output_path / "hr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理图像
    processed_count = 0
    failed_count = 0
    
    for i, image_path in enumerate(image_files):
        print(f"处理 [{i+1}/{len(image_files)}]: {image_path.name}")
        
        lr_image, hr_image = process_image_for_sr(image_path, lr_size, hr_size)
        
        if lr_image and hr_image:
            # 保存LR和HR图像
            base_name = image_path.stem
            lr_path = lr_dir / f"{base_name}_lr.png"
            hr_path = hr_dir / f"{base_name}_hr.png"
            
            lr_image.save(lr_path)
            hr_image.save(hr_path)
            
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"\n✅ 数据准备完成!")
    print(f"📊 成功处理: {processed_count} 个图像")
    print(f"❌ 失败: {failed_count} 个图像")
    print(f"📁 LR图像保存到: {lr_dir}")
    print(f"📁 HR图像保存到: {hr_dir}")
    
    # 保存数据集信息
    dataset_info = {
        'total_images': len(image_files),
        'processed_images': processed_count,
        'failed_images': failed_count,
        'lr_size': lr_size,
        'hr_size': hr_size,
        'lr_dir': str(lr_dir),
        'hr_dir': str(hr_dir)
    }
    
    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"📋 数据集信息保存到: {info_path}")
    return True

def create_sample_images(output_dir: str, num_samples: int = 50):
    """创建示例图像用于测试"""
    output_path = Path(output_dir)
    sample_dir = output_path / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 创建 {num_samples} 个示例图像...")
    
    # 创建不同类型的示例图像
    for i in range(num_samples):
        # 创建随机彩色图像
        image = Image.new('RGB', (512, 512))
        pixels = []
        
        for y in range(512):
            for x in range(512):
                # 创建渐变效果
                r = int(255 * (x / 512))
                g = int(255 * (y / 512))
                b = int(255 * ((x + y) / 1024))
                
                # 添加一些随机性
                import random
                r = max(0, min(255, r + random.randint(-50, 50)))
                g = max(0, min(255, g + random.randint(-50, 50)))
                b = max(0, min(255, b + random.randint(-50, 50)))
                
                pixels.append((r, g, b))
        
        image.putdata(pixels)
        
        # 保存图像
        image_path = sample_dir / f"sample_{i:03d}.png"
        image.save(image_path)
    
    print(f"✅ 示例图像创建完成，保存到: {sample_dir}")
    return sample_dir

def main():
    parser = argparse.ArgumentParser(description="LoRA微调数据准备")
    parser.add_argument("--source", type=str, help="源图像目录")
    parser.add_argument("--output", type=str, default="./data/processed", help="输出目录")
    parser.add_argument("--lr-size", type=int, default=64, help="LR图像尺寸")
    parser.add_argument("--hr-size", type=int, default=256, help="HR图像尺寸")
    parser.add_argument("--max-images", type=int, help="最大处理图像数")
    parser.add_argument("--create-samples", action="store_true", help="创建示例图像")
    parser.add_argument("--num-samples", type=int, default=50, help="示例图像数量")
    
    args = parser.parse_args()
    
    print("🚀 LoRA微调数据准备")
    print("=" * 40)
    
    # 创建数据目录结构
    data_dir = create_data_structure()
    
    if args.create_samples:
        # 创建示例图像
        sample_dir = create_sample_images(data_dir, args.num_samples)
        
        # 使用示例图像作为源
        if not args.source:
            args.source = str(sample_dir)
            print(f"📁 使用示例图像作为源: {args.source}")
    
    if args.source:
        # 准备数据集
        success = prepare_dataset(
            source_dir=args.source,
            output_dir=args.output,
            lr_size=args.lr_size,
            hr_size=args.hr_size,
            max_images=args.max_images
        )
        
        if success:
            print(f"\n🎯 数据准备完成！现在可以开始LoRA微调训练")
            print(f"运行命令: python train_lora_stable_diffusion.py")
        else:
            print(f"\n❌ 数据准备失败")
    else:
        print(f"\n💡 使用方法:")
        print(f"  创建示例数据: python prepare_lora_data.py --create-samples")
        print(f"  处理现有图像: python prepare_lora_data.py --source /path/to/images")
        print(f"  完整示例: python prepare_lora_data.py --source ./images --lr-size 64 --hr-size 256 --max-images 100")

if __name__ == "__main__":
    main()
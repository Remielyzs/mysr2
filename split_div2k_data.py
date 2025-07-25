#!/usr/bin/env python3
"""
DIV2K数据集分割脚本
专门用于处理DIV2K数据集的文件名格式
"""

import os
import shutil
import random
import glob
from pathlib import Path

def split_div2k_data(hr_source_dir, lr_source_dir, output_base_dir, split_ratio=0.8):
    """
    分割DIV2K数据集为训练集和验证集
    
    Args:
        hr_source_dir: 高分辨率图像目录 (DIV2K_train_HR)
        lr_source_dir: 低分辨率图像目录 (DIV2K_train_LR_bicubic/X4)
        output_base_dir: 输出目录
        split_ratio: 训练集比例
    """
    
    # 创建输出目录
    train_lr_dir = Path(output_base_dir) / 'train' / 'lr'
    train_hr_dir = Path(output_base_dir) / 'train' / 'hr'
    val_lr_dir = Path(output_base_dir) / 'val' / 'lr'
    val_hr_dir = Path(output_base_dir) / 'val' / 'hr'
    
    for d in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 获取HR图像列表
    hr_images = sorted(glob.glob(os.path.join(hr_source_dir, '*.png')))
    
    if not hr_images:
        print(f"在HR目录中未找到图像: {hr_source_dir}")
        return
    
    # 匹配LR和HR图像对
    matched_pairs = []
    for hr_path in hr_images:
        hr_filename = os.path.basename(hr_path)
        # 从HR文件名提取编号 (例如: 0001.png -> 0001)
        image_id = os.path.splitext(hr_filename)[0]
        
        # 构造对应的LR文件名 (例如: 0001x4.png)
        lr_filename = f"{image_id}x4.png"
        lr_path = os.path.join(lr_source_dir, lr_filename)
        
        if os.path.exists(lr_path):
            matched_pairs.append((lr_path, hr_path))
        else:
            print(f"警告: 未找到对应的LR图像 {lr_path}")
    
    if not matched_pairs:
        print("未找到匹配的LR/HR图像对")
        return
    
    print(f"找到 {len(matched_pairs)} 对匹配的LR/HR图像")
    
    # 随机打乱
    random.seed(42)  # 设置随机种子以确保可重现性
    random.shuffle(matched_pairs)
    
    # 计算分割点
    split_point = int(len(matched_pairs) * split_ratio)
    
    # 分割数据
    train_pairs = matched_pairs[:split_point]
    val_pairs = matched_pairs[split_point:]
    
    print(f"数据分割: {len(train_pairs)} 训练样本, {len(val_pairs)} 验证样本")
    
    # 复制训练数据
    print("复制训练数据...")
    for lr_src, hr_src in train_pairs:
        lr_dest = train_lr_dir / os.path.basename(lr_src)
        hr_dest = train_hr_dir / os.path.basename(hr_src)
        shutil.copy2(lr_src, lr_dest)
        shutil.copy2(hr_src, hr_dest)
    
    # 复制验证数据
    print("复制验证数据...")
    for lr_src, hr_src in val_pairs:
        lr_dest = val_lr_dir / os.path.basename(lr_src)
        hr_dest = val_hr_dir / os.path.basename(hr_src)
        shutil.copy2(lr_src, hr_dest)
        shutil.copy2(hr_src, hr_dest)
    
    print("数据分割完成!")
    print(f"训练数据保存在: {train_lr_dir.parent}")
    print(f"验证数据保存在: {val_lr_dir.parent}")

def main():
    """主函数"""
    # DIV2K数据路径
    hr_source_dir = "data/DIV2K/DIV2K_train_HR"
    lr_source_dir = "data/DIV2K/DIV2K_train_LR_bicubic/X4"
    output_base_dir = "data/split_sample"
    
    print("=" * 60)
    print("DIV2K数据集分割")
    print("=" * 60)
    print(f"HR源目录: {hr_source_dir}")
    print(f"LR源目录: {lr_source_dir}")
    print(f"输出目录: {output_base_dir}")
    print()
    
    # 检查源目录是否存在
    if not os.path.exists(hr_source_dir):
        print(f"错误: HR源目录不存在: {hr_source_dir}")
        return
    
    if not os.path.exists(lr_source_dir):
        print(f"错误: LR源目录不存在: {lr_source_dir}")
        return
    
    # 执行分割
    split_div2k_data(hr_source_dir, lr_source_dir, output_base_dir)

if __name__ == '__main__':
    main()
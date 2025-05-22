"""通用数据集下载脚本"""

import os
import argparse
from pathlib import Path
from datasets.div2k_config import DIV2KConfig
from datasets.download_manager import DownloadManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='通用数据集下载器')
    
    # 数据集类型参数
    parser.add_argument('--dataset', type=str, default='div2k',
                        help='要下载的数据集名称，目前支持：div2k')
    
    # 数据类型参数（针对DIV2K）
    parser.add_argument('--types', nargs='+', choices=['hr', 'lr_bicubic', 'lr_unknown', 'lr_special'],
                        help='要下载的数据类型，可选：hr, lr_bicubic, lr_unknown, lr_special')
    
    # 数据集划分参数
    parser.add_argument('--splits', nargs='+', choices=['train', 'valid'],
                        help='要下载的数据集划分，可选：train, valid')
    
    # 下载目录参数
    parser.add_argument('--download-dir', type=str, default='./data_download_temp',
                        help='下载文件保存目录')
    
    # 解压目录参数
    parser.add_argument('--extract-dir', type=str, default='./data',
                        help='数据集解压目录')
    
    # 是否清理下载的压缩文件
    parser.add_argument('--clean-downloads', action='store_true', default= False,
                        help='是否清理下载的压缩文件')
    
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 根据数据集类型获取下载URL列表
    if args.dataset.lower() == 'div2k':
        urls = DIV2KConfig.get_download_urls(args.types, args.splits)
        extract_subdir = 'DIV2K'
    else:
        print(f"错误：不支持的数据集类型 '{args.dataset}'")
        return

    if not urls:
        print("没有找到需要下载的文件，请检查参数设置。")
        return
    
    # 创建下载管理器
    download_manager = DownloadManager(
        download_dir=args.download_dir,
        extract_dir=args.extract_dir
    )
    
    # 下载并解压文件
    success_count = 0
    total_files = len(urls)
    
    print(f"\n开始下载和解压 {total_files} 个文件...")
    for url in urls:
        filename = url.split('/')[-1]
        print(f"\n处理文件: {filename}")
        
        extract_path = download_manager.download_and_extract(
            url=url,
            filename=filename,
            extract_subdir=extract_subdir  # 使用数据集特定的解压目录
        )
        
        if extract_path is not None:
            success_count += 1
    
    # 清理下载文件（如果需要）
    if args.clean_downloads:
        download_manager.cleanup_downloads()
    
    # 打印总结信息
    print(f"\n下载和解压完成！")
    print(f"成功处理: {success_count}/{total_files} 个文件")
    print(f"数据集保存在: {args.extract_dir}")
    if not args.clean_downloads:
        print(f"下载的压缩文件保存在: {args.download_dir}")

if __name__ == '__main__':
    main()
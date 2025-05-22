"""数据集下载管理器"""

import os
import hashlib
import requests
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path

class DownloadManager:
    """数据集下载管理器"""
    
    def __init__(self, download_dir='./data_download_temp', extract_dir='./data'):
        """初始化下载管理器
        
        Args:
            download_dir: 下载目录
            extract_dir: 解压目录
        """
        self.download_dir = Path(download_dir)
        self.extract_dir = Path(extract_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_files = []
        
    def download_file(self, url, filename=None):
        """下载文件
        
        Args:
            url: 下载URL
            filename: 保存的文件名，如果为None则从URL中获取
            
        Returns:
            下载文件的路径
        """
        if filename is None:
            filename = url.split('/')[-1]
        
        file_path = self.download_dir / filename
        if file_path.exists():
            print(f"File {filename} already exists, skipping download...")
            return file_path
        
        print(f"\nDownloading {url} to {file_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 使用tqdm显示下载进度
        with open(file_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        return file_path
    
    def verify_file(self, file_path, expected_hash=None):
        """验证文件完整性
        
        Args:
            file_path: 文件路径
            expected_hash: 预期的哈希值，如果为None则跳过验证
            
        Returns:
            bool: 验证是否通过
        """
        if expected_hash is None:
            return True
            
        print(f"Verifying {file_path}...")
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        actual_hash = sha256_hash.hexdigest()
        if actual_hash != expected_hash:
            print(f"Hash mismatch for {file_path}")
            print(f"Expected: {expected_hash}")
            print(f"Got: {actual_hash}")
            return False
        
        print(f"Hash verified for {file_path}")
        return True
    
    def extract_zip(self, zip_path, extract_subdir=None):
        """解压ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            extract_subdir: 解压到的子目录，如果为None则使用ZIP文件名（不含扩展名）
            
        Returns:
            解压目录的路径
        """
        if extract_subdir is None:
            extract_subdir = zip_path.stem
        
        extract_path = self.extract_dir / extract_subdir
        extract_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExtracting {zip_path} to {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取所有文件的总大小
            total_size = sum(info.file_size for info in zip_ref.filelist)
            extracted_size = 0
            
            # 使用tqdm显示解压进度
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for member in zip_ref.namelist():
                    # 解压单个文件
                    member_path = extract_path / member
                    # 如果文件已存在，先删除
                    if member_path.exists():
                        if member_path.is_dir():
                            shutil.rmtree(member_path)
                        else:
                            member_path.unlink()
                    # 解压文件
                    zip_ref.extract(member, extract_path)
                    # 更新进度条
                    extracted_size += zip_ref.getinfo(member).file_size
                    pbar.update(zip_ref.getinfo(member).file_size)
        
        return extract_path
    
    def cleanup_downloads(self):
        """清理下载目录"""
        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
            print(f"Cleaned up download directory {self.download_dir}")
            
    def download_files(self, urls, filenames=None, expected_hashes=None):
        """批量下载文件
        
        Args:
            urls: 下载URL列表
            filenames: 保存的文件名列表，如果为None则从URL中获取
            expected_hashes: 预期的哈希值列表
            
        Returns:
            下载文件的路径列表
        """
        if filenames is None:
            filenames = [None] * len(urls)
        if expected_hashes is None:
            expected_hashes = [None] * len(urls)
            
        for url, filename, expected_hash in zip(urls, filenames, expected_hashes):
            try:
                # 下载文件
                file_path = self.download_file(url, filename)
                
                # 验证文件
                if not self.verify_file(file_path, expected_hash):
                    print(f"File verification failed for {file_path}")
                    continue
                    
                self.downloaded_files.append(file_path)
                
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                
        return self.downloaded_files
    
    def extract_all_files(self):
        """批量解压所有下载的文件，并按照DIV2K的目录结构组织"""
        import re
        
        for zip_path in self.downloaded_files:
            try:
                filename = zip_path.name
                if not filename.startswith('DIV2K_'):
                    continue
                    
                base_name = zip_path.stem
                
                # 确定目标子目录
                if base_name.endswith('_HR'):
                    target_subdir = self.extract_dir / base_name
                    expected_content_folder = base_name
                else:
                    # 匹配带有缩放因子的LR
                    match_lr_scale = re.match(r'(DIV2K_(train|valid)_LR_(bicubic|unknown))_X([234])', base_name)
                    if match_lr_scale:
                        lr_type_base = match_lr_scale.group(1)
                        scale = f'X{match_lr_scale.group(4)}'
                        target_subdir = self.extract_dir / lr_type_base / scale
                        expected_content_folder = f"{lr_type_base}/{scale}"
                    else:
                        # 匹配其他类型的LR
                        match_lr_other = re.match(r'(DIV2K_(train|valid)_LR_(difficult|mild|wild|x8))', base_name)
                        if match_lr_other:
                            target_subdir = self.extract_dir / base_name
                            expected_content_folder = base_name
                        else:
                            print(f"Warning: Could not determine target subdirectory for {filename}")
                            target_subdir = self.extract_dir / base_name
                            expected_content_folder = base_name
                
                # 创建临时解压目录
                temp_extract_path = self.extract_dir / f"_temp_extract_{base_name}"
                if temp_extract_path.exists():
                    shutil.rmtree(temp_extract_path)
                temp_extract_path.mkdir(parents=True, exist_ok=True)
                
                # 解压到临时目录
                print(f"\nExtracting {filename} to temporary directory")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_path)
                
                # 确保目标目录存在
                target_subdir.mkdir(parents=True, exist_ok=True)
                
                # 移动文件到目标目录
                actual_content_dir = temp_extract_path
                potential_content_dir = temp_extract_path / expected_content_folder
                if potential_content_dir.exists():
                    actual_content_dir = potential_content_dir
                
                # 移动文件
                for item in actual_content_dir.iterdir():
                    dest = target_subdir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            # 如果目标是目录，合并内容
                            for sub_item in item.iterdir():
                                sub_dest = dest / sub_item.name
                                if sub_dest.exists():
                                    if sub_dest.is_dir():
                                        shutil.rmtree(sub_dest)
                                    else:
                                        sub_dest.unlink()
                                shutil.move(str(sub_item), str(sub_dest))
                            item.rmdir()  # 删除空源目录
                        else:
                            # 如果目标是文件，直接覆盖
                            dest.unlink()
                            shutil.move(str(item), str(dest))
                    else:
                        # 如果目标不存在，直接移动
                        shutil.move(str(item), str(dest))
                
                # 清理临时目录
                shutil.rmtree(temp_extract_path)
                print(f"Successfully extracted and organized {filename}")
                
            except Exception as e:
                print(f"Error extracting {zip_path}: {str(e)}")
                if 'temp_extract_path' in locals() and temp_extract_path.exists():
                    shutil.rmtree(temp_extract_path)
    
    def download_and_extract(self, url, filename=None, extract_subdir=None, expected_hash=None):
        """下载并解压单个文件（保持向后兼容）
        
        Args:
            url: 下载URL
            filename: 保存的文件名
            extract_subdir: 解压到的子目录
            expected_hash: 预期的哈希值
            
        Returns:
            解压目录的路径
        """
        try:
            # 下载文件
            file_path = self.download_file(url, filename)
            
            # 验证文件
            if not self.verify_file(file_path, expected_hash):
                print(f"File verification failed for {file_path}")
                return None
            
            # 解压文件
            extract_path = self.extract_zip(file_path, extract_subdir)
            return extract_path
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None
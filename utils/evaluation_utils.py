import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchmetrics
from typing import Optional, Tuple, List, Dict, Any, Union

class EvaluationUtils:
    """工具类，用于封装超分辨率模型评估中的通用功能"""
    
    def __init__(self, device: str = 'cpu'):
        """初始化评估工具类
        
        Args:
            device: 计算设备，默认为'cpu'
        """
        self.device = device
        self.transform = transforms.ToTensor()
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
        self.ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    def load_and_preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """加载并预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像张量，如果加载失败则返回None
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def calculate_metrics(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> Tuple[float, float]:
        """计算PSNR和SSIM指标
        
        Args:
            sr_tensor: 超分辨率结果张量
            hr_tensor: 高分辨率目标张量
            
        Returns:
            PSNR值和SSIM值的元组
        """
        if sr_tensor.shape != hr_tensor.shape:
            raise ValueError(f"Shape mismatch: SR shape {sr_tensor.shape} != HR shape {hr_tensor.shape}")
            
        psnr_value = self.psnr_metric(sr_tensor, hr_tensor)
        ssim_value = self.ssim_metric(sr_tensor, hr_tensor)
        return psnr_value.item(), ssim_value.item()
    
    def save_image(self, tensor: torch.Tensor, save_path: str) -> bool:
        """保存图像张量为文件
        
        Args:
            tensor: 要保存的图像张量
            save_path: 保存路径
            
        Returns:
            保存是否成功
        """
        try:
            image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
            image.save(save_path)
            return True
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")
            return False
    
    def process_batch_metrics(self, metrics_list: List[Tuple[float, float]]) -> Dict[str, float]:
        """处理批量评估的指标结果
        
        Args:
            metrics_list: 包含(PSNR, SSIM)元组的列表
            
        Returns:
            包含平均PSNR和SSIM值的字典
        """
        if not metrics_list:
            return {'avg_psnr': None, 'avg_ssim': None}
            
        psnr_values = [m[0] for m in metrics_list]
        ssim_values = [m[1] for m in metrics_list]
        
        return {
            'avg_psnr': np.mean(psnr_values),
            'avg_ssim': np.mean(ssim_values)
        }
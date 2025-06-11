import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology, filters, measure
from skimage.feature import canny
from skimage.morphology import skeletonize
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Dict, Tuple, Any
import seaborn as sns
from pathlib import Path
from models.simple_srcnn import SimpleSRCNN  # 导入SimpleSRCNN模型

from data_utils import SRDataset
from config.experiment_config import ExperimentConfig # 导入ExperimentConfig

class SuperResolutionDataset(SRDataset):
    """超分辨率验证数据集，继承自SRDataset"""
    def __init__(self, lr_path: str, hr_path: str, transform=None, edge_methods=None, device='cpu', lr_patch_size=None, upscale_factor=None, image_size=None):
        # 调用父类的初始化方法，直接传递参数
        super().__init__(
            lr_dir=None,
            hr_dir=None,
            transform=transform,
            mode='eval',
            edge_methods=edge_methods,
            device=device,
            val_lr_dir=lr_path,
            val_hr_dir=hr_path,
            lr_patch_size=lr_patch_size,
            upscale_factor=upscale_factor,
            image_size=image_size
        )

class MetricsCalculator:
    """图像质量指标计算器"""
    
    @staticmethod
    def psnr(img1, img2):
        """计算PSNR"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    @staticmethod
    def ssim(img1, img2):
        """计算SSIM (简化版本)"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    @staticmethod
    def lpips_simple(img1, img2):
        """简化的感知损失计算"""
        # 转换为灰度图进行简单的边缘感知比较
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # 使用Sobel算子检测边缘
        sobel1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=3)
        sobel2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=3)
        
        # 计算边缘差异
        edge_diff = np.mean(np.abs(sobel1 - sobel2))
        return edge_diff / 255.0

class EdgeSkeletonAnalyzer:
    """边缘和骨架分析器"""
    
    @staticmethod
    def extract_edges(image, method='canny', **kwargs):
        """提取图像边缘"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method == 'canny':
            low = kwargs.get('low_threshold', 50)
            high = kwargs.get('high_threshold', 150)
            edges = canny(gray, low_threshold=low, high_threshold=high)
        elif method == 'sobel':
            edges = filters.sobel(gray)
        elif method == 'laplacian':
            edges = filters.prewitt(gray)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        return edges.astype(np.uint8) * 255
    
    @staticmethod
    def extract_skeleton(image, method='medial_axis'):
        """提取图像骨架"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 二值化
        binary = gray > filters.threshold_otsu(gray)
        
        if method == 'medial_axis':
            skeleton = morphology.medial_axis(binary)
        elif method == 'skeletonize':
            skeleton = skeletonize(binary)
        else:
            raise ValueError(f"Unknown skeleton extraction method: {method}")
        
        return skeleton.astype(np.uint8) * 255
    
    @staticmethod
    def analyze_structure_preservation(hr_img, sr_img):
        """分析结构保持程度"""
        # 提取边缘
        hr_edges = EdgeSkeletonAnalyzer.extract_edges(hr_img)
        sr_edges = EdgeSkeletonAnalyzer.extract_edges(sr_img)
        
        # 提取骨架
        hr_skeleton = EdgeSkeletonAnalyzer.extract_skeleton(hr_img)
        sr_skeleton = EdgeSkeletonAnalyzer.extract_skeleton(sr_img)
        
        # 计算边缘保持度
        edge_preservation = np.mean(hr_edges == sr_edges)
        
        # 计算骨架保持度
        skeleton_preservation = np.mean(hr_skeleton == sr_skeleton)
        
        return {
            'edge_preservation': edge_preservation,
            'skeleton_preservation': skeleton_preservation,
            'hr_edges': hr_edges,
            'sr_edges': sr_edges,
            'hr_skeleton': hr_skeleton,
            'sr_skeleton': sr_skeleton
        }

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.val_lr_path = config.VAL_LR_DIR
        self.val_hr_path = config.VAL_HR_DIR
        # 初始化时加载所有可能的边缘检测方法
        self.all_edge_methods = ['canny', 'sobel', 'laplacian']
        self.device = config.BASE_TRAIN_PARAMS.get('device', 'cpu')
        self.lr_patch_size = config.BASE_TRAIN_PARAMS.get('lr_patch_size', None)
        self.upscale_factor = config.BASE_TRAIN_PARAMS.get('upscale_factor', None)
        self.image_size = config.BASE_TRAIN_PARAMS.get('image_size', None)
        
        self.metrics_calc = MetricsCalculator()
        self.edge_analyzer = EdgeSkeletonAnalyzer()
        
        # 加载验证数据集，使用所有边缘检测方法
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = SuperResolutionDataset(
            lr_path=self.val_lr_path,
            hr_path=self.val_hr_path,
            transform=transform,
            edge_methods=self.all_edge_methods,  # 使用所有边缘检测方法
            device=self.device,
            lr_patch_size=self.lr_patch_size,
            upscale_factor=self.upscale_factor,
            image_size=self.image_size
        )
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        # 记录每个边缘检测方法对应的通道索引
        self.edge_channel_indices = {}
        start_idx = 3  # RGB通道后的起始索引
        for method in self.all_edge_methods:
            self.edge_channel_indices[method] = start_idx
            start_idx += 1
    
    def load_model(self, model_path: str):
        """加载模型 (支持加载完整模型或仅状态字典)"""
        try:
            # 从模型名称中解析边缘检测方法
            model_name = os.path.basename(model_path)
            edge_methods = []
            if 'canny' in model_name.lower():
                edge_methods.append('canny')
            if 'sobel' in model_name.lower():
                edge_methods.append('sobel')
            if 'laplacian' in model_name.lower():
                edge_methods.append('laplacian')
            
            # 根据边缘检测方法确定输入通道数和通道索引
            in_channels = 3 + len(edge_methods)  # RGB + 每种边缘检测方法一个通道
            channel_indices = list(range(3))  # RGB通道
            for method in edge_methods:
                if method in self.edge_channel_indices:
                    channel_indices.append(self.edge_channel_indices[method])
            
            # 加载模型文件
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查加载的对象类型并获取状态字典
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                model = checkpoint
                model.eval()
                return model, channel_indices
            
            # 处理DataParallel的键名前缀
            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            # 创建模型实例并加载权重
            model = SimpleSRCNN(in_channels=in_channels)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)
            
            return model, channel_indices
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None, None

            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查加载的对象是否为状态字典（OrderedDict）
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # 如果是带有state_dict键的字典，提取状态字典
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                # 如果是纯状态字典
                state_dict = checkpoint
            elif hasattr(checkpoint, 'state_dict'):
                # 如果是模型对象但有state_dict方法
                state_dict = checkpoint.state_dict()
            else:
                # 如果是完整模型对象
                model = checkpoint
                model.eval()
                return model
            # 处理可能的键名不匹配问题
            if any(key.startswith('module.') for key in state_dict.keys()):
                # 如果状态字典中的键以'module.'开头，说明模型是用DataParallel训练的
                # 需要移除'module.'前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 移除'module.'前缀
                    new_state_dict[name] = v
                state_dict = new_state_dict
            # 创建SimpleSRCNN模型实例
            model = SimpleSRCNN(in_channels=in_channels)
            # 加载状态字典到模型
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device) # 将模型移动到指定设备
            
            # 更新边缘检测方法列表，使其与模型期望的输入通道数一致
            self.edge_detection_methods = edge_methods
            
            # 重新创建数据集，使用ModelValidator实例中存储的参数
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = SuperResolutionDataset(
                lr_path=self.val_lr_path,
                hr_path=self.val_hr_path,
                transform=transform,
                edge_methods=self.edge_detection_methods,
                device=self.device,
                lr_patch_size=self.lr_patch_size,
                upscale_factor=self.upscale_factor,
                image_size=self.image_size
            )
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None

            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查加载的对象是否为状态字典（OrderedDict）
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # 如果是带有state_dict键的字典，提取状态字典
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                # 如果是纯状态字典
                state_dict = checkpoint
            elif hasattr(checkpoint, 'state_dict'):
                # 如果是模型对象但有state_dict方法
                state_dict = checkpoint.state_dict()
            else:
                # 如果是完整模型对象
                model = checkpoint
                model.eval()
                return model
            
            # 处理可能的键名不匹配问题
            if any(key.startswith('module.') for key in state_dict.keys()):
                # 如果状态字典中的键以'module.'开头，说明模型是用DataParallel训练的
                # 需要移除'module.'前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 移除'module.'前缀
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            # 创建SimpleSRCNN模型实例
            model = SimpleSRCNN(in_channels=in_channels)
            
            # 加载状态字典到模型
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # 重新创建数据集，使用ModelValidator实例中存储的参数
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = SuperResolutionDataset(
                lr_path=self.val_lr_path,
                hr_path=self.val_hr_path,
                transform=transform,
                edge_methods=self.edge_detection_methods,
                device=self.device,
                lr_patch_size=self.lr_patch_size,
                upscale_factor=self.upscale_factor,
                image_size=self.image_size
            )
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
            
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    
    def validate_model(self, model_path: str, model_name: str = None, num_samples: int = 100) -> Dict:
        """验证单个模型
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
            num_samples: 验证样本数量，默认为10。TODO: 后续修改为全部样本
        """
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        print(f"Validating model: {model_name}")
        
        # 加载模型和获取所需的通道索引
        model, channel_indices = self.load_model(model_path)
        if model is None or channel_indices is None:
            return None
        
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'psnr_scores': [],
            'ssim_scores': [],
            'lpips_scores': [],
            'edge_preservation_scores': [],
            'skeleton_preservation_scores': [],
            'sample_indices': [],
            'sample_results': []  # 存储前两个样本的详细结果
        }
        
        # 限制验证样本数量
        total_samples = len(self.dataloader)
        samples_to_process = min(num_samples, total_samples)
        print(f"Processing {samples_to_process} samples out of {total_samples} total samples")
        
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                if i >= samples_to_process:
                    break
                    
                try:
                    # 从data中解包lr_batch和hr_batch
                    lr_batch, hr_batch = data
                    
                    # 选择模型所需的通道
                    lr_batch = lr_batch[:, channel_indices, :, :]
                    
                    # 将数据移动到指定设备
                    lr_batch = lr_batch.to(self.device)
                    hr_batch = hr_batch.to(self.device)
                    
                    # 模型推理
                    sr_batch = model(lr_batch)
                    
                    # 转换为numpy数组，只保留RGB通道
                    lr_img = (lr_batch[0, :3].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    hr_img = (hr_batch[0, :3].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    sr_img = (sr_batch[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # 计算基础指标
                    psnr = self.metrics_calc.psnr(hr_img, sr_img)
                    ssim = self.metrics_calc.ssim(hr_img, sr_img)
                    lpips = self.metrics_calc.lpips_simple(hr_img, sr_img)
                    
                    # 分析结构保持
                    structure_analysis = self.edge_analyzer.analyze_structure_preservation(hr_img, sr_img)
                    
                    # 记录结果
                    results['psnr_scores'].append(psnr)
                    results['ssim_scores'].append(ssim)
                    results['lpips_scores'].append(lpips)
                    results['edge_preservation_scores'].append(structure_analysis['edge_preservation'])
                    results['skeleton_preservation_scores'].append(structure_analysis['skeleton_preservation'])
                    results['sample_indices'].append(i)  # 使用循环索引作为样本索引
                    
                    # 保存前两个样本的详细结果用于展示
                    if i < 2:
                        sample_result = {
                            'index': i,  # 使用循环索引
                            'lr_img': lr_img,
                            'hr_img': hr_img,
                            'sr_img': sr_img,
                            'psnr': psnr,
                            'ssim': ssim,
                            'lpips': lpips,
                            'structure_analysis': structure_analysis
                        }
                        results['sample_results'].append(sample_result)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1} samples")
                
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
        
        # 计算平均指标
        results['avg_psnr'] = np.mean(results['psnr_scores'])
        results['avg_ssim'] = np.mean(results['ssim_scores'])
        results['avg_lpips'] = np.mean(results['lpips_scores'])
        results['avg_edge_preservation'] = np.mean(results['edge_preservation_scores'])
        results['avg_skeleton_preservation'] = np.mean(results['skeleton_preservation_scores'])
        
        print(f"Model {model_name} validation completed")
        print(f"Average PSNR: {results['avg_psnr']:.2f}")
        print(f"Average SSIM: {results['avg_ssim']:.4f}")
        print(f"Average LPIPS: {results['avg_lpips']:.4f}")
        print(f"Average Edge Preservation: {results['avg_edge_preservation']:.4f}")
        print(f"Average Skeleton Preservation: {results['avg_skeleton_preservation']:.4f}")
        
        return results
    
    def validate_multiple_models(self, model_paths: List[str], model_names: List[str] = None) -> List[Dict]:
        """验证多个模型"""
        if model_names is None:
            model_names = [os.path.basename(path) for path in model_paths]
        
        all_results = []
        for model_path, model_name in zip(model_paths, model_names):
            result = self.validate_model(model_path, model_name)
            if result is not None:
                all_results.append(result)
        
        return all_results
    
    def plot_metrics_comparison(self, results_list: List[Dict], save_path: str = None):
        """绘制指标对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        metric_names = ['PSNR', 'SSIM', 'LPIPS', 'Edge Preservation', 'Skeleton Preservation']
        metric_keys = ['psnr_scores', 'ssim_scores', 'lpips_scores', 
                      'edge_preservation_scores', 'skeleton_preservation_scores']
        
        # 绘制各项指标的样本级对比
        for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
            ax = axes[i // 3, i % 3]
            
            avg_scores = []
            for j, result in enumerate(results_list):
                scores = result[metric_key]
                sample_indices = result['sample_indices']
                
                # 绘制每个模型的得分曲线
                ax.plot(sample_indices, scores, 
                       color=colors[j], marker=markers[j % len(markers)], 
                       markersize=3, linewidth=1, alpha=0.7,
                       label=result['model_name'])
                
                avg_scores.append(np.mean(scores))
            
            # 绘制平均线
            ax.axhline(y=np.mean(avg_scores), color='red', linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'Overall Average: {np.mean(avg_scores):.3f}')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 绘制综合对比柱状图
        ax = axes[1, 2]
        model_names = [result['model_name'] for result in results_list]
        avg_metrics = []
        
        for result in results_list:
            # 计算综合得分 (归一化后的加权平均)
            psnr_norm = result['avg_psnr'] / 50.0  # 假设PSNR最大值为50
            ssim_norm = result['avg_ssim']  # SSIM已经在0-1之间
            lpips_norm = 1 - min(result['avg_lpips'], 1.0)  # LPIPS越小越好，转换为越大越好
            edge_norm = result['avg_edge_preservation']
            skeleton_norm = result['avg_skeleton_preservation']
            
            composite_score = (psnr_norm + ssim_norm + lpips_norm + 
                             edge_norm + skeleton_norm) / 5.0
            avg_metrics.append(composite_score)
        
        bars = ax.bar(model_names, avg_metrics, color=colors[:len(model_names)])
        ax.set_ylabel('Composite Score')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, score in zip(bars, avg_metrics):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_sample_results(self, results_list: List[Dict], sample_idx: int = 0, save_path: str = None):
        """可视化样本结果"""
        if sample_idx >= 2:
            print("Only first 2 samples are saved for visualization")
            return
        
        n_models = len(results_list)
        fig, axes = plt.subplots(n_models, 7, figsize=(21, 3*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results_list):
            if sample_idx >= len(result['sample_results']):
                continue
                
            sample = result['sample_results'][sample_idx]
            structure = sample['structure_analysis']
            
            # 显示原始图像
            axes[i, 0].imshow(sample['lr_img'])
            axes[i, 0].set_title(f'LR\n{result["model_name"]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(sample['hr_img'])
            axes[i, 1].set_title(f'HR (Ground Truth)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(sample['sr_img'])
            axes[i, 2].set_title(f'SR\nPSNR: {sample["psnr"]:.2f}')
            axes[i, 2].axis('off')
            
            # 显示边缘检测结果
            axes[i, 3].imshow(structure['hr_edges'], cmap='gray')
            axes[i, 3].set_title('HR Edges')
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(structure['sr_edges'], cmap='gray')
            axes[i, 4].set_title(f'SR Edges\nPreserv: {structure["edge_preservation"]:.3f}')
            axes[i, 4].axis('off')
            
            # 显示骨架提取结果
            axes[i, 5].imshow(structure['hr_skeleton'], cmap='gray')
            axes[i, 5].set_title('HR Skeleton')
            axes[i, 5].axis('off')
            
            axes[i, 6].imshow(structure['sr_skeleton'], cmap='gray')
            axes[i, 6].set_title(f'SR Skeleton\nPreserv: {structure["skeleton_preservation"]:.3f}')
            axes[i, 6].axis('off')
        
        plt.suptitle(f'Sample {sample_idx + 1} Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results_list: List[Dict], save_path: str):
        """保存验证结果"""
        # 准备要保存的数据 (去除图像数据以减小文件大小)
        save_data = []
        for result in results_list:
            save_result = result.copy()
            # 移除图像数据
            if 'sample_results' in save_result:
                for sample in save_result['sample_results']:
                    for key in ['lr_img', 'hr_img', 'sr_img']:
                        if key in sample:
                            del sample[key]
                    if 'structure_analysis' in sample:
                        for key in ['hr_edges', 'sr_edges', 'hr_skeleton', 'sr_skeleton']:
                            if key in sample['structure_analysis']:
                                del sample['structure_analysis'][key]
            save_data.append(save_result)
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"Results saved to {save_path}")

def unified_model_evaluation(config: ExperimentConfig, model_paths: List[str], model_names: List[str] = None, output_dir: str = './evaluation_results'):
    """
    统一验证多个模型在同一验证集上的性能
    
    Args:
        model_paths: 多个模型的路径列表
        val_lr_path: 验证集低分辨率图像路径
        val_hr_path: 验证集高分辨率图像路径
        model_names: 模型名称列表，如果为None则使用模型文件名
        output_dir: 输出结果保存目录
        
    Returns:
        所有模型的验证结果列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化验证器
    validator = ModelValidator(config)
    
    # 验证所有模型
    print("开始统一模型验证...")
    results = validator.validate_multiple_models(model_paths, model_names)
    
    if results:
        # 绘制指标对比图
        print("绘制指标对比图...")
        metrics_comparison_path = os.path.join(output_dir, "metrics_comparison.png")
        validator.plot_metrics_comparison(results, metrics_comparison_path)
        
        # 可视化样本结果
        print("可视化样本结果...")
        sample_0_path = os.path.join(output_dir, "sample_0_comparison.png")
        sample_1_path = os.path.join(output_dir, "sample_1_comparison.png")
        validator.visualize_sample_results(results, sample_idx=0, save_path=sample_0_path)
        validator.visualize_sample_results(results, sample_idx=1, save_path=sample_1_path)
        
        # 保存结果
        print("保存结果...")
        results_json_path = os.path.join(output_dir, "validation_results.json")
        validator.save_results(results, results_json_path)
        
        print(f"验证完成！结果已保存到 {output_dir}")
        return results
    else:
        print("未获得有效结果。")
        return None

# 使用示例
def main():
    # 加载配置
    config = ExperimentConfig()
    
    # 配置参数 (从config中获取验证集路径)
    val_lr_path = config.VAL_LR_DIR
    val_hr_path = config.VAL_HR_DIR
    
    model_paths = [
        "./results_edge_experiments/no_edge_mse_20250522-143215/no_edge_mse_best.pth",
        "./results_edge_experiments/canny_edge_combined_loss_20250523-013145/canny_edge_combined_loss_best.pth",
        "./results_edge_experiments/canny_edge_mse_20250522-185553/canny_edge_mse_best.pth",
        "./results_edge_experiments/canny_laplacian_edge_combined_loss_20250523-130522/canny_laplacian_edge_combined_loss_best.pth",
        "./results_edge_experiments/laplacian_edge_combined_loss_20250523-084104/laplacian_edge_combined_loss_best.pth",
        "./results_edge_experiments/laplacian_edge_mse_20250523-063049/laplacian_edge_mse_best.pth"
        "./results_edge_experiments/sobel_canny_edge_mse_20250522-090149/sobel_canny_edge_mse_best.pth",
        "./results_edge_experiments/sobel_edge_mse_20250522-164323/sobel_edge_mse_best.pth"
    ]
    model_names = [
        "no_edge_mse", 
        "canny_edge_combined", 
        "canny_edge_mse",
        "canny_laplacian_edge_combined_loss",
        "laplacian_edge_combined_loss",
        "laplacian_edge_mse",
        "sobel_canny_edge_mse",
        "sobel_edge_mse"
    ]
    output_dir = "./evaluation_results"
    
    # 调用统一验证函数
    unified_model_evaluation(config, model_paths, model_names, output_dir)

if __name__ == "__main__":
    main()
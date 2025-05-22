import torch
import os
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any
from models.simple_srcnn import SimpleSRCNN
from data_utils import SRDataset
from utils.evaluation_utils import EvaluationUtils

def evaluate_image(
    model_class=SimpleSRCNN,
    model_params: Optional[Dict[str, Any]] = None,
    model_path: str = 'simple_srcnn.pth',
    input_image_path: Optional[str] = None,
    hr_image_path: Optional[str] = None,
    text_description: Optional[str] = None,
    output_image_path: str = 'output_hr.png',
    device: str = 'cpu'
) -> Optional[Tuple[float, float]]:
    """评估单张图像的超分辨率效果
    
    Args:
        model_class: 模型类
        model_params: 模型参数字典
        model_path: 模型权重文件路径
        input_image_path: 输入低分辨率图像路径
        hr_image_path: 高分辨率参考图像路径（可选）
        text_description: 文本描述（可选）
        output_image_path: 输出图像保存路径
        device: 计算设备
        
    Returns:
        如果提供了hr_image_path，返回(PSNR, SSIM)元组，否则返回None
    """
    # 参数检查
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    if not input_image_path or not os.path.exists(input_image_path):
        print(f"Error: Input image file not found at {input_image_path}")
        return None

    # 初始化评估工具
    eval_utils = EvaluationUtils(device)
    
    # 加载模型
    if model_params is None:
        model_params = {}
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 加载和预处理输入图像
    input_tensor = eval_utils.load_and_preprocess_image(input_image_path)
    if input_tensor is None:
        return None

    # 执行超分辨率
    with torch.no_grad():
        output_tensor = model(input_tensor, text_input=[text_description] if text_description else None)

    # 保存输出图像
    if not eval_utils.save_image(output_tensor, output_image_path):
        print(f"Warning: Failed to save output image to {output_image_path}")

    # 如果提供了HR图像，计算评估指标
    if hr_image_path and os.path.exists(hr_image_path):
        hr_tensor = eval_utils.load_and_preprocess_image(hr_image_path)
        if hr_tensor is None:
            return None
            
        try:
            psnr, ssim = eval_utils.calculate_metrics(output_tensor, hr_tensor)
            print(f"PSNR: {psnr:.4f}")
            print(f"SSIM: {ssim:.4f}")
            return psnr, ssim
        except ValueError as e:
            print(f"Error calculating metrics: {e}")
            return None
    return None

def evaluate_dataset_subset(
    model_class=SimpleSRCNN,
    model_params: Optional[Dict[str, Any]] = None,
    model_path: str = 'simple_srcnn.pth',
    val_lr_data_dir: str = './data/split_sample/val/lr',
    val_hr_data_dir: str = './data/split_sample/val/hr',
    edge_detection_methods: Optional[list] = None,
    num_samples: int = 10,
    output_dir: str = './evaluation_results',
    device: str = 'cpu',
    upscale_factor: int = 2
) -> Optional[Tuple[float, float]]:
    """评估数据集子集的超分辨率效果
    
    Args:
        model_class: 模型类
        model_params: 模型参数字典
        model_path: 模型权重文件路径
        val_lr_data_dir: 验证集低分辨率图像目录
        val_hr_data_dir: 验证集高分辨率图像目录
        edge_detection_methods: 边缘检测方法列表
        num_samples: 评估样本数量
        output_dir: 输出目录
        device: 计算设备
        upscale_factor: 放大倍数
        
    Returns:
        返回平均PSNR和SSIM值的元组，如果评估失败则返回None
    """
    # 参数检查
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    if not os.path.exists(val_lr_data_dir) or not os.path.exists(val_hr_data_dir):
        print(f"Error: Validation data directories not found at {val_lr_data_dir} or {val_hr_data_dir}")
        return None

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化评估工具
    eval_utils = EvaluationUtils(device)

    # 准备模型参数
    if model_params is None:
        model_params = {}
    num_edge_channels = len(edge_detection_methods) if edge_detection_methods else 0
    in_channels_for_model = 3 + num_edge_channels

    # 更新模型参数
    if hasattr(model_class, '__init__'):
        if 'in_channels' in model_class.__init__.__code__.co_varnames:
            model_params['in_channels'] = in_channels_for_model
        if 'upscale_factor' in model_class.__init__.__code__.co_varnames:
            model_params['upscale_factor'] = upscale_factor

    # 加载模型
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 加载数据集
    dataset = SRDataset(
        lr_dir=None,
        hr_dir=None,
        text_descriptions=None,
        transform=eval_utils.transform,
        mode='eval',
        edge_methods=edge_detection_methods,
        device=device,
        upscale_factor=upscale_factor,
        val_lr_dir=val_lr_data_dir,
        val_hr_dir=val_hr_data_dir
    )

    if len(dataset) == 0:
        print("No images found in the dataset.")
        return None

    # 随机选择样本
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Evaluating {len(sample_indices)} random samples.")

    # 评估样本
    metrics_list = []
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            lr_image_tensor, hr_image_tensor = dataset[idx]
            lr_image_tensor = lr_image_tensor.unsqueeze(0).to(device)
            hr_image_tensor = hr_image_tensor.unsqueeze(0).to(device)

            # 执行超分辨率
            if hasattr(model, 'forward') and 'text_input' in model.forward.__code__.co_varnames:
                sr_output_tensor = model(lr_image_tensor, text_input=None)
            else:
                sr_output_tensor = model(lr_image_tensor)

            # 检查输出尺寸
            if sr_output_tensor.shape[-2:] != hr_image_tensor.shape[-2:]:
                print(f"Warning: Sample {i} SR output size {sr_output_tensor.shape[-2:]} and HR target size {hr_image_tensor.shape[-2:]} mismatch. Skipping metrics for this sample.")
                continue

            try:
                psnr, ssim = eval_utils.calculate_metrics(sr_output_tensor, hr_image_tensor)
                metrics_list.append((psnr, ssim))
            except ValueError as e:
                print(f"Error calculating metrics for sample {i}: {e}")

    # 处理评估结果
    results = eval_utils.process_batch_metrics(metrics_list)
    if results['avg_psnr'] is not None:
        print(f"\nAverage PSNR over {len(metrics_list)} samples: {results['avg_psnr']:.4f}")
        print(f"Average SSIM over {len(metrics_list)} samples: {results['avg_ssim']:.4f}")
        return results['avg_psnr'], results['avg_ssim']
    return None

if __name__ == '__main__':
    # 示例用法
    example_lr_dir = './data_example_eval/lr'
    os.makedirs(example_lr_dir, exist_ok=True)
    dummy_lr_path = os.path.join(example_lr_dir, 'eval_lr_dummy.png')
    
    # 创建示例图像
    dummy_hr_size = 64
    dummy_lr_size = 32
    dummy_hr_img_np = np.random.randint(0, 256, (dummy_hr_size, dummy_hr_size, 3), dtype=np.uint8)
    
    # 保存示例图像用于评估
    eval_utils = EvaluationUtils()
    dummy_tensor = torch.from_numpy(dummy_hr_img_np.transpose(2, 0, 1)).float() / 255.0
    eval_utils.save_image(dummy_tensor.unsqueeze(0), dummy_lr_path)
    
    # 评估示例
    print("\n--- Evaluating with default SimpleSRCNN ---")
    evaluate_image(
        model_path='simple_srcnn.pth',
        input_image_path=dummy_lr_path,
        text_description="magnification: 2x, content: test image",
        output_image_path='./dummy_output_hr_eval.png'
    )


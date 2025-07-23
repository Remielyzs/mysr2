#!/usr/bin/env python3
"""
改进的模型测试脚本

主要功能：
- 加载训练好的模型进行推理测试
- 支持多种输入格式和批量处理
- 提供详细的性能分析和质量评估
- 集成配置管理和日志系统
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
except ImportError as e:
    print(f"导入PyTorch失败: {e}")
    sys.exit(1)

from models.simple_unet import SimpleUNet
from models.noise_scheduler import NoiseScheduler
from models.lr_encoder import create_lr_encoder
from utils.config_manager import ConfigManager
from utils.logger import setup_logging

class ModelTester:
    """模型测试器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.logger = setup_logging(
            log_dir=config.get('log_dir', 'logs'),
            console_output=config.get('console_logging', True)
        )
        
        # 初始化模型组件
        self.model = None
        self.noise_scheduler = None
        self.lr_encoder = None
        
        # 性能统计
        self.inference_times = []
        self.memory_usage = []
        
    def load_model(self, checkpoint_path):
        """加载训练好的模型"""
        try:
            self.logger.info(f"从检查点加载模型: {checkpoint_path}")
            
            # 检查检查点文件
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.logger.info(f"检查点信息: epoch={checkpoint.get('epoch', 'unknown')}, "
                           f"loss={checkpoint.get('loss', 'unknown')}")
            
            # 创建模型
            model_config = {
                'in_channels': self.config.get('model_in_channels', 35),
                'out_channels': self.config.get('model_out_channels', 3),
                'channels': self.config.get('unet_channels', [32, 64, 128]),
                'attention_resolutions': self.config.get('attention_resolutions', [16, 8]),
                'num_res_blocks': self.config.get('num_res_blocks', 2),
                'dropout': self.config.get('dropout', 0.0)
            }
            
            self.model = SimpleUNet(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 创建噪声调度器
            self.noise_scheduler = NoiseScheduler(
                num_timesteps=self.config.get('num_timesteps', 1000),
                beta_start=self.config.get('beta_start', 0.001),
                beta_end=self.config.get('beta_end', 0.02),
                schedule=self.config.get('noise_schedule', 'linear')
            )
            
            # 创建低分辨率编码器
            self.lr_encoder = create_lr_encoder(
                upscale_factor=self.config.get('upscale_factor', 4)
            ).to(self.device)
            self.lr_encoder.eval()
            
            self.logger.info("模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def preprocess_image(self, image_path, target_size=None):
        """预处理输入图像"""
        try:
            # 加载图像
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path
            
            # 调整大小
            if target_size:
                image = image.resize(target_size, Image.LANCZOS)
            
            # 转换为张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            tensor = transform(image).unsqueeze(0)  # 添加批次维度
            return tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return None
    
    def postprocess_output(self, tensor):
        """后处理模型输出"""
        try:
            # 反归一化
            tensor = (tensor + 1.0) / 2.0
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
            # 转换为PIL图像
            tensor = tensor.squeeze(0).cpu()
            transform = transforms.ToPILImage()
            image = transform(tensor)
            
            return image
            
        except Exception as e:
            self.logger.error(f"输出后处理失败: {e}")
            return None
    
    def inference_single(self, lr_image_path, num_inference_steps=50):
        """单张图像推理"""
        if self.model is None:
            self.logger.error("模型未加载")
            return None
        
        try:
            start_time = time.time()
            
            # 预处理低分辨率图像
            lr_image = self.preprocess_image(lr_image_path)
            if lr_image is None:
                return None
            
            batch_size, channels, height, width = lr_image.shape
            hr_height, hr_width = height * self.config.get('upscale_factor', 4), width * self.config.get('upscale_factor', 4)
            
            # 编码低分辨率特征
            with torch.no_grad():
                lr_features = self.lr_encoder(lr_image)
            
            # 初始化噪声
            noise = torch.randn(batch_size, 3, hr_height, hr_width, device=self.device)
            
            # 扩散去噪过程
            timesteps = torch.linspace(
                self.noise_scheduler.num_timesteps - 1, 0, 
                num_inference_steps, dtype=torch.long, device=self.device
            )
            
            current_sample = noise
            
            for i, t in enumerate(timesteps):
                # 准备模型输入
                model_input = torch.cat([current_sample, lr_features], dim=1)
                timestep = t.unsqueeze(0).expand(batch_size)
                
                # 模型预测
                with torch.no_grad():
                    noise_pred = self.model(model_input, timestep)
                
                # 更新样本
                if i < len(timesteps) - 1:
                    alpha_t = self.noise_scheduler.alphas_cumprod[t]
                    alpha_t_prev = self.noise_scheduler.alphas_cumprod[timesteps[i + 1]]
                    
                    # 简化的DDIM更新
                    pred_x0 = (current_sample - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    current_sample = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise_pred
                else:
                    # 最后一步
                    current_sample = noise_pred
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 记录内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                self.memory_usage.append(memory_used)
                torch.cuda.reset_peak_memory_stats()
            
            self.logger.info(f"推理完成，耗时: {inference_time:.2f}秒")
            
            # 后处理输出
            output_image = self.postprocess_output(current_sample)
            return output_image
            
        except Exception as e:
            self.logger.error(f"推理过程失败: {e}")
            return None
    
    def batch_inference(self, input_dir, output_dir, num_inference_steps=50):
        """批量推理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            self.logger.warning(f"在 {input_dir} 中未找到图像文件")
            return
        
        self.logger.info(f"开始批量推理，共 {len(image_files)} 张图像")
        
        success_count = 0
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"处理 {i}/{len(image_files)}: {image_file.name}")
            
            # 推理
            result = self.inference_single(image_file, num_inference_steps)
            
            if result is not None:
                # 保存结果
                output_file = output_path / f"sr_{image_file.stem}.png"
                result.save(output_file)
                success_count += 1
                self.logger.info(f"结果已保存: {output_file}")
            else:
                self.logger.error(f"处理失败: {image_file.name}")
        
        self.logger.info(f"批量推理完成，成功处理 {success_count}/{len(image_files)} 张图像")
    
    def print_performance_stats(self):
        """打印性能统计"""
        if not self.inference_times:
            self.logger.info("无性能数据")
            return
        
        avg_time = np.mean(self.inference_times)
        min_time = np.min(self.inference_times)
        max_time = np.max(self.inference_times)
        
        stats = f"""
性能统计:
- 平均推理时间: {avg_time:.2f}秒
- 最快推理时间: {min_time:.2f}秒
- 最慢推理时间: {max_time:.2f}秒
- 总推理次数: {len(self.inference_times)}
"""
        
        if self.memory_usage:
            avg_memory = np.mean(self.memory_usage)
            max_memory = np.max(self.memory_usage)
            stats += f"- 平均内存使用: {avg_memory:.1f}MB\n- 峰值内存使用: {max_memory:.1f}MB"
        
        print(stats)
        self.logger.info(stats.replace('\n', ' | '))

def create_test_config():
    """创建测试配置"""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_dir': 'logs',
        'console_logging': True,
        
        # 模型配置
        'model_in_channels': 35,
        'model_out_channels': 3,
        'unet_channels': [32, 64, 128],
        'attention_resolutions': [16, 8],
        'num_res_blocks': 2,
        'dropout': 0.0,
        
        # 扩散配置
        'num_timesteps': 1000,
        'beta_start': 0.001,
        'beta_end': 0.02,
        'noise_schedule': 'linear',
        'upscale_factor': 4,
    }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='改进的模型测试脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='模型检查点路径'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='输入图像路径或目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='test_outputs',
        help='输出目录'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='推理步数'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='推理设备'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='运行性能基准测试'
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    try:
        args = parse_arguments()
        
        # 创建配置
        if args.config and Path(args.config).exists():
            config_manager = ConfigManager.load_from_file(args.config)
            config = config_manager.to_dict()
        else:
            config = create_test_config()
        
        # 设备配置
        if args.device != 'auto':
            config['device'] = args.device
        
        # 创建测试器
        tester = ModelTester(config)
        
        # 加载模型
        if not tester.load_model(args.checkpoint):
            sys.exit(1)
        
        # 执行测试
        if args.input:
            input_path = Path(args.input)
            
            if input_path.is_file():
                # 单文件推理
                tester.logger.info(f"单文件推理: {input_path}")
                result = tester.inference_single(input_path, args.steps)
                
                if result:
                    output_path = Path(args.output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_path / f"sr_{input_path.stem}.png"
                    result.save(output_file)
                    print(f"结果已保存: {output_file}")
                
            elif input_path.is_dir():
                # 批量推理
                tester.batch_inference(input_path, args.output, args.steps)
            
            else:
                print(f"输入路径不存在: {input_path}")
                sys.exit(1)
        
        # 性能基准测试
        if args.benchmark:
            tester.logger.info("运行性能基准测试")
            # 创建测试图像
            test_image = torch.randn(1, 3, 64, 64) * 0.5 + 0.5
            test_image = transforms.ToPILImage()(test_image.squeeze(0))
            
            # 多次推理测试
            for i in range(5):
                tester.logger.info(f"基准测试 {i+1}/5")
                tester.inference_single(test_image, args.steps)
        
        # 打印性能统计
        tester.print_performance_stats()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
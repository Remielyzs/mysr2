import torch
import torch.nn as nn
import math

class NoiseScheduler:
    """扩散模型的噪声调度器
    
    实现了不同的噪声调度策略，用于扩散模型的前向和反向过程。
    """
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
        """初始化噪声调度器
        
        Args:
            num_timesteps: 扩散步数
            beta_start: 起始beta值
            beta_end: 结束beta值
            schedule: 调度策略 ('linear' 或 'cosine')
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # 设置噪声调度
        self._setup_schedule()
    
    def _setup_schedule(self):
        """设置噪声调度参数"""
        if self.schedule == 'linear':
            # 线性噪声调度
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.schedule == 'cosine':
            # 余弦噪声调度
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"不支持的噪声调度类型: {self.schedule}")
        
        # 计算扩散过程中的关键参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于反向过程的参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_start, timesteps, noise=None):
        """向原始图像添加噪声
        
        Args:
            x_start: 原始图像 (B, C, H, W)
            timesteps: 时间步长 (B,)
            noise: 可选的噪声张量，如果为None则随机生成
            
        Returns:
            noisy_images: 添加噪声后的图像
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 将timesteps移动到调度器参数的设备并确保是long类型
        timesteps = timesteps.long().to(self.sqrt_alphas_cumprod.device)
        
        # 获取对应时间步的参数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        # 将参数移动到正确的设备
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(x_start.device)
        
        # 确保noise在正确的设备上
        noise = noise.to(x_start.device)
        
        # 添加噪声: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        noisy_images = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_images, noise
    
    def sample_timesteps(self, batch_size, device):
        """随机采样时间步长
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            timesteps: 随机时间步长 (batch_size,)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def get_variance(self, timestep):
        """获取指定时间步的方差
        
        Args:
            timestep: 时间步长
            
        Returns:
            variance: 方差值
        """
        if timestep == 0:
            return 0
        
        variance = self.posterior_variance[timestep]
        return variance
    
    def to(self, device):
        """将调度器参数移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
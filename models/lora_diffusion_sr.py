import torch
import torch.nn as nn
from models.diffusion_sr import DiffusionSR

class LoRALayer(nn.Module):
    """LoRA层实现"""
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = 1.0
        
        # 初始化为零以确保训练开始时不影响原始权重
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=1/rank)

    def forward(self, x):
        return self.up(self.down(x)) * self.scale

class LoRADiffusionSR(DiffusionSR):
    """使用LoRA技术的Stable Diffusion超分辨率模型"""
    def __init__(self, base_model_params, lora_rank=4, lora_alpha=1.0, edge_channels=0):
        super().__init__(**base_model_params)
        
        # 添加边缘信息通道
        if edge_channels > 0:
            self.input_conv = nn.Conv2d(
                self.input_channels + edge_channels,
                self.unet_channels[0],
                kernel_size=3,
                padding=1
            )
        
        # 为UNet中的关键层添加LoRA
        self.lora_layers = nn.ModuleDict()
        self.lora_alpha = lora_alpha
        
        # 为每个UNet block添加LoRA层
        for i, channels in enumerate(self.unet_channels):
            # 下采样路径
            self.lora_layers[f'down_{i}'] = LoRALayer(
                channels,
                channels,
                rank=lora_rank
            )
            
            # 上采样路径
            if i > 0:  # 跳过第一层，因为它没有上采样
                self.lora_layers[f'up_{i}'] = LoRALayer(
                    channels * 2,  # 考虑skip connection
                    channels,
                    rank=lora_rank
                )
        
        # 设置LoRA缩放因子
        self._set_lora_scale(self.lora_alpha)

    def _set_lora_scale(self, scale):
        """设置所有LoRA层的缩放因子"""
        for lora in self.lora_layers.values():
            lora.scale = scale

    def forward(self, x, edge_features=None):
        """前向传播
        Args:
            x: 输入图像张量
            edge_features: 边缘特征张量列表
        Returns:
            超分辨率输出
        """
        # 合并边缘特征
        if edge_features is not None:
            x = torch.cat([x] + edge_features, dim=1)
        
        # UNet编码器路径
        features = []
        current = self.input_conv(x)
        
        for i in range(len(self.unet_channels)):
            # 应用LoRA
            lora_out = self.lora_layers[f'down_{i}'](current.permute(0,2,3,1))
            current = current + lora_out.permute(0,3,1,2)
            
            # 下采样
            current = self.down_blocks[i](current)
            features.append(current)
            if i < len(self.unet_channels) - 1:
                current = self.down_samplers[i](current)
        
        # UNet解码器路径
        for i in range(len(self.unet_channels) - 1, -1, -1):
            if i < len(self.unet_channels) - 1:
                # 上采样和特征融合
                current = self.up_samplers[i](current)
                current = torch.cat([current, features[i]], dim=1)
                
                # 应用LoRA到上采样路径
                lora_out = self.lora_layers[f'up_{i}'](current.permute(0,2,3,1))
                current = current + lora_out.permute(0,3,1,2)
            
            current = self.up_blocks[i](current)
        
        # 输出层
        return self.output_conv(current)

    def freeze_base_model(self):
        """冻结基础模型参数，只训练LoRA层"""
        for param in self.parameters():
            param.requires_grad = False
        
        for lora in self.lora_layers.values():
            for param in lora.parameters():
                param.requires_grad = True

    def unfreeze_base_model(self):
        """解冻所有模型参数"""
        for param in self.parameters():
            param.requires_grad = True
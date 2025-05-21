import torch
import torch.nn as nn

class TextGuidedSRModel(nn.Module):
    def __init__(self, upscale_factor=4, text_embedding_dim=512, image_feature_dim=64):
        super(TextGuidedSRModel, self).__init__()
        self.upscale_factor = upscale_factor
        self.text_embedding_dim = text_embedding_dim
        self.image_feature_dim = image_feature_dim

        # 文本编码器 (简化示例，实际中可能使用预训练的BERT, CLIP等)
        # 这里假设文本已经被编码为 text_embedding_dim 维度的向量
        # self.text_encoder = nn.Linear(vocab_size, text_embedding_dim) # 假设有词汇表

        # 图像特征提取器 (与BasicSRModel类似，但可能通道数不同)
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(3, image_feature_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_feature_dim, image_feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # 融合文本和图像特征的模块
        # 简单示例：将文本特征扩展并与图像特征拼接或相加
        # 假设文本特征作用于每个像素，或者通过某种注意力机制融合
        # 这里我们假设文本特征被处理成与图像特征维度兼容的形式
        self.fusion_conv = nn.Conv2d(image_feature_dim + text_embedding_dim, image_feature_dim, kernel_size=1)
        # 或者使用更复杂的融合策略，如 FiLM (Feature-wise Linear Modulation)
        # self.film_generator = nn.Linear(text_embedding_dim, image_feature_dim * 2) # 生成 gamma 和 beta

        # 上采样模块 (与BasicSRModel类似)
        self.upsample_module = nn.Sequential(
            nn.Conv2d(image_feature_dim, image_feature_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_feature_dim * 2, (upscale_factor ** 2) * 3, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        self._initialize_weights()

    def forward(self, lr_image, text_embedding):
        """
        Args:
            lr_image (torch.Tensor): 低分辨率图像 (B, 3, H_lr, W_lr)
            text_embedding (torch.Tensor): 文本嵌入 (B, text_embedding_dim)
        Returns:
            torch.Tensor: 高分辨率图像 (B, 3, H_hr, W_hr)
        """
        img_feat = self.image_feature_extractor(lr_image) # (B, image_feature_dim, H_lr, W_lr)

        # 融合文本特征
        # 简单拼接示例：
        # 1. 将文本特征扩展到与图像特征图相同的空间维度
        text_feat_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1) # (B, text_embedding_dim, 1, 1)
        text_feat_expanded = text_feat_expanded.expand(-1, -1, img_feat.size(2), img_feat.size(3)) # (B, text_embedding_dim, H_lr, W_lr)
        
        # 2. 拼接特征
        fused_feat = torch.cat([img_feat, text_feat_expanded], dim=1) # (B, image_feature_dim + text_embedding_dim, H_lr, W_lr)
        fused_feat = self.fusion_conv(fused_feat) # (B, image_feature_dim, H_lr, W_lr)

        # FiLM 示例 (如果使用):
        # gamma_beta = self.film_generator(text_embedding) # (B, image_feature_dim * 2)
        # gamma = gamma_beta[:, :self.image_feature_dim].unsqueeze(-1).unsqueeze(-1) # (B, image_feature_dim, 1, 1)
        # beta = gamma_beta[:, self.image_feature_dim:].unsqueeze(-1).unsqueeze(-1)  # (B, image_feature_dim, 1, 1)
        # fused_feat = gamma * img_feat + beta

        sr_image = self.upsample_module(fused_feat)
        return torch.tanh(sr_image) # 输出范围 [-1, 1] 或根据需要调整

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # 定义参数
    upscale = 4
    text_dim = 256
    img_dim = 64
    batch_size = 2
    lr_h, lr_w = 32, 32

    # 创建模型
    model = TextGuidedSRModel(upscale_factor=upscale, text_embedding_dim=text_dim, image_feature_dim=img_dim)
    print(model)

    # 创建虚拟输入
    dummy_lr_image = torch.randn(batch_size, 3, lr_h, lr_w)
    dummy_text_embedding = torch.randn(batch_size, text_dim)

    # 前向传播
    output_sr_image = model(dummy_lr_image, dummy_text_embedding)

    print(f"\nInput LR image shape: {dummy_lr_image.shape}")
    print(f"Input text embedding shape: {dummy_text_embedding.shape}")
    print(f"Output SR image shape: {output_sr_image.shape}")

    expected_hr_h, expected_hr_w = lr_h * upscale, lr_w * upscale
    assert output_sr_image.shape == (batch_size, 3, expected_hr_h, expected_hr_w), \
        f"Output shape mismatch. Expected: {(batch_size, 3, expected_hr_h, expected_hr_w)}, Got: {output_sr_image.shape}"
    
    print("\nText-guided SR model created and tested successfully.")
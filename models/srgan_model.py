import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, upscale_factor=4):
        super(Generator, self).__init__()
        self.upscale_factor = upscale_factor

        # 输入层
        self.conv_in = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

        # 残差块 (示例，可以根据需要增加更多)
        residual_blocks = []
        for _ in range(16):
            residual_blocks.append(ResidualBlock(64))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # 残差块后的卷积层
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        # 上采样块
        upsample_blocks = []
        for _ in range(int(upscale_factor / 2)): # 假设 upscale_factor 是 2 的倍数
            upsample_blocks.append(UpsampleBlock(64, 256))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        # 输出层
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

        self._initialize_weights()

    def forward(self, x):
        x_in = self.relu(self.conv_in(x))
        x_res = self.residual_blocks(x_in)
        x_mid = self.bn_mid(self.conv_mid(x_res))
        x = x_in + x_mid # 主干网络的跳跃连接
        x = self.upsample_blocks(x)
        x = self.conv_out(x)
        return torch.tanh(x) # 输出范围 [-1, 1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        # out_channels 应该是 in_channels * (scale_factor ** 2)
        # 这里 scale_factor 默认为 2，所以 out_channels = in_channels * 4
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2) # 放大两倍
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 输入图像尺寸 (batch_size, 3, H, W)
        # 输出一个概率值，表示图像是真实图像的概率
        layers = []
        # C64
        layers.append(self._make_layer(3, 64, batch_norm=False))
        # C128
        layers.append(self._make_layer(64, 128))
        # C256
        layers.append(self._make_layer(128, 256))
        # C512
        layers.append(self._make_layer(256, 512))

        self.features = nn.Sequential(*layers)

        # 全连接层
        # 需要根据输入图像大小动态计算 in_features
        # 假设输入图像经过卷积后尺寸变为 H/16, W/16
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(512, 1024, kernel_size=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(1024, 1, kernel_size=1)
        # )
        # 或者使用更传统的全连接层
        self.dense_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)), # 假设最终特征图大小为 6x6
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
            # nn.Sigmoid() # Sigmoid 通常在损失函数中处理 (BCEWithLogitsLoss)
        )
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        # 判别器通常会逐步减小特征图尺寸，所以 stride=2 的卷积层可以加在这里
        # 例如，每隔一个或两个 _make_layer 块，可以加一个 stride=2 的卷积
        # 这里为了简化，保持 stride=1，尺寸变化依赖于 AdaptiveAvgPool2d
        # 或者在每个 block 后加一个 stride=2 的卷积
        # layer.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
        # if batch_norm:
        #     layer.append(nn.BatchNorm2d(out_channels))
        # layer.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = self.dense_layers(x)
        return x # 返回 logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    # 测试生成器
    print("--- Testing Generator ---")
    generator = Generator(upscale_factor=4)
    # print(generator)
    dummy_lr = torch.randn(1, 3, 32, 32) # 低分辨率输入
    generated_hr = generator(dummy_lr)
    print(f"Generator input shape: {dummy_lr.shape}")
    print(f"Generator output shape: {generated_hr.shape}")

    # 测试判别器
    print("\n--- Testing Discriminator ---")
    discriminator = Discriminator()
    # print(discriminator)
    dummy_hr_real = torch.randn(1, 3, 128, 128) # 真实高分辨率图像 (32*4 = 128)
    dummy_hr_fake = generated_hr # 生成的假高分辨率图像

    prob_real = discriminator(dummy_hr_real)
    prob_fake = discriminator(dummy_hr_fake)

    print(f"Discriminator input shape (real): {dummy_hr_real.shape}, output: {prob_real.shape}, value: {prob_real.item()}")
    print(f"Discriminator input shape (fake): {dummy_hr_fake.shape}, output: {prob_fake.shape}, value: {prob_fake.item()}")

    # 确保判别器输入尺寸与生成器输出尺寸匹配
    assert dummy_hr_real.shape == generated_hr.shape, \
        f"Shape mismatch: Real HR {dummy_hr_real.shape} vs Generated HR {generated_hr.shape}"

    print("\nSRGAN models created and tested successfully.")
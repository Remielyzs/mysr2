import torch
import torch.nn as nn

class BasicSRModel(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=2):
        super(BasicSRModel, self).__init__()
        self.in_channels = in_channels
        self.upscale_factor = upscale_factor

        # 示例：一个简单的卷积层用于上采样
        # 实际模型会更复杂
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, (upscale_factor ** 2) * 3, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        # 输入 x: (batch_size, channels, height, width)
        # 输出: (batch_size, channels, height * upscale_factor, width * upscale_factor)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # 测试模型
    # 测试默认输入通道 (3)
    model_default = BasicSRModel(upscale_factor=4)
    print("Model with default in_channels=3:")
    print(model_default)
    dummy_input_default = torch.randn(1, 3, 64, 64)
    output_default = model_default(dummy_input_default)
    print(f"Input shape (default): {dummy_input_default.shape}")
    print(f"Output shape (default): {output_default.shape}")

    # 测试自定义输入通道 (例如 6，RGB + 3 边缘图)
    model_custom_channels = BasicSRModel(in_channels=6, upscale_factor=2)
    print("\nModel with in_channels=6:")
    print(model_custom_channels)
    dummy_input_custom = torch.randn(1, 6, 32, 32) # 假设输入尺寸也不同
    output_custom = model_custom_channels(dummy_input_custom)
    print(f"Input shape (custom): {dummy_input_custom.shape}")
    print(f"Output shape (custom): {output_custom.shape}")

    # 保持原始测试用例的输出格式，但使用默认模型进行
    dummy_input = dummy_input_default
    output = output_default
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
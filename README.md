# 超分辨率模型项目 (mysr2)

本项目旨在实现一个灵活且模块化的图像超分辨率深度学习框架。

## 项目目标

-   构建一个易于扩展和维护的超分辨率模型训练和评估流程。
-   支持多种图像输入格式。
-   允许模型接收图像数据和相关的文本描述作为输入。
-   实现数据处理、模型定义、训练和评估模块的解耦，方便进行模型比较和实验。

## 功能架构

项目主要包含以下几个核心模块：

1.  **数据处理 (`data_utils.py`)**: 
    *   负责读取和预处理图像数据。
    *   支持多种图像格式，如 PNG, TIFF, NPZ。
    *   为训练和评估提供不同的数据加载流程。
    *   能够处理与图像相关的文本描述信息。
2.  **模型定义 (`models/`)**: 
    *   包含所有模型定义的Python文件，例如 `models/simple_srcnn.py`。 
    *   定义超分辨率模型的网络结构。
    *   模型能够接收低分辨率 (LR) 图像和可选的文本描述作为输入。
    *   在训练时，使用高分辨率 (HR) 图像作为真值 (ground truth)。
    *   在评估时，根据 LR 图像和文本描述生成超分辨率 (SR) 图像。
3.  **模型训练 (`train.py`)**: 
    *   实现模型训练的核心逻辑。
    *   与具体的模型定义解耦，可以方便地切换和训练不同的模型架构。
    *   支持配置训练参数，如周期数、批大小、学习率等。
4.  **模型评估 (`evaluate.py`)**: 
    *   实现模型性能评估的逻辑。
    *   与具体的模型定义解耦，可以使用相同的评估流程测试不同的模型。
    *   根据输入的 LR 图像（和文本描述）生成 SR 图像并保存。

## 技术栈

-   Python
-   PyTorch
-   Pillow (PIL)
-   NumPy

## 目录结构

```
mysr2/
├── data/                     # 存放训练和测试数据 (示例)
│   ├── hr/                   # 高分辨率图像
│   └── lr/                   # 低分辨率图像
├── data_utils.py             # 数据加载和预处理工具
├── models/                   # 存放模型定义文件
│   ├── __init__.py
│   └── simple_srcnn.py       # SimpleSRCNN模型示例
├── model.py                  # (此文件已移除，模型现在位于models/目录下)
├── train.py                  # 模型训练脚本
├── evaluate.py               # 模型评估脚本
├── requirements.txt          # Python依赖包
├── simple_srcnn.pth          # 预训练模型权重 (示例)
└── README.md                 # 项目说明文档
```

## 使用说明

### 1. 环境准备

确保已安装 Python 和 PyTorch。然后安装项目依赖：
```bash
pip install -r requirements.txt
```

### 2. 数据准备

-   将高分辨率 (HR) 图像放置在 `data/hr/` 目录下。
-   将对应的低分辨率 (LR) 图像放置在 `data/lr/` 目录下。
-   `data_utils.py` 中的 `SRDataset` 类支持 `.png`, `.tiff`, `.tif`, `.npz` 格式的图像。
-   如果使用 `.npz` 文件，请确保图像数据存储在名为 `image`, `lr`, 或 `hr` 的键下，或者作为文件中的第一个数组。
-   `generate_synthetic_data` 函数可以用于生成合成数据以进行快速测试。

### 3. 模型训练

运行训练脚本：
```bash
python train.py
```

可以修改 `train.py` 中的 `train_model` 函数参数或 `if __name__ == '__main__':` 部分来调整训练配置，例如：
-   选择不同的模型 (`model_class`)
-   设置模型参数 (`model_params`)
-   指定数据目录 (`data_dir`)
-   调整训练周期 (`epochs`)、批大小 (`batch_size`)、学习率 (`learning_rate`)
-   启用文本描述 (`use_text_descriptions=True`)

训练过程中，模型检查点（包括模型权重、优化器状态、当前周期和损失）将保存在 `checkpoints/` 目录下（可配置）。例如，`checkpoints/model_name_epoch_X.pth` 和 `checkpoints/model_name_best.pth`。训练完成后，最终模型也会保存在此目录。

-   启用文本描述 (`use_text_descriptions=True`)
-   指定损失函数 (`criterion`)
-   配置检查点目录 (`checkpoint_dir`)
-   从特定检查点恢复训练 (`resume_checkpoint`)
-   为保存的模型和检查点指定名称 (`model_name`)
-   指定边缘检测方法 (`edge_detection_methods`)，例如 `['sobel']` 或 `['canny']`

训练过程中，训练损失和验证损失都会被记录，并在训练结束时绘制损失曲线图保存在结果目录中。

#### 示例：使用自定义损失函数、边缘检测和检查点

```python
# 在 train.py 的 if __name__ == '__main__': 部分
from losses import L1Loss, CombinedLoss, EdgeLoss # 假设这些在 losses.py 中定义
import torch.nn as nn

# 1. 使用L1损失函数进行训练，并将模型命名为 'srcnn_l1'
# train_model(
#     model_class=SimpleSRCNN, 
#     epochs=5, 
#     batch_size=4, 
#     criterion=L1Loss(), 
#     model_name='srcnn_l1',
#     checkpoint_dir='my_custom_checkpoints'
# )

# 2. 假设第一次训练中断，现在从 'my_custom_checkpoints/srcnn_l1_epoch_3.pth' 恢复训练
# train_model(
#     model_class=SimpleSRCNN, 
#     epochs=10, # 总共希望训练到10个epochs
#     batch_size=4, 
#     criterion=L1Loss(), 
#     model_name='srcnn_l1', 
#     resume_checkpoint='my_custom_checkpoints/srcnn_l1_epoch_3.pth',
#     checkpoint_dir='my_custom_checkpoints'
# )

# 3. 使用组合损失函数 (MSE + 边缘损失) 并指定边缘检测方法
# combined_loss = CombinedLoss([
#     (nn.MSELoss(), 1.0), 
#     (EdgeLoss(edge_detector_type='sobel'), 0.1) # 边缘损失权重为0.1
# ])
# train_model(
#     model_class=SimpleSRCNN, 
#     epochs=5, 
#     batch_size=2,
#     criterion=combined_loss,
#     model_name='srcnn_mse_edge',
#     checkpoint_dir='checkpoints_mse_edge',
#     edge_detection_methods=['sobel'] # 指定使用Sobel边缘
# )
```

训练过程中，训练损失和验证损失都会被记录，并在训练结束时绘制损失曲线图保存在结果目录中。

### 4. 模型评估

运行评估脚本：
```bash
python evaluate.py
```

`evaluate.py` 脚本提供了 `evaluate_image` 函数，用于对单张图片进行超分辨率处理并保存结果。如果提供了高分辨率（HR）图像路径，它还会计算 PSNR 和 SSIM 指标。

此外，新增了 `evaluate_dataset_subset` 函数，用于对验证数据集中的随机子集进行评估，并计算 PSNR、SSIM 以及 FRC（频率相关性）。FRC 曲线图将为每个样本生成并保存。

#### 示例：评估单张图片

```python
# 在 evaluate.py 的 if __name__ == '__main__': 部分
evaluate_image(
    model_path='simple_srcnn.pth', # 替换为你的模型路径
    input_image_path='./path/to/your/lr_image.png', # 替换为你的低分辨率图片路径
    hr_image_path='./path/to/your/hr_image.png', # 可选，替换为对应的高分辨率图片路径以计算指标
    text_description='example description', # 可选，如果模型支持文本输入
    output_image_path='./output_sr_image.png' # 输出保存路径
)
```

#### 示例：评估数据集子集

```python
# 在 evaluate.py 的 if __name__ == '__main__': 部分
evaluate_dataset_subset(
    model_path='simple_srcnn.pth', # 替换为你的模型路径
    lr_data_dir='./data/val/lr', # 替换为你的验证LR数据目录
    hr_data_dir='./data/val/hr', # 替换为你的验证HR数据目录
    edge_detection_methods=['sobel'], # 可选，指定边缘检测方法，需与训练时一致
    num_samples=10, # 评估10张随机图片
    output_dir='./evaluation_results_subset', # 评估结果保存目录
    upscale_factor=2 # 根据你的模型设置
)
```

可以修改 `evaluate.py` 中的函数参数或 `if __name__ == '__main__':` 部分来调整评估配置。

## 关于处理不同尺寸的图像输入

对于像 `SimpleSRCNN` 这样的全卷积网络 (FCN)，模型本身在理论上可以处理不同大小的输入图像。然而，在实际训练和批处理中需要注意以下几点：

-   **批处理 (Batching)**: 当 `batch_size > 1` 时，同一批次内的所有图像通常需要具有相同的尺寸，以便能够堆叠成一个张量。常见的处理方法包括：
    -   **裁剪 (Cropping)**: 从原始图像中随机或固定位置裁剪出固定大小的图像块 (patches) 进行训练。
    -   **缩放 (Resizing)**: 将所有输入图像缩放到一个统一的尺寸。
-   **批大小为1 (Batch Size = 1)**: 如果训练时设置 `batch_size = 1`，则可以直接处理可变尺寸的输入图像，无需裁剪或缩放，只要内存允许。
-   **数据加载器 (`data_utils.py`)**: 当前的 `SRDataset` 在加载图像时并未强制统一尺寸。如果您计划使用大于1的批大小并遇到尺寸不匹配的错误，您需要在数据预处理阶段（例如在 `SRDataset` 的 `__getitem__` 方法中，或者通过 `transforms`）加入图像裁剪或缩放的逻辑。
-   **评估阶段**: 在评估单个图像时，通常可以直接使用原始尺寸的图像，因为不需要批处理。

## 后续工作与展望

-   **完善文本特征处理**: 目前模型对文本输入仅做了初步集成，尚未实现有效的文本特征提取和与图像特征的融合机制。后续可以引入文本编码器（如BERT、Transformer等）并将文本特征有效融入到超分辨率网络中。
-   **集成物理约束**: 当前已通过 `losses.py` 和灵活的损失函数参数 (`criterion`) 为引入物理约束打下基础。未来可以实现更复杂的物理约束，例如：
    -   **基于边缘的约束**: 如 `EdgeLoss` 所示例，可以利用Canny、Sobel等算子提取图像边缘，并约束模型生成的图像在边缘区域的表现，使其更锐利或符合特定边缘特征。
    -   **基于频率域的约束**: 约束SR图像在傅里叶变换等频域空间上的特性。
    -   **其他基于图像内容的先验知识**: 例如，约束特定区域的颜色、纹理等。这些约束可以通过在 `losses.py` 中定义新的损失项，并将其组合到 `CombinedLoss` 中来实现。
-   **更复杂的模型架构**: 当前 `SimpleSRCNN` 是一个基础模型，可以探索和集成更先进的超分辨率模型架构 (如 ESRGAN, RRDB, SwinIR 等)。
-   **更丰富的评估指标**: 除了视觉评估，已引入 PSNR, SSIM 等量化评估指标 (使用 `torchmetrics`)。可以进一步扩展支持更多指标 (如 LPIPS, FRC)。
-   **配置文件管理**: 使用配置文件 (如 YAML, JSON) 管理训练和评估参数，而不是硬编码在脚本中。
-   **日志和可视化**: 集成更完善的日志系统 (如 TensorBoard) 来监控训练过程。
-   **单元测试和集成测试**: 增加测试用例以确保代码的健壮性。
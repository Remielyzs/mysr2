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
├── config/                   # 配置文件目录
│   └── experiment_config.py  # 实验配置定义
├── data/                     # 存放训练和测试数据 (示例)
│   ├── hr/                   # 高分辨率图像
│   └── lr/                   # 低分辨率图像
├── data_utils.py             # 数据加载和预处理工具
├── evaluate.py               # 模型评估脚本
├── experiment/               # 实验管理目录
│   └── experiment_runner.py  # 实验运行器
├── losses.py                 # 损失函数定义
├── models/                   # 存放模型定义文件
│   ├── __init__.py
│   ├── simple_srcnn.py       # SimpleSRCNN模型
│   ├── basic_sr.py           # 基础超分辨率模型
│   └── ...                   # 其他模型定义
├── trainers/                 # 训练器目录
│   ├── base_trainer.py       # 基础训练器
│   ├── sr_trainer.py         # 超分辨率训练器
│   └── train_config.py       # 训练配置
├── train.py                  # 模型训练脚本
├── train_controller.py       # 训练控制器
├── utils/                    # 工具函数目录
│   └── evaluation_utils.py   # 评估工具
├── requirements.txt          # Python依赖包
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

项目现在使用实验配置和训练控制器来管理训练流程。运行训练脚本：

```bash
python train_controller.py
```

训练配置分为两个层次：

1. **基础训练参数** (`config/experiment_config.py` 中的 `BASE_TRAIN_PARAMS`)：
   - 模型类型和参数
   - 训练周期、批大小、学习率等基本参数
   - 数据目录配置
   - 设备选择
   - 检查点目录等

2. **实验配置** (`config/experiment_config.py` 中的 `TRAINING_RUNS`)：
   - 实验名称
   - 边缘检测方法
   - 损失函数配置
   - 其他实验特定参数

#### 配置示例

```python
# 在 config/experiment_config.py 中
from losses import L1Loss, CombinedLoss, EdgeLoss
import torch.nn as nn

class ExperimentConfig:
    # 基础训练参数
    BASE_TRAIN_PARAMS = {
        'model_class': SimpleSRCNN,
        'epochs': 10,
        'batch_size': 4,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # ... 其他基本参数
    }
    
    # 实验配置列表
    TRAINING_RUNS = [
        # 实验1：使用L1损失
        {
            'name': 'srcnn_l1',
            'edge_methods': None,
            'criterion': L1Loss()
        },
        # 实验2：使用边缘检测和组合损失
        {
            'name': 'srcnn_edge',
            'edge_methods': ['sobel'],
            'criterion': CombinedLoss([
                (nn.MSELoss(), 1.0),
                (EdgeLoss(edge_detector_type='sobel'), 0.1)
            ])
        }
        # ... 可以添加更多实验配置
    ]
```

训练控制器会自动：
1. 预处理所需的边缘检测数据
2. 为每个实验创建独立的结果目录
3. 保存检查点和训练日志
4. 记录训练和验证损失
5. 在训练结束时生成损失曲线图

所有实验结果将保存在各自的目录中，便于后续比较和分析。

### 4. 模型评估

项目提供了两种评估方式，都集成在 `evaluate.py` 中：

1. **单张图片评估**：用于快速测试模型效果
2. **数据集评估**：用于系统性评估模型性能

#### 单张图片评估

```bash
python evaluate.py --mode single \
    --model-path results/srcnn_l1/model_best.pth \
    --input-image path/to/lr_image.png \
    --hr-image path/to/hr_image.png \
    --output-image output_sr.png
```

主要参数说明：
- `--model-path`：模型权重文件路径
- `--input-image`：输入的低分辨率图像
- `--hr-image`：参考的高分辨率图像（可选）
- `--output-image`：超分辨率结果保存路径
- `--text-description`：文本描述（如果模型支持）
- `--edge-methods`：边缘检测方法（如果模型使用）

#### 数据集评估

```bash
python evaluate.py --mode dataset \
    --model-path results/srcnn_edge/model_best.pth \
    --val-lr-dir data/val/lr \
    --val-hr-dir data/val/hr \
    --edge-methods sobel \
    --num-samples 10 \
    --output-dir evaluation_results
```

主要参数说明：
- `--val-lr-dir`：验证集低分辨率图像目录
- `--val-hr-dir`：验证集高分辨率图像目录
- `--num-samples`：评估样本数量
- `--edge-methods`：边缘检测方法列表（多个方法用空格分隔）
- `--output-dir`：评估结果保存目录

评估结果包括：
1. 每个样本的PSNR和SSIM值
2. 整体平均PSNR和SSIM
3. 超分辨率结果图像
4. 评估报告（包含配置信息和结果统计）

#### 评估工具模块

评估相关的核心功能已重构到 `utils/evaluation_utils.py` 中的 `EvaluationUtils` 类，提供：
- 图像加载和预处理
- 评估指标计算（PSNR、SSIM等）
- 结果可视化和保存
- 批量评估支持

这种模块化设计使得评估流程更加清晰和可扩展，同时确保了评估方法的一致性。

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
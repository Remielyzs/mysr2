import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from image_processing import EdgeDetector

def preprocess_edge_data(
    input_dir: str,
    edge_methods: list = ['sobel', 'canny', 'laplacian'],
    device: str = 'cpu'
):
    """
    对指定目录下的图像进行边缘检测，并将结果保存为 .npy 文件。

    Args:
        input_dir (str): 包含输入图像的目录。
        output_base_dir (str): 保存边缘检测结果的根目录。
        edge_methods (list): 要使用的边缘检测方法列表。
        device (str): 计算设备 ('cpu' 或 'cuda').
    """
    if not edge_methods:
        print("No edge detection methods specified. Skipping pre-processing.")
        return

    print(f"Starting edge data pre-processing for images in {input_dir}...")

    print(f"Edge methods: {edge_methods}")

    detector = EdgeDetector(device=device)

    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')

    # 获取所有图像文件路径
    image_paths = sorted(
        [p for ext in supported_formats for p in glob.glob(os.path.join(input_dir, f'*{ext}'))]
    )

    if not image_paths:
        print(f"No supported image files found in {input_dir}. Skipping pre-processing.")
        return

    num_images = len(image_paths)
    print(f"Found {num_images} images to process.")

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]

        try:
            # 加载图像并转换为 RGB
            img = Image.open(img_path).convert('RGB')
            # 转换为 PyTorch 张量 [C, H, W], 范围 [0, 1]
            img_tensor = to_tensor(img).to(device)

            for method in edge_methods:
                output_dir = os.path.join(input_dir, method)
                os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, f"{name_without_ext}.npy")

                # 检查文件是否已存在，如果存在则跳过
                if os.path.exists(output_file_path):
                    # print(f"Edge file already exists: {output_file_path}. Skipping.")
                    continue

                # 执行边缘检测
                # EdgeDetector.detect 期望输入是 (B, C, H, W) 或 (C, H, W)
                # 返回 (1, H, W) 张量
                edge_tensor = detector.detect(img_tensor.unsqueeze(0), method=method).squeeze(0)

                # 将结果转回 CPU 并转换为 NumPy 数组 (H, W)
                edge_np = edge_tensor.cpu().numpy()

                # 保存为 .npy 文件
                np.save(output_file_path, edge_np)
                # print(f"Saved {method} edge for {img_name} to {output_file_path}")

            print(f"Processed {i+1}/{num_images}: {img_name}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("Edge data pre-processing complete.")

if __name__ == '__main__':
    # 示例用法：
    # 假设你的低分辨率图像在 ./data/lr 目录下
    # 预处理后的边缘数据将保存在 ./data/lr/method 目录下
    input_lr_dir = './data/lr'
    methods_to_process = ['sobel', 'canny'] # 指定要预处理的边缘方法

    # 创建一些虚拟的输入图像用于测试
    if not os.path.exists(input_lr_dir):
        os.makedirs(input_lr_dir)
        dummy_img_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        dummy_img = Image.fromarray(dummy_img_np, 'RGB')
        dummy_img.save(os.path.join(input_lr_dir, 'dummy_lr_001.png'))
        dummy_img.save(os.path.join(input_lr_dir, 'dummy_lr_002.png'))
        print(f"Created dummy images in {input_lr_dir} for testing.")

    preprocess_edge_data(
        input_dir=input_lr_dir,
        edge_methods=methods_to_process,
        device='cpu' # 或 'cuda' 如果可用
    )

    # 验证生成的文件
    print("\nVerifying generated files...")
    for method in methods_to_process:
        expected_file1 = os.path.join(input_lr_dir, method, 'dummy_lr_001.npy')
        expected_file2 = os.path.join(input_lr_dir, method, 'dummy_lr_002.npy')
        print(f"Checking {method} files:")
        print(f"  {expected_file1} exists: {os.path.exists(expected_file1)}")
        print(f"  {expected_file2} exists: {os.path.exists(expected_file2)}")

    # 清理虚拟数据和边缘数据 (可选)
    # import shutil
    # if os.path.exists(input_lr_dir):
    #     shutil.rmtree(input_lr_dir)
    # if os.path.exists(os.path.join(input_lr_dir, 'sobel')):
    #     shutil.rmtree(os.path.join(input_lr_dir, 'sobel'))
    # if os.path.exists(os.path.join(input_lr_dir, 'canny')):
    #     shutil.rmtree(os.path.join(input_lr_dir, 'canny'))
    # print("Cleaned up dummy data and edge data.")
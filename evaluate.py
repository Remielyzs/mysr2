import torch
import torchvision.transforms as transforms
from PIL import Image
import os

import numpy as np
import torchmetrics # 引入torchmetrics
from models.simple_srcnn import SimpleSRCNN # 默认模型
# from other_models import AnotherSRModel # 示例，用于演示模型切换
from data_utils import SRDataset # 引入SRDataset以支持评估模式的数据加载
import random # 用于随机抽样
import glob # 用于查找文件
import matplotlib.pyplot as plt # 用于绘制FRC曲线
# 需要安装 frc 库: pip install frc
import frc

def evaluate_image(model_class=SimpleSRCNN, model_params=None, model_path='simple_srcnn.pth', input_image_path=None, hr_image_path=None, text_description=None, output_image_path='output_hr.png'):
    """Evaluates a single image using the specified Super-Resolution model and calculates PSNR/SSIM if HR image is provided."""
    """Evaluates a single image using the specified Super-Resolution model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not input_image_path or not os.path.exists(input_image_path):
        print(f"Error: Input image file not found at {input_image_path}")
        return

    # Load the model
    if model_params is None:
        model_params = {}
    model = model_class(**model_params) # 实例化模型
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the input image
    input_image = Image.open(input_image_path).convert('RGB')
    image_transform = transforms.ToTensor()
    input_tensor = image_transform(input_image).unsqueeze(0) # Add batch dimension

    # Perform super-resolution
    print(f"Processing image: {input_image_path}")
    if text_description:
        print(f"With text description: {text_description}")
    with torch.no_grad():
        # 将图像和文本描述传递给模型
        # 注意：当前SimpleSRCNN的forward方法虽然接收text_input，但未实际使用它进行计算
        output_tensor = model(input_tensor, text_input=[text_description] if text_description else None)


    # Postprocess and save the output image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    output_image.save(output_image_path)
    print(f"Super-resolved image saved to {output_image_path}")

    if hr_image_path and os.path.exists(hr_image_path):
        hr_image = Image.open(hr_image_path).convert('RGB')
        # 确保HR图像和SR图像尺寸一致，如果SR模型改变了尺寸，HR图像可能需要对应调整大小
        # 这里假设SR输出与HR目标尺寸一致
        if output_image.size != hr_image.size:
            print(f"Warning: SR image size {output_image.size} and HR image size {hr_image.size} mismatch. Resizing HR to SR size for metrics.")
            hr_image = hr_image.resize(output_image.size, Image.BICUBIC)
        
        hr_tensor = image_transform(hr_image).unsqueeze(0) # Add batch dimension

        # 初始化指标
        psnr_metric = torchmetrics.PeakSignalNoiseRatio()
        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0) # data_range=1.0 for ToTensor output

        # 计算指标
        # torchmetrics expects inputs in range [0,1] if not specified otherwise for ssim
        psnr_value = psnr_metric(output_tensor, hr_tensor)
        ssim_value = ssim_metric(output_tensor, hr_tensor)
        frc_metric = frc.FRC()
        frc_value = frc_metric(output_tensor, hr_tensor)
        print(f"FRC: {frc_value.item():.4f}")
        report_content += f"- PSNR: {psnr_value.item():.4f}\n"
        report_content += f"- SSIM: {ssim_value.item():.4f}\n"
        report_content += f"- FRC: {frc_value.item():.4f}\n"

        print(f"PSNR: {psnr_value.item():.4f}")
        print(f"SSIM: {ssim_value.item():.4f}")
    elif hr_image_path:
        print(f"Warning: HR image path provided ({hr_image_path}) but file not found. Skipping metrics.")

if __name__ == '__main__':
    # Example usage:
    # 确保模型已经训练并保存 (例如，通过运行 train.py)
    # python train.py

    # 创建一个示例低分辨率图像用于评估
    example_lr_dir = './data_example_eval/lr'
    os.makedirs(example_lr_dir, exist_ok=True)
    dummy_lr_path = os.path.join(example_lr_dir, 'eval_lr_dummy.png')
    dummy_hr_size = 64
    dummy_lr_size = 32
    dummy_hr_img_np = np.random.randint(0, 256, (dummy_hr_size, dummy_hr_size, 3), dtype=np.uint8)
    dummy_hr_img = Image.fromarray(dummy_hr_img_np, 'RGB')
    dummy_lr_img = dummy_hr_img.resize((dummy_lr_size, dummy_lr_size), Image.BICUBIC)
    dummy_lr_img.save(dummy_lr_path)
    print(f"Created dummy LR image for evaluation at {dummy_lr_path}")

    # 示例文本描述
    example_text = "magnification: 2x, content: test image"

    # 评估这个图像 (使用默认的SimpleSRCNN模型)
    print("\n--- Evaluating with default SimpleSRCNN ---")
    evaluate_image(
        model_path='simple_srcnn.pth', # 确保这个模型文件存在
        input_image_path=dummy_lr_path,
        text_description=example_text,
        output_image_path='./dummy_output_hr_eval.png'
    )

    # 示例：如何使用不同的模型进行评估 (假设 AnotherSRModel 已定义并训练保存为 another_model.pth)
    # class AnotherSRModel(torch.nn.Module): # 简单定义以便运行
    #     def __init__(self, custom_param=128):
    #         super().__init__()
    #         self.conv = torch.nn.Conv2d(3,3,3,1,1)
    #         self.upsample = torch.nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
    #     def forward(self, x, text_input=None):
    #         x = self.conv(x)
    #         x = self.upsample(x)
    #         return x
    # # 假设你已经训练并保存了 AnotherSRModel 的权重到 'another_model.pth'
    # # torch.save(AnotherSRModel().state_dict(), 'another_model.pth') 
    # print("\n--- Evaluating with AnotherSRModel (example) ---")
    # evaluate_image(
    #     model_class=AnotherSRModel,
    #     model_params={'custom_param': 256},
    #     model_path='another_model.pth', # 需要确保此文件存在
    #     input_image_path=dummy_lr_path,
    #     text_description="another model test",
    #     output_image_path='./dummy_output_hr_another_model.png'
    # )

    # 清理示例文件
    # if os.path.exists(dummy_lr_path):
    #     os.remove(dummy_lr_path)
    # if os.path.exists('./dummy_output_hr_eval.png'):
    #     os.remove('./dummy_output_hr_eval.png')
    # if os.path.exists('./dummy_output_hr_another_model.png'):
    #     os.remove('./dummy_output_hr_another_model.png')
    # import shutil
    # if os.path.exists('./data_example_eval'):
    # shutil.rmtree('./data_example_eval')
    print("\nEvaluation example complete. Check for output images.")

def evaluate_dataset_subset(
    model_class=SimpleSRCNN,
    model_params=None,
    model_path='simple_srcnn.pth',
    val_lr_data_dir='./data/split_sample/val/lr', # Use new parameter name
    val_hr_data_dir='./data/split_sample/val/hr', # Use new parameter name
    edge_detection_methods=None,
    num_samples=10,
    output_dir='./evaluation_results',
    device='cpu',
    upscale_factor=2 # Assuming a default upscale factor
):
    """Evaluates a random subset of images from a dataset using the specified model and calculates metrics including FRC."""

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Check for validation data directories
    if not os.path.exists(val_lr_data_dir) or not os.path.exists(val_hr_data_dir):
        print(f"Error: Validation data directories not found at {val_lr_data_dir} or {val_hr_data_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    if model_params is None:
        model_params = {}

    # Determine number of input channels for the model (similar logic as in train.py)
    num_edge_channels = len(edge_detection_methods) if edge_detection_methods else 0 # 0 if None, as SRDataset handles this
    # Assuming original image is RGB (3 channels)
    in_channels_for_model = 3 + num_edge_channels

    # Add in_channels and upscale_factor to model_params if the model is BasicSRModel or similar that accepts it
    # This part might need adjustment based on the actual model classes used
    if hasattr(model_class, '__init__') and 'in_channels' in model_class.__init__.__code__.co_varnames:
         model_params['in_channels'] = in_channels_for_model
    if hasattr(model_class, '__init__') and 'upscale_factor' in model_class.__init__.__code__.co_varnames:
         model_params['upscale_factor'] = upscale_factor

    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Data loading
    image_transform = transforms.ToTensor()
    # Assuming text descriptions are not used for evaluation metrics calculation
    # Pass the validation directories explicitly
    dataset = SRDataset(lr_dir=None, hr_dir=None, text_descriptions=None, transform=image_transform, mode='eval', edge_methods=edge_detection_methods, device=device, upscale_factor=upscale_factor, val_lr_dir=val_lr_data_dir, val_hr_dir=val_hr_data_dir)

    if len(dataset) == 0:
        print("No images found in the dataset.")
        return

    # Select random samples
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Evaluating {len(sample_indices)} random samples.")

    psnr_values = []
    ssim_values = []
    frc_curves = []

    # Initialize metrics
    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            lr_image_tensor, hr_image_tensor, _ = dataset[idx] # text_desc is None in eval mode
            lr_image_tensor = lr_image_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device
            hr_image_tensor = hr_image_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

            # Perform super-resolution
            # Need to handle models that might expect text_input even if it's None
            if hasattr(model, 'forward') and 'text_input' in model.forward.__code__.co_varnames:
                 sr_output_tensor = model(lr_image_tensor, text_input=None)
            else:
                 sr_output_tensor = model(lr_image_tensor)

            # Ensure SR output and HR target have the same size for metrics
            # This might involve cropping or resizing the HR target if the model output size is fixed/different
            # Assuming model output matches HR size for now based on typical SR setup
            if sr_output_tensor.shape[-2:] != hr_image_tensor.shape[-2:]:
                 print(f"Warning: Sample {i} SR output size {sr_output_tensor.shape[-2:]} and HR target size {hr_image_tensor.shape[-2:]} mismatch. Skipping metrics for this sample.")
                 continue

            # Calculate PSNR and SSIM
            psnr_value = psnr_metric(sr_output_tensor, hr_image_tensor)
            ssim_value = ssim_metric(sr_output_tensor, hr_image_tensor)
            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())

            # Calculate FRC
            # FRC library expects numpy arrays, potentially grayscale
            # Convert tensors to numpy, remove batch dim, move to CPU, convert to grayscale if needed
            sr_np = sr_output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() # C, H, W -> H, W, C
            hr_np = hr_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() # C, H, W -> H, W, C

            # Convert to grayscale for FRC if it's a color image
            if sr_np.shape[-1] == 3:
                 sr_np = np.mean(sr_np, axis=-1) # Simple grayscale conversion
            if hr_np.shape[-1] == 3:
                 hr_np = np.mean(hr_np, axis=-1) # Simple grayscale conversion

            # Ensure images are square and apply windowing if needed by the FRC library
            # The 'frc' library's example uses square_image and apply_tukey
            try:
                # Need two noise-independent images for standard FRC, or use one_frc
                # Since we have SR and HR, we'll use standard FRC if possible, or one_frc on SR output
                # The user asked for FRC between SR and HR, so let's try standard FRC first.
                # This might require splitting HR into two noisy versions, which is complex.
                # A simpler interpretation is to calculate FRC of the SR output itself (1FRC) or between SR and HR directly (less standard for resolution metric, but possible for correlation visualization).
                # Let's calculate FRC between SR and HR as requested, although it's not the standard resolution FRC.
                # The 'frc' library's main function seems to expect two images.
                # Let's check the 'frc' library documentation or examples again.
                # The 'frc' library's `frc.frc` function takes two images.
                # It also has `frc.one_frc` for a single image.
                # Given the user's request to compare SR and HR, using `frc.frc(sr_np, hr_np)` seems most direct, though its interpretation as a resolution metric might be non-standard.
                # Let's use `frc.frc` and plot the curve.

                # Ensure images are float and in the expected range for FRC library (e.g., 0-1 or 0-255)
                # Assuming ToTensor gives 0-1, let's keep it that way.
                sr_np = sr_np.astype(np.float32)
                hr_np = hr_np.astype(np.float32)

                # Ensure images are the same size (already checked above)
                # Ensure images are square for standard FRC implementation
                size = min(sr_np.shape[0], sr_np.shape[1])
                sr_square = sr_np[:size, :size]
                hr_square = hr_np[:size, :size]

                # Apply windowing (optional but common)
                sr_windowed = frc.util.apply_tukey(sr_square)
                hr_windowed = frc.util.apply_tukey(hr_square)

                # Calculate FRC curve
                # The frc.frc function expects two images
                frc_curve = frc.frc(sr_windowed, hr_windowed)
                frc_curves.append(frc_curve)

                # Plot FRC curve for this sample
                plt.figure()
                img_size = size # Use the size of the square image
                xs_pix = np.arange(len(frc_curve)) / img_size
                plt.plot(xs_pix, frc_curve)
                plt.xlabel('Spatial Frequency (cycles/pixel)')
                plt.ylabel('FRC')
                plt.title(f'FRC Curve for Sample {i+1}')
                frc_plot_path = os.path.join(output_dir, f'sample_{idx}_frc_curve.png')
                plt.savefig(frc_plot_path)
                plt.close()
                print(f"Saved FRC plot for sample {idx} to {frc_plot_path}")

            except Exception as e:
                print(f"Could not calculate FRC for sample {idx}: {e}")
                # Optionally save the SR and HR images for debugging
                # transforms.ToPILImage()((sr_output_tensor.squeeze(0).cpu())).save(os.path.join(output_dir, f'sample_{idx}_sr_error.png'))
                # transforms.ToPILImage()((hr_image_tensor.squeeze(0).cpu())).save(os.path.join(output_dir, f'sample_{idx}_hr_error.png'))

    # Report average metrics
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        print(f"\nAverage PSNR over {len(psnr_values)} samples: {avg_psnr:.4f}")
    if ssim_values:
        avg_ssim = np.mean(ssim_values)
        print(f"Average SSIM over {len(ssim_values)} samples: {avg_ssim:.4f}")

    # Note: Averaging FRC curves is not standard. We saved individual plots.
    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    # Example usage:
    # 确保模型已经训练并保存 (例如，通过运行 train.py)
    # python train.py

    # 创建一个示例低分辨率图像用于评估
    example_lr_dir = './data_example_eval/lr'
    os.makedirs(example_lr_dir, exist_ok=True)
    dummy_lr_path = os.path.join(example_lr_dir, 'eval_lr_dummy.png')
    dummy_hr_size = 64
    dummy_lr_size = 32
    dummy_hr_img_np = np.random.randint(0, 256, (dummy_hr_size, dummy_hr_size, 3), dtype=np.uint8)
    dummy_hr_img = Image.fromarray(dummy_hr_img_np, 'RGB')
    dummy_lr_img = dummy_hr_img.resize((dummy_lr_size, dummy_lr_size), Image.BICUBIC)
    dummy_lr_img.save(dummy_lr_path)
    print(f"Created dummy LR image for evaluation at {dummy_lr_path}")

    # 示例文本描述
    example_text = "magnification: 2x, content: test image"

    # 评估这个图像 (使用默认的SimpleSRCNN模型)
    print("\n--- Evaluating with default SimpleSRCNN ---")
    evaluate_image(
        model_path='simple_srcnn.pth', # 确保这个模型文件存在
        input_image_path=dummy_lr_path,
        text_description=example_text,
        output_image_path='./dummy_output_hr_eval.png'
    )

    # 示例：如何使用不同的模型进行评估 (假设 AnotherSRModel 已定义并训练保存为 another_model.pth)
    # class AnotherSRModel(torch.nn.Module): # 简单定义以便运行
    #     def __init__(self, custom_param=128):
    #         super().__init__()
    #         self.conv = torch.nn.Conv2d(3,3,3,1,1)
    #         self.upsample = torch.nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
    #     def forward(self, x, text_input=None):
    #         x = self.conv(x)
    #         x = self.upsample(x) # Assuming 2x upscale
    #         return x
    # # 假设你已经训练并保存了 AnotherSRModel 的权重到 'another_model.pth'
    # # torch.save(AnotherSRModel().state_dict(), 'another_model.pth')
    # print("\n--- Evaluating with AnotherSRModel (example) ---")
    # evaluate_image(
    #     model_class=AnotherSRModel,
    #     model_params={'custom_param': 256},
    #     model_path='another_model.pth', # 需要确保此文件存在
    #     input_image_path=dummy_lr_path,
    #     text_description="another model test",
    #     output_image_path='./dummy_output_hr_another_model.png'
    # )

    # 示例：评估数据集子集
    print("\n--- Evaluating dataset subset ---")
    # 确保你有一个验证数据集在 ./data/val/lr 和 ./data/val/hr
    # 确保你有一个训练好的模型文件，例如 simple_srcnn.pth
    evaluate_dataset_subset(
        model_path='simple_srcnn.pth', # 替换为你的模型路径
        lr_data_dir='./data/val/lr', # 替换为你的验证LR数据目录
        hr_data_dir='./data/val/hr', # 替换为你的验证HR数据目录
        edge_detection_methods=['sobel'], # 示例：使用Sobel边缘
        num_samples=10, # 评估10张随机图片
        output_dir='./evaluation_results_subset',
        upscale_factor=2 # 根据你的模型设置
    )

    # 清理示例文件
    # if os.path.exists(dummy_lr_path):
    #     os.remove(dummy_lr_path)
    # if os.path.exists('./dummy_output_hr_eval.png'):
    #     os.remove('./dummy_output_hr_eval.png')
    # if os.path.exists('./dummy_output_hr_another_model.png'):
    #     os.remove('./dummy_output_hr_another_model.png')
    # import shutil
    # if os.path.exists('./data_example_eval'):
    #     shutil.rmtree('./data_example_eval')
    # print("\nEvaluation example complete. Check for output images.")


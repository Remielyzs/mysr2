import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import datetime
import matplotlib.pyplot as plt
from PIL import Image # 用于加载和保存图像样本
import shutil # 用于复制文件

import torch.nn as nn # 引入nn以支持更灵活的模型定义
from models.simple_srcnn import SimpleSRCNN # 默认模型
from models.basic_sr import BasicSRModel # 导入BasicSRModel
from data_utils import SRDataset, generate_synthetic_data
from evaluate import evaluate_dataset_subset # 导入评估函数

# 允许在这里定义或导入其他模型
# from other_models import AnotherSRModel # 示例

def train_model(model_class=SimpleSRCNN, model_params=None, lr_data_dir='./data/lr', hr_data_dir='./data/hr', epochs=10, batch_size=64, learning_rate=0.001, use_text_descriptions=False, criterion=None, results_base_dir='results', resume_checkpoint=None, model_name='model', edge_detection_methods=None, device='cpu', lr_patch_size=None, upscale_factor=None, edge_data_dir=None, val_lr_data_dir=None, val_hr_data_dir=None, image_size=None):
    """Trains the Super-Resolution model with checkpointing and flexible loss."""
    print(f"Using device: {device}")
    """Trains the Super-Resolution model."""
    # Ensure data exists
    # lr_dir and hr_dir are now direct parameters
    if not os.path.exists(lr_data_dir) or not os.path.exists(hr_data_dir):
        print(f"Data not found. Expected LR data at {lr_data_dir} and HR data at {hr_data_dir}.")
        # Decide if synthetic data generation is still appropriate or should be handled differently
        # For now, let's assume if specific dirs are given, synthetic data generation might not be desired
        # Or, we could try to generate into a default 'data_dir' if that's still a concept
        # For this change, we will raise an error if specific dirs are not found.
        raise FileNotFoundError(f"LR directory {lr_data_dir} or HR directory {hr_data_dir} not found.")
        # If you still want to generate synthetic data, you'd need a base 'data_dir' for generate_synthetic_data
        # print("Data not found. Generating synthetic data...")
        # generate_synthetic_data(os.path.dirname(lr_data_dir)) # Assuming lr_data_dir is like 'data/lr'

    # Data loading
    image_transform = transforms.ToTensor()
    
    # 准备文本描述 (示例，实际应用中应从文件或其他来源加载)
    text_descriptions_train = None
    if use_text_descriptions:
        # 获取lr图像数量以匹配文本描述数量
        num_lr_images = len([f for ext in ('.png', '.tiff', '.tif', '.npz') for f in glob.glob(os.path.join(lr_data_dir, f'*{ext}'))])
        text_descriptions_train = [f"Sample text for image {i}" for i in range(num_lr_images)]
        if not text_descriptions_train:
            print("Warning: use_text_descriptions is True, but no LR images found to generate dummy text descriptions.")

    # Determine number of input channels for the model
    # Assuming original image is RGB (3 channels)
    num_edge_channels = len(edge_detection_methods) if edge_detection_methods else 0
    in_channels_for_model = 3 + num_edge_channels

    # Set a default image_size if neither lr_patch_size nor image_size is specified
    # This ensures consistent image sizes for the DataLoader when not using patching.
    default_image_size = None
    default_upscale_factor = None
    if lr_patch_size is None and image_size is None:
        print("Neither lr_patch_size nor image_size specified. Setting default image_size to 64 and upscale_factor to 2.")
        default_image_size = 64
        default_upscale_factor = 2 # Assuming a default upscale factor of 2
    
    # Use the provided image_size or the default one
    effective_image_size = image_size if image_size is not None else default_image_size
    effective_upscale_factor = upscale_factor if upscale_factor is not None else default_upscale_factor

    # Ensure upscale_factor is available if image_size is set (either provided or default)
    if effective_image_size is not None and effective_upscale_factor is None:
         raise ValueError("upscale_factor must be provided or inferrable if image_size is specified or defaulted.")

    dataset = SRDataset(lr_data_dir, hr_data_dir, text_descriptions=text_descriptions_train, transform=image_transform, mode='train', edge_methods=edge_detection_methods, device=device, lr_patch_size=lr_patch_size, upscale_factor=effective_upscale_factor, image_size=effective_image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Validation data loading
    # Assuming validation data is in a 'val' subdirectory within the data directories
    val_lr_data_dir = lr_data_dir.replace('/lr', '/val/lr')
    val_hr_data_dir = hr_data_dir.replace('/hr', '/val/hr')

    val_dataset = None
    val_dataloader = None
    if os.path.exists(val_lr_data_dir) and os.path.exists(val_hr_data_dir):
        print(f"Found validation data at {val_lr_data_dir} and {val_hr_data_dir}.")
        # Prepare text descriptions for validation if needed (assuming similar structure)
        text_descriptions_val = None
        if use_text_descriptions:
             # 获取lr图像数量以匹配文本描述数量
            num_lr_images_val = len([f for ext in ('.png', '.tiff', '.tif', '.npz') for f in glob.glob(os.path.join(val_lr_data_dir, f'*{ext}'))])
            text_descriptions_val = [f"Sample text for validation image {i}" for i in range(num_lr_images_val)]
            if not text_descriptions_val:
                 print("Warning: use_text_descriptions is True for validation, but no LR images found to generate dummy text descriptions.")

        val_dataset = SRDataset(lr_dir=None, hr_dir=None, text_descriptions=text_descriptions_val, transform=image_transform, mode='eval', edge_methods=edge_detection_methods, device=device, lr_patch_size=lr_patch_size, upscale_factor=effective_upscale_factor, val_lr_dir=val_lr_data_dir, val_hr_dir=val_hr_data_dir, image_size=effective_image_size)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Set batch_size to 1 for validation to handle varying image sizes
    else:
        print("Warning: Validation data directories not found. Skipping validation during training.")


    # Model, Loss, and Optimizer
    if model_params is None:
        model_params = {}
    
    # Add in_channels to model_params if the model accepts it
    # Assuming SimpleSRCNN and BasicSRModel accept 'in_channels'
    if model_class in [SimpleSRCNN, BasicSRModel]:
        model_params['in_channels'] = in_channels_for_model
        # Ensure upscale_factor is present if not already in model_params, BasicSRModel needs it
        if model_class == BasicSRModel and 'upscale_factor' not in model_params:
            model_params['upscale_factor'] = 2 # Default or make it configurable
    
    model = model_class(**model_params) # 实例化传入的模型类
    model.to(device) # 将模型移动到指定设备

    if criterion is None:
        criterion = nn.MSELoss() # 默认损失函数
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_loss = float('inf')
    all_train_losses = [] # 用于记录每个epoch的训练loss
    all_val_losses = [] # 用于记录每个epoch的验证loss

    # 创建本次训练的专属结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_specific_results_dir = os.path.join(results_base_dir, f"{model_name}_{timestamp}")
    checkpoint_dir = os.path.join(run_specific_results_dir, 'checkpoints') # 检查点保存在子目录中
    images_dir = os.path.join(run_specific_results_dir, 'images') # 用于存放报告中的图片

    if not os.path.exists(run_specific_results_dir):
        os.makedirs(run_specific_results_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Load from checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        best_loss = checkpoint.get('loss', float('inf')) # Get loss if available
        print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint.get('loss', 'N/A')}")
    elif resume_checkpoint:
        print(f"Checkpoint not found at {resume_checkpoint}. Starting from scratch.")


    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for i, batch_data in enumerate(dataloader):
            # Conditionally unpack batch_data based on use_text_descriptions
            if use_text_descriptions:
                lr_images, hr_images, text_descs = batch_data
            else:
                lr_images, hr_images = batch_data
                text_descs = None # Ensure text_descs is None if not used

            lr_images, hr_images = lr_images.to(device), hr_images.to(device) # 将数据移动到指定设备
            
            optimizer.zero_grad()

            if use_text_descriptions and hasattr(model, 'forward') and 'text_input' in model.forward.__code__.co_varnames:
                outputs = model(lr_images, text_input=text_descs)
            else:
                outputs = model(lr_images) # BasicSRModel不需要text_input
            loss = criterion(outputs, hr_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0: # Print progress every 10 batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(dataloader)
        all_train_losses.append(epoch_train_loss) # 记录训练loss
        print(f"Epoch {epoch+1}/{epochs}, Average Training Loss: {epoch_train_loss:.4f}")

        # --- Validation loop ---
        epoch_val_loss = float('inf') # Default to infinity if no validation data
        if val_dataloader:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    # In evaluation mode, SRDataset only returns lr_images and hr_images
                    lr_images, hr_images = batch_data
                    text_descs = None # Ensure text_descs is None for evaluation

                    lr_images, hr_images = lr_images.to(device), hr_images.to(device)

                    if use_text_descriptions and hasattr(model, 'forward') and 'text_input' in model.forward.__code__.co_varnames:
                        outputs = model(lr_images, text_input=text_descs)
                    else:
                        outputs = model(lr_images)
                    loss = criterion(outputs, hr_images)
                    running_val_loss += loss.item()

            epoch_val_loss = running_val_loss / len(val_dataloader)
            all_val_losses.append(epoch_val_loss) # 记录验证loss
            print(f"Epoch {epoch+1}/{epochs}, Average Validation Loss: {epoch_val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss, # Save validation loss in checkpoint
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save the best model (optional, based on validation loss)
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            # 将最佳模型保存在 run_specific_results_dir 根目录
            best_model_path = os.path.join(run_specific_results_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} based on validation loss {best_loss:.4f}")

    print("Training complete.")
    # Save the final model to run_specific_results_dir root
    final_model_path = os.path.join(run_specific_results_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # --- 评估模型并生成报告 ---
    print("\n--- Evaluating model on validation set ---")
    # Pass the trained model instance and relevant parameters to the evaluation function
    # Ensure the model is on the correct device for evaluation
    model.to(device)
    avg_psnr, avg_ssim = evaluate_dataset_subset(
        model_class=model_class,
        model_params=model_params, # Pass the same parameters used for training
        model_path=final_model_path, # Use the final trained model
        val_lr_data_dir=val_lr_data_dir,
        val_hr_data_dir=val_hr_data_dir,
        edge_detection_methods=edge_detection_methods,
        num_samples=10, # Evaluate on a subset of validation data
        output_dir=os.path.join(run_specific_results_dir, 'evaluation_metrics'), # Save evaluation results in a subdirectory
        device=device,
        upscale_factor=upscale_factor
    )

    report_path = os.path.join(run_specific_results_dir, f"report_{model_name}_{timestamp}.md")

    # 1. 绘制Loss曲线图
    plt.figure()
    plt.plot(range(start_epoch + 1, epochs + 1), all_train_losses, label='Training Loss')
    if all_val_losses:
        plt.plot(range(start_epoch + 1, epochs + 1), all_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    loss_curve_path = os.path.join(images_dir, f"loss_curve_{model_name}_{timestamp}.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")

    # 2. 生成最佳结果样图 (选择一个样本)
        # 2. 生成最佳结果样图 (选择第一个样本)
    best_model_to_load_path = os.path.join(run_specific_results_dir, f"{model_name}_best.pth")
    if os.path.exists(best_model_to_load_path):
        model.load_state_dict(torch.load(best_model_to_load_path))
        model.eval()

        # 从dataloader获取第一个样本，确保dataloader不是空的
        if len(dataloader) > 0:
            # Conditionally unpack sample_data based on use_text_descriptions
            sample_data = next(iter(dataloader))
            if use_text_descriptions:
                sample_lr_img_tensor, sample_hr_img_tensor, sample_text_desc = sample_data
            else:
                sample_lr_img_tensor, sample_hr_img_tensor = sample_data
                sample_text_desc = None # Ensure text_descs is None if not used

            sample_lr_to_show = sample_lr_img_tensor[0:1] # 取第一个样本
            sample_hr_to_show = sample_hr_img_tensor[0:1]

            with torch.no_grad():
                if use_text_descriptions and hasattr(model, 'forward') and 'text_input' in model.forward.__code__.co_varnames: # Check if model accepts text_input
                    sample_output_tensor = model(sample_lr_to_show.to(device), text_input=[sample_text_desc[0]] if isinstance(sample_text_desc, list) else [sample_text_desc])
                else:
                    sample_output_tensor = model(sample_lr_to_show.to(device))

            # 保存图像
            # 输入LR图像 (可能包含边缘图)
            # 如果有边缘特征，将边缘特征叠加到LR图像旁边进行展示
            lr_img_tensor_rgb = sample_lr_to_show[0,:3].cpu().squeeze(0) # 只取RGB通道
            lr_pil = transforms.ToPILImage()(lr_img_tensor_rgb)
            lr_sample_path = os.path.join(images_dir, f"sample_lr_{timestamp}.png")
            lr_pil.save(lr_sample_path)

            # 如果使用了边缘检测，保存边缘特征图
            edge_sample_path = None
            if edge_detection_methods and len(edge_detection_methods) > 0:
                # 提取边缘特征通道 (从第4个通道开始)
                edge_features_tensor = sample_lr_to_show[0, 3:].cpu().squeeze(0)
                # 将多个边缘特征叠加显示 (例如，转换为灰度并叠加)
                # 如果是多个通道，可以考虑求平均或直接堆叠可视化
                # 这里简单地将所有边缘通道求平均，然后转换为灰度图可视化
                if edge_features_tensor.ndim == 3: # C, H, W
                    # 如果有多个边缘通道，求平均
                    edge_features_mean = torch.mean(edge_features_tensor, dim=0, keepdim=True) # 1, H, W
                else:
                    # 如果只有一个边缘通道
                    edge_features_mean = edge_features_tensor.unsqueeze(0) # 1, H, W

                # 将灰度张量转换为PIL图像 (需要确保数据范围在0-1或0-255)
                # 假设边缘特征是0-1范围的浮点数
                edge_pil = transforms.ToPILImage(mode='L')(edge_features_mean.squeeze(0))
                edge_sample_path = os.path.join(images_dir, f"sample_edge_features_{timestamp}.png")
                edge_pil.save(edge_sample_path)
                print(f"Sample edge features image saved to {edge_sample_path}")

            # 输出SR图像
            sr_pil = transforms.ToPILImage()(sample_output_tensor.cpu().squeeze(0))
            sr_sample_path = os.path.join(images_dir, f"sample_sr_output_{timestamp}.png")
            sr_pil.save(sr_sample_path)

            # 真实HR图像
            hr_pil = transforms.ToPILImage()(sample_hr_to_show[0].cpu().squeeze(0))
            hr_sample_path = os.path.join(images_dir, f"sample_hr_groundtruth_{timestamp}.png")
            hr_pil.save(hr_sample_path)
            print(f"Sample images (LR, SR, HR) saved in {images_dir}")
        else:
            print("Dataloader is empty, cannot generate sample images.")
            lr_sample_path, sr_sample_path, hr_sample_path, edge_sample_path = None, None, None, None
    else:
        print(f"Best model not found at {best_model_to_load_path}, cannot generate sample images.")
        lr_sample_path, sr_sample_path, hr_sample_path, edge_sample_path = None, None, None, None

    # 3. 编写报告内容
    report_content = f"# Training Report for {model_name}\n\n"
    report_content += f"Timestamp: {timestamp}\n\n"
    report_content += f"## Dataset Information\n"
    report_content += f"- LR Data Directory: `{lr_data_dir}`\n"
    report_content += f"- HR Data Directory: `{hr_data_dir}`\n\n"
    report_content += f"## Model Information\n"
    report_content += f"- Model Class: `{model_class.__name__}`\n"
    report_content += f"- Model Parameters: `{model_params}`\n"
    report_content += f"- Model Structure:\n```\n{str(model)}\n```\n\n"
    report_content += f"## Training Configuration\n"
    report_content += f"- Epochs: {epochs}\n"
    report_content += f"- Batch Size: {batch_size}\n"
    report_content += f"- Learning Rate: {learning_rate}\n"
    report_content += f"- Loss Function: `{criterion.__class__.__name__}`\n"
    report_content += f"- Device: `{device}`\n"
    if edge_detection_methods:
        report_content += f"- Edge Detection Methods: `{edge_detection_methods}`\n"
    report_content += f"- Using Text Descriptions: {use_text_descriptions}\n\n"
    report_content += f"## Training Results\n"
    report_content += f"- Best Loss: {best_loss:.4f}\n"
    report_content += f"- Final Model Path: `{os.path.relpath(final_model_path, results_base_dir)}`\n"
    report_content += f"- Best Model Path: `{os.path.relpath(best_model_path, results_base_dir) if os.path.exists(best_model_to_load_path) else 'N/A'}`\n\n"
    
    # Add evaluation metrics to the report
    report_content += f"## Evaluation Metrics\n"
    if avg_psnr is not None:
        report_content += f"- Average PSNR on Validation Set: {avg_psnr:.4f}\n"
    if avg_ssim is not None:
        report_content += f"- Average SSIM on Validation Set: {avg_ssim:.4f}\n"
    # if frc_curves:
    #     report_content += f"- FRC Curves for {len(frc_curves)} samples saved in `./evaluation_metrics` directory.\n"
    # else:
    #     report_content += f"- Evaluation metrics could not be calculated.\n"
    # report_content += f"\n"

    report_content += f"### Loss Curve\n"
    report_content += f"![Loss Curve](./images/{os.path.basename(loss_curve_path)})\n\n"
    report_content += f"### Sample Result\n"
    if lr_sample_path and sr_sample_path and hr_sample_path:
        report_content += f"- Input LR Image: ![LR Sample](./images/{os.path.basename(lr_sample_path)})\n"
        if edge_sample_path:
             report_content += f"- Input Edge Features: ![Edge Sample](./images/{os.path.basename(edge_sample_path)})\n"
        report_content += f"- Output SR Image: ![SR Sample](./images/{os.path.basename(sr_sample_path)})\n"
        report_content += f"- Ground Truth HR Image: ![HR Sample](./images/{os.path.basename(hr_sample_path)})\n"
    else:
        report_content += f"- Sample images could not be generated.\n"

    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Report saved to {report_path}")

    # 将checkpoint目录下的所有内容复制到 run_specific_results_dir/checkpoints (如果需要的话)
    # 当前逻辑是直接保存在那里，所以不需要复制
    # 如果之前的逻辑是保存在项目根目录的checkpoints，则需要复制
    # e.g., if os.path.exists(original_checkpoint_dir_path) and original_checkpoint_dir_path != checkpoint_dir:
    #    shutil.copytree(original_checkpoint_dir_path, os.path.join(run_specific_results_dir, 'all_checkpoints'))

    print(f"All results, models, and reports saved in: {run_specific_results_dir}")

if __name__ == '__main__':
    import argparse
    import json # 用于解析model_params_json
    import glob # 需要glob来辅助生成示例文本描述

    parser = argparse.ArgumentParser(description='Train Super-Resolution Models')
    parser.add_argument('--lr_data_dir', type=str, default='./data/lr', help='Directory for low-resolution images')
    parser.add_argument('--hr_data_dir', type=str, default='./data/hr', help='Directory for high-resolution images')
    parser.add_argument('--model_class_name', type=str, default='SimpleSRCNN', help='Name of the model class to use (e.g., SimpleSRCNN, BasicSRModel)')
    parser.add_argument('--model_params_json', type=str, default='{}', help='JSON string of model parameters')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--use_text_descriptions', action='store_true', help='Whether to use text descriptions')
    parser.add_argument('--criterion_name', type=str, default='MSELoss', help='Loss function name (e.g., MSELoss, L1Loss)')
    parser.add_argument('--results_base_dir', type=str, default='results', help='Base directory to save results')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--model_name', type=str, default='model', help='Name for the current training run and saved model files')
    parser.add_argument('--edge_detection_methods', nargs='*', default=None, help='List of edge detection methods (e.g., sobel canny laplacian) or None for default in SRDataset')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., cpu, cuda)')
    parser.add_argument('--lr_patch_size', type=int, default=None, help='Size of the LR image patch (e.g., 64). If specified, images will be resized.')
    parser.add_argument('--upscale_factor', type=int, default=None, help='Upscale factor for HR images when lr_patch_size is specified.')

    args = parser.parse_args()

    # Map model_class_name to actual class
    model_classes = {
        'SimpleSRCNN': SimpleSRCNN,
        'BasicSRModel': BasicSRModel,
        # 'AnotherSRModel': AnotherSRModel, # Add other models here if defined
    }
    model_class_to_use = model_classes.get(args.model_class_name)
    if model_class_to_use is None:
        raise ValueError(f"Unknown model class name: {args.model_class_name}. Available: {list(model_classes.keys())}")

    # Parse model_params_json
    try:
        model_params_to_use = json.loads(args.model_params_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON string for model_params_json: {args.model_params_json}")

    # Map criterion_name to actual loss function
    loss_functions = {
        'MSELoss': nn.MSELoss,
        'L1Loss': nn.L1Loss,
        # Add other loss functions here
    }
    criterion_to_use = loss_functions.get(args.criterion_name)
    if criterion_to_use is None:
        raise ValueError(f"Unknown criterion name: {args.criterion_name}. Available: {list(loss_functions.keys())}")
    criterion_instance = criterion_to_use()

    print(f"--- Starting training with command line arguments for {args.model_name} ---")
    train_model(
        model_class=model_class_to_use,
        model_params=model_params_to_use,
        lr_data_dir=args.lr_data_dir,
        hr_data_dir=args.hr_data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_text_descriptions=args.use_text_descriptions,
        criterion=criterion_instance,
        results_base_dir=args.results_base_dir,
        resume_checkpoint=args.resume_checkpoint,
        model_name=args.model_name,
        edge_detection_methods=args.edge_detection_methods,
        device=args.device,
        lr_patch_size=args.lr_patch_size,
        upscale_factor=args.upscale_factor
    )

    # Example usage (commented out as we are now using argparse):
    # print("--- Training with default SimpleSRCNN and checkpointing ---")
    # All example usages below are now handled by command-line arguments.
    # You can run the script with --help to see all options.
    # Example for DIV2K x2 training:
    # python train.py --lr_data_dir ./data/DIV2K/DIV2K_train_LR_bicubic/X2 \
    #                 --hr_data_dir ./data/DIV2K/DIV2K_train_HR \
    #                 --model_class_name BasicSRModel \
    #                 --model_params_json '{"upscale_factor": 2}' \
    #                 --epochs 50 \
    #                 --batch_size 16 \
    #                 --learning_rate 0.0001 \
    #                 --model_name div2k_x2_basicsr \
    #                 --device cuda \
    #                 --edge_detection_methods None # Or specify, e.g., sobel canny
    pass # Keep the if __name__ == '__main__': block, but it's now driven by argparse
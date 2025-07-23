import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import glob # 用于查找文件
from image_processing import EdgeDetector # 导入边缘检测器

def generate_synthetic_data(base_output_dir, num_samples=100, split_ratio=0.8):
    """Generates synthetic low-resolution and high-resolution image pairs and splits into train/val."""
    train_lr_dir = os.path.join(base_output_dir, 'train_lr')
    train_hr_dir = os.path.join(base_output_dir, 'train_hr')
    val_lr_dir = os.path.join(base_output_dir, 'val_lr')
    val_hr_dir = os.path.join(base_output_dir, 'val_hr')

    for d in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    print(f"Generating {num_samples} synthetic data samples...")

    num_train = int(num_samples * split_ratio)

    for i in range(num_samples):
        # Create a random high-resolution image (e.g., 64x64)
        hr_size = 64
        hr_img_np = np.random.randint(0, 256, (hr_size, hr_size, 3), dtype=np.uint8)
        hr_img = Image.fromarray(hr_img_np, 'RGB')

        # Create a low-resolution image by downsampling (e.g., 32x32)
        lr_size = 32
        lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)

        if i < num_train:
            # Save to training directories
            hr_img.save(os.path.join(train_hr_dir, f'hr_{i:04d}.png'))
            lr_img.save(os.path.join(train_lr_dir, f'lr_{i:04d}.png'))
        else:
            # Save to validation directories
            hr_img.save(os.path.join(val_hr_dir, f'hr_{i:04d}.png'))
            lr_img.save(os.path.join(val_lr_dir, f'lr_{i:04d}.png'))

    print("Synthetic data generation complete.")

class SRDataset(torch.utils.data.Dataset):
    """Custom Dataset for Super-Resolution with edge information."""
    def __init__(self, lr_dir, hr_dir=None, text_descriptions=None, transform=None, mode='train', edge_methods=None, device='cpu', lr_patch_size=None, upscale_factor=None, val_lr_dir=None, val_hr_dir=None, image_size=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.mode = mode
        self.text_descriptions = text_descriptions # 存储文本描述
        self.edge_methods = edge_methods if edge_methods is not None else [] # Default to empty list if None
        self.device = device
        self.lr_patch_size = lr_patch_size
        self.upscale_factor = upscale_factor
        self.image_size = image_size # 新增参数：用于在不使用patching时指定图像尺寸

        # EdgeDetector is only needed for pre-calculation, not during loading
        self.edge_detector = None

        # 支持多种图像格式
        self.supported_formats = ('.png', '.tiff', '.tif', '.npz')

        if self.mode == 'train':
            if not hr_dir:
                raise ValueError("Training mode requires hr_dir.")
            self.lr_image_paths = sorted(
                [p for ext in self.supported_formats for p in glob.glob(os.path.join(lr_dir, f'*{ext}'))] # Use glob for more robust file listing
            )
            self.hr_image_paths = sorted(
                [p for ext in self.supported_formats for p in glob.glob(os.path.join(hr_dir, f'*{ext}'))] # Use glob
            )
            if len(self.lr_image_paths) != len(self.hr_image_paths):
                raise ValueError("Training mode: Number of LR and HR images must match.")
            if self.text_descriptions and len(self.lr_image_paths) != len(self.text_descriptions):
                raise ValueError("Number of LR images and text descriptions must match.")
            # In train mode, if lr_patch_size is specified, upscale_factor must also be provided
            if self.lr_patch_size is not None and self.upscale_factor is None:
                 raise ValueError("Training mode with lr_patch_size requires upscale_factor.")
        elif self.mode == 'eval':
            if not val_lr_dir:
                 raise ValueError("Evaluation mode requires val_lr_dir.")
            self.lr_image_paths = sorted(
                [p for ext in self.supported_formats for p in glob.glob(os.path.join(val_lr_dir, f'*{ext}'))] # Use glob
            )
            # HR images are optional for evaluation, but needed for metrics
            self.hr_image_paths = sorted(
                [p for ext in self.supported_formats for p in glob.glob(os.path.join(val_hr_dir, f'*{ext}'))] if val_hr_dir and os.path.exists(val_hr_dir) else []
            )
            if self.text_descriptions and len(self.lr_image_paths) != len(self.text_descriptions):
                raise ValueError("Number of LR images and text descriptions must match.")
            # In eval mode, if HR paths are provided, they must match LR paths
            if self.hr_image_paths and len(self.lr_image_paths) != len(self.hr_image_paths):
                 print("Warning: Number of LR and HR images mismatch in evaluation mode. Metrics requiring HR will be skipped.")
                 self.hr_image_paths = [] # Clear HR paths if mismatch
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose 'train' or 'eval'.")

        # Add resize transform if patch size is specified
        # Add resize transform if patch size is specified
        # Note: This is for resizing the whole image, not patching.
        # Patching logic will be handled in __getitem__ for training mode.
        # If image_size is specified, it overrides patch_size for evaluation resizing
        if self.image_size is not None:
             # Resize to a fixed size if specified
             self.lr_resize_transform = transforms.Resize((self.image_size, self.image_size), Image.BICUBIC)
             if self.upscale_factor is None:
                  raise ValueError("upscale_factor must be provided if image_size is specified.")
             self.hr_resize_transform = transforms.Resize((self.image_size * self.upscale_factor, self.image_size * self.upscale_factor), Image.BICUBIC)
        elif self.lr_patch_size is not None and self.mode == 'eval':
            # Only resize for evaluation if patch size is specified (legacy behavior if image_size is not used)
            self.lr_resize_transform = transforms.Resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
            if self.upscale_factor is None:
                 raise ValueError("upscale_factor must be provided if lr_patch_size is specified for evaluation.")
            self.hr_resize_transform = transforms.Resize((self.lr_patch_size * self.upscale_factor, self.lr_patch_size * self.upscale_factor), Image.BICUBIC)
        else:
            self.lr_resize_transform = None
            self.hr_resize_transform = None

        print(f"Initialized SRDataset in '{self.mode}' mode with {len(self.lr_image_paths)} LR images.")
        if self.mode == 'train':
            print(f"Found {len(self.hr_image_paths)} HR images.")
        elif self.mode == 'eval' and self.hr_image_paths:
             print(f"Found {len(self.hr_image_paths)} HR images for evaluation metrics.")
        if self.edge_methods:
            print(f"Edge detection methods enabled: {self.edge_methods}")

    def __len__(self):
        return len(self.lr_image_paths)

    def __getitem__(self, idx):
        lr_img_path = self.lr_image_paths[idx]
        lr_img = self._load_image(lr_img_path)

        text_desc = None # Initialize text_desc to None

        # Load HR image if available (only required for train mode or eval with metrics)
        hr_img = None
        hr_tensor_pil = None # Initialize hr_tensor_pil
        if self.mode == 'train' or (self.mode == 'eval' and self.hr_image_paths):
             hr_img_path = self.hr_image_paths[idx]
             hr_img = self._load_image(hr_img_path)
             
             # Ensure HR and LR images have compatible dimensions
             # Check if upscale_factor is not None before multiplication
             if self.upscale_factor is not None and hr_img.size != (lr_img.size[0] * self.upscale_factor, lr_img.size[1] * self.upscale_factor):
                 hr_img = hr_img.resize(
                     (lr_img.size[0] * self.upscale_factor, lr_img.size[1] * self.upscale_factor),
                     Image.BICUBIC
                 )

        # Apply patch extraction for training mode OR resize if image_size is specified
        if self.mode == 'train' and self.lr_patch_size is not None and hr_img is not None:
            if self.upscale_factor is None:
                 raise ValueError("upscale_factor must be provided for training with patching.")

            # Ensure LR and HR images have compatible sizes for patching
            lr_width, lr_height = lr_img.size
            hr_width, hr_height = hr_img.size

            # Calculate HR patch size based on LR patch size and upscale factor
            hr_patch_size = self.lr_patch_size * self.upscale_factor

            # Ensure images are large enough for patching
            if lr_width < self.lr_patch_size or lr_height < self.lr_patch_size or \
               hr_width < hr_patch_size or hr_height < hr_patch_size:
                # If image is smaller than patch size, pad it with reflection
                print(f"Warning: Image {lr_img_path} is smaller than patch size. Padding.")
                
                # Calculate padding amounts
                pad_width = max(0, self.lr_patch_size - lr_width)
                pad_height = max(0, self.lr_patch_size - lr_height)
                
                # Pad LR image
                lr_img = transforms.Pad((pad_width//2, pad_height//2, 
                                       pad_width - pad_width//2, pad_height - pad_height//2), 
                                      padding_mode='reflect')(lr_img)
                
                # Pad HR image
                hr_pad_width = pad_width * self.upscale_factor
                hr_pad_height = pad_height * self.upscale_factor
                hr_img = transforms.Pad((hr_pad_width//2, hr_pad_height//2, 
                                       hr_pad_width - hr_pad_width//2, hr_pad_height - hr_pad_height//2), 
                                      padding_mode='reflect')(hr_img)
                
                lr_width, lr_height = lr_img.size
                hr_width, hr_height = hr_img.size

            # Randomly select top-left corner of the patch in LR image
            # The maximum possible top-left corner is such that the patch fits within the image
            max_x = lr_width - self.lr_patch_size
            max_y = lr_height - self.lr_patch_size

            # Handle cases where max_x or max_y might be negative if image is smaller than patch size (should be handled by resize above)
            if max_x < 0 or max_y < 0:
                 raise ValueError(f"Image {lr_img_path} is too small for patching even after resize attempt.")

            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)

            # Crop LR patch
            lr_patch = lr_img.crop((x, y, x + self.lr_patch_size, y + self.lr_patch_size))

            # Calculate corresponding HR patch coordinates
            hr_x = x * self.upscale_factor
            hr_y = y * self.upscale_factor

            # Crop HR patch
            hr_patch = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))

            # Apply transform (ToTensor) to patches
            lr_tensor_pil = self.transform(lr_patch) if self.transform else transforms.ToTensor()(lr_patch)
            hr_tensor_pil = self.transform(hr_patch) if self.transform else transforms.ToTensor()(hr_patch)

            # Get text description for the patch if available
            if self.text_descriptions:
                 # Assuming text descriptions are per image, not per patch. Use the description for the original image.
                 text_desc = self.text_descriptions[idx]

        # Add resizing logic for when not using patching but image_size is specified
        elif self.image_size is not None:
             # Apply resize transform to full images
             if self.lr_resize_transform:
                 lr_img = self.lr_resize_transform(lr_img)
             if hr_img is not None and self.hr_resize_transform:
                 hr_img = self.hr_resize_transform(hr_img)

             # Apply transform (ToTensor) to full images
             lr_tensor_pil = self.transform(lr_img) if self.transform else transforms.ToTensor()(lr_img)
             # If hr_img is None, set hr_tensor_pil to an empty tensor
             hr_tensor_pil = self.transform(hr_img) if hr_img is not None and self.transform else (transforms.ToTensor()(hr_img) if hr_img is not None else torch.empty(0))

             # Get text description for the image if available
             if self.text_descriptions:
                 text_desc = self.text_descriptions[idx]

        else:
            # Default case (e.g., no patching/resizing specified)
            lr_tensor_pil = self.transform(lr_img) if self.transform else transforms.ToTensor()(lr_img)
            # If hr_img is None, set hr_tensor_pil to an empty tensor
            hr_tensor_pil = self.transform(hr_img) if hr_img is not None and self.transform else (transforms.ToTensor()(hr_img) if hr_img is not None else torch.empty(0))

            # Get text description if available
            if self.text_descriptions:
                 text_desc = self.text_descriptions[idx]

        lr_combined = lr_tensor_pil # Initialize lr_combined with the processed LR tensor (patch or full image)

        # Load and concatenate edge information if specified
        if self.edge_methods:
            edge_tensors = []
            lr_image_name = os.path.basename(lr_img_path)
            name_without_ext = os.path.splitext(lr_image_name)[0]
            # Edge data is now expected in subdirectories within the LR directory
            edge_data_base_dir = os.path.dirname(lr_img_path) # This is the LR directory

            for method in self.edge_methods:
                # Construct the expected edge file path
                edge_file_path = os.path.join(edge_data_base_dir, method, f"{name_without_ext}.npy")
                if not os.path.exists(edge_file_path):
                    raise FileNotFoundError(f"Pre-calculated edge file not found: {edge_file_path}. Please run pre-processing first.")
                edge_np = np.load(edge_file_path)
                # Convert numpy array (H, W) to tensor (1, H, W) and move to device
                # Assuming edge is uint8 [0, 255], convert to float [0, 1]
                edge_tensor = torch.from_numpy(edge_np).unsqueeze(0).float().to(self.device) / 255.0

                # If patching, crop the edge tensor to match the LR patch size
                if self.mode == 'train' and self.lr_patch_size is not None:
                     # Assuming edge tensor has the same spatial dimensions as the original LR image
                     # Need to crop the edge tensor using the same coordinates (x, y) as the LR patch
                     # edge_tensor shape is (1, H, W)
                     edge_tensor = edge_tensor[:, y:y + self.lr_patch_size, x:x + self.lr_patch_size,]

                # If not patching but image_size is specified, resize edge tensor
                elif self.image_size is not None:
                     # Resize edge tensor to match the resized LR image size
                     # Edge tensor is (1, H, W), need to resize H and W
                     # Use nearest neighbor or bilinear for edge maps, depending on type. Nearest is safer.
                     # Need to add batch dim for transforms.Resize
                     edge_tensor = transforms.Resize((self.image_size, self.image_size), Image.NEAREST)(edge_tensor.unsqueeze(0)).squeeze(0)

                edge_tensors.append(edge_tensor)

            if edge_tensors:
                # Concatenate original LR image with edge tensors along channel dimension
                lr_combined = torch.cat([lr_combined] + edge_tensors, dim=0)

        # Conditionally return text_desc
        if self.text_descriptions:
            return lr_combined, hr_tensor_pil, text_desc
        else:
            return lr_combined, hr_tensor_pil

    def _load_image(self, img_path):
        """Loads an image based on its extension."""
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.png', '.tiff', '.tif']:
            img = Image.open(img_path).convert('RGB')
        elif ext == '.npz':
            # 假设npz文件中的图像数据存储在名为 'image' 的键下
            # 或者根据您的NPZ结构调整键名，例如 'lr' 或 'hr'
            # 这里我们假设图像数据直接是数组
            data = np.load(img_path)
            if 'image' in data:
                img_array = data['image']
            elif 'lr' in data: # 尝试常见的键名
                img_array = data['lr']
            elif 'hr' in data:
                img_array = data['hr']
            else: # 如果没有特定键，尝试获取第一个数组
                img_array = data[list(data.keys())[0]]
            
            # 确保图像数组是 HWC 格式且数据类型正确
            if img_array.ndim == 2: # 灰度图，添加通道维度
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.ndim == 3 and img_array.shape[0] == 3: # CHW to HWC
                img_array = np.transpose(img_array, (1, 2, 0))
            
            if img_array.dtype != np.uint8:
                if np.max(img_array) <= 1.0 and np.min(img_array) >=0.0: # 归一化到0-1的浮点数
                    img_array = (img_array * 255).astype(np.uint8)
                else: # 其他情况，直接转换
                    img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, 'RGB')
        else:
            raise ValueError(f"Unsupported image format: {ext} for {img_path}")
        return img

if __name__ == '__main__':
    # Example usage:
    data_root = './data_example'
    lr_example_dir = os.path.join(data_root, 'lr')
    hr_example_dir = os.path.join(data_root, 'hr')

    # 清理并创建示例数据目录
    if os.path.exists(data_root):
        import shutil
        shutil.rmtree(data_root)
    os.makedirs(lr_example_dir, exist_ok=True)
    os.makedirs(hr_example_dir, exist_ok=True)

    # 生成一些示例 PNG 图像
    for i in range(2):
        hr_img_np = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        hr_img = Image.fromarray(hr_img_np, 'RGB')
        lr_img = hr_img.resize((32, 32), Image.BICUBIC)
        hr_img.save(os.path.join(hr_example_dir, f'hr_example_{i}.png'))
        lr_img.save(os.path.join(lr_example_dir, f'lr_example_{i}.png'))

    # 生成一些示例 NPZ 文件 (假设包含 'image' 键)
    for i in range(2):
        hr_img_np = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        lr_img_np = np.array(Image.fromarray(hr_img_np, 'RGB').resize((32, 32), Image.BICUBIC))
        np.savez(os.path.join(hr_example_dir, f'hr_example_npz_{i}.npz'), image=hr_img_np)
        np.savez(os.path.join(lr_example_dir, f'lr_example_npz_{i}.npz'), image=lr_img_np)

    print(f"Generated example data in {data_root}")

    # 示例文本描述
    sample_texts = [
        "magnification: 2x, pixel size: 0.5um, content: urban scene",
        "magnification: 2x, pixel size: 0.5um, content: natural landscape",
        "magnification: 2x, pixel size: 0.5um, content: portrait",
        "magnification: 2x, pixel size: 0.5um, content: abstract art"
    ]

    # Define transforms
    image_transform = transforms.ToTensor()

    # Create dataset for training
    print("\n--- Training Mode Dataset ---")
    train_dataset = SRDataset(lr_example_dir, hr_example_dir, text_descriptions=sample_texts, transform=image_transform, mode='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    print(f"Training Dataset size: {len(train_dataset)}")
    for lr_batch, hr_batch, text_batch in train_dataloader:
        print(f"LR batch shape: {lr_batch.shape}")
        print(f"HR batch shape: {hr_batch.shape}")
        print(f"Text batch: {text_batch}")
        break

    # Create dataset for evaluation
    print("\n--- Evaluation Mode Dataset ---")
    # 对于评估，我们通常只有LR图像和文本
    eval_lr_paths = [os.path.join(lr_example_dir, f) for f in os.listdir(lr_example_dir)]
    eval_texts = sample_texts[:len(eval_lr_paths)] # 确保文本数量匹配

    eval_dataset = SRDataset(lr_example_dir, None, text_descriptions=eval_texts, transform=image_transform, mode='eval')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=2, shuffle=False)

    print(f"Evaluation Dataset size: {len(eval_dataset)}")
    for lr_batch, text_batch in eval_dataloader:
        print(f"LR batch shape: {lr_batch.shape}")
        print(f"Text batch: {text_batch}")
        break

    # 清理示例数据
    # import shutil
    # shutil.rmtree(data_root)
    # print(f"\nCleaned up example data directory: {data_root}")
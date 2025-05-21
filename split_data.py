import os
import shutil
import random
import glob

def split_data(lr_source_dir, hr_source_dir, output_base_dir, split_ratio=0.8):
    """Splits LR and HR image pairs into training and validation sets."""

    train_lr_dir = os.path.join(output_base_dir, 'train', 'lr')
    train_hr_dir = os.path.join(output_base_dir, 'train', 'hr')
    val_lr_dir = os.path.join(output_base_dir, 'val', 'lr')
    val_hr_dir = os.path.join(output_base_dir, 'val', 'hr')

    # Create output directories if they don't exist
    for d in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
        os.makedirs(d, exist_ok=True)

    # Get list of LR image paths
    supported_formats = ('.png', '.tiff', '.tif', '.npz')
    lr_image_paths = sorted(
        [p for ext in supported_formats for p in glob.glob(os.path.join(lr_source_dir, f'*{ext}'))]
    )

    if not lr_image_paths:
        print(f"No images found in LR source directory: {lr_source_dir}")
        return

    # Get list of LR image paths
    supported_formats = ('.png', '.tiff', '.tif', '.npz')
    lr_image_paths = sorted(
        [p for ext in supported_formats for p in glob.glob(os.path.join(lr_source_dir, f'*{ext}'))]
    )

    if not lr_image_paths:
        print(f"No images found in LR source directory: {lr_source_dir}")
        return

    # Build pairs based on filename index (assuming lr_0000.png matches hr_0000.png)
    matched_pairs = []
    for lr_path in lr_image_paths:
        lr_filename = os.path.basename(lr_path)
        # Assuming filename format is prefix_index.ext (e.g., lr_0000.png)
        parts = os.path.splitext(lr_filename)[0].split('_')
        if len(parts) > 1:
            index_str = parts[-1] # Get the index part
            hr_filename = f"hr_{index_str}{os.path.splitext(lr_filename)[1]}" # Construct HR filename
            hr_path = os.path.join(hr_source_dir, hr_filename)
            if os.path.exists(hr_path):
                matched_pairs.append((lr_path, hr_path))
            else:
                 print(f"Warning: Corresponding HR image not found for {lr_path} at expected path {hr_path}. Skipping pair.")
        else:
            print(f"Warning: Could not parse index from LR filename {lr_filename}. Skipping.")

    if not matched_pairs:
        print("No matching LR/HR pairs found based on filename index.")
        return

    print(f"Found {len(matched_pairs)} matching LR/HR pairs.")

    # Shuffle pairs
    random.shuffle(matched_pairs)

    # Calculate split point
    split_point = int(len(matched_pairs) * split_ratio)

    # Split pairs
    train_pairs = matched_pairs[:split_point]
    val_pairs = matched_pairs[split_point:]

    print(f"Splitting {len(matched_pairs)} pairs: {len(train_pairs)} for training, {len(val_pairs)} for validation.")

    # Copy files to train/val directories
    for lr_src, hr_src in train_pairs:
        lr_dest = os.path.join(train_lr_dir, os.path.basename(lr_src))
        hr_dest = os.path.join(train_hr_dir, os.path.basename(hr_src))
        shutil.copy2(lr_src, lr_dest)
        shutil.copy2(hr_src, hr_dest)

    for lr_src, hr_src in val_pairs:
        lr_dest = os.path.join(val_lr_dir, os.path.basename(lr_src))
        hr_dest = os.path.join(val_hr_dir, os.path.basename(hr_src))
        shutil.copy2(lr_src, lr_dest)
        shutil.copy2(hr_src, hr_dest)

    print("Data splitting complete.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split LR/HR image data into train/val sets.')
    parser.add_argument('--lr_source_dir', type=str, required=True, help='Source directory for low-resolution images')
    parser.add_argument('--hr_source_dir', type=str, required=True, help='Source directory for high-resolution images')
    parser.add_argument('--output_base_dir', type=str, default='./data/sample_split', help='Base directory to save the split train/val data')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of data to use for training (e.g., 0.8 for 80% train, 20% val)')

    args = parser.parse_args()

    split_data(args.lr_source_dir, args.hr_source_dir, args.output_base_dir, args.split_ratio)
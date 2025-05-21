import os
import zipfile
import shutil
import re

# 源目录，存放下载的zip文件
SOURCE_DIR = './data_download_temp/'
# 目标目录，存放解压和整理后的数据
TARGET_DIR_BASE = './data/DIV2K/'

# 创建目标目录如果它不存在
if not os.path.exists(TARGET_DIR_BASE):
    os.makedirs(TARGET_DIR_BASE)
    print(f"Created base directory: {TARGET_DIR_BASE}")

def get_target_subdir_and_content_folder_name(zip_filename):
    """
    根据zip文件名确定目标子目录和解压后内容应该在的文件夹名。
    返回 (target_subdirectory, expected_content_folder_name_in_zip)
    expected_content_folder_name_in_zip 如果zip直接包含文件则为 None
    """
    # DIV2K_train_HR.zip -> (TARGET_DIR_BASE/DIV2K_train_HR, DIV2K_train_HR)
    # DIV2K_valid_LR_bicubic_X2.zip -> (TARGET_DIR_BASE/DIV2K_valid_LR_bicubic/X2, DIV2K_valid_LR_bicubic/X2)
    # DIV2K_train_LR_mild.zip -> (TARGET_DIR_BASE/DIV2K_train_LR_mild, DIV2K_train_LR_mild)
    # DIV2K_train_LR_unknown_X4.zip -> (TARGET_DIR_BASE/DIV2K_train_LR_unknown/X4, DIV2K_train_LR_unknown/X4)
    # DIV2K_train_LR_x8.zip -> (TARGET_DIR_BASE/DIV2K_train_LR_x8, DIV2K_train_LR_x8)

    base_name = zip_filename.replace('.zip', '') # e.g., DIV2K_train_HR

    if base_name.endswith('_HR'): # e.g., DIV2K_train_HR
        target_subdir = os.path.join(TARGET_DIR_BASE, base_name)
        # HR zip 通常直接包含图片，或者包含一个与zip同名的文件夹
        # 我们假设解压后，内容应该直接在 target_subdir 下，或者在 target_subdir/base_name 下
        return target_subdir, base_name 

    # 匹配 LR 类型，例如 LR_bicubic, LR_unknown, LR_mild, LR_difficult, LR_wild, LR_x8
    # DIV2K_train_LR_bicubic_X2 -> (TARGET_DIR_BASE/DIV2K_train_LR_bicubic/X2, DIV2K_train_LR_bicubic/X2)
    # DIV2K_train_LR_mild -> (TARGET_DIR_BASE/DIV2K_train_LR_mild, DIV2K_train_LR_mild)
    # DIV2K_train_LR_x8 -> (TARGET_DIR_BASE/DIV2K_train_LR_x8, DIV2K_train_LR_x8)
    
    # 尝试匹配带有缩放因子的 LR (bicubic, unknown)
    # e.g., DIV2K_train_LR_bicubic_X2 or DIV2K_valid_LR_unknown_X4
    match_lr_scale = re.match(r'(DIV2K_(train|valid)_LR_(bicubic|unknown))_X([234])', base_name)
    if match_lr_scale:
        lr_type_base = match_lr_scale.group(1) # DIV2K_train_LR_bicubic
        scale = f'X{match_lr_scale.group(4)}'   # X2
        target_subdir = os.path.join(TARGET_DIR_BASE, lr_type_base, scale)
        # zip内部文件夹结构可能是 DIV2K_train_LR_bicubic/X2 或者直接是 X2 (如果zip是针对特定scale的)
        # 更常见的是 DIV2K_train_LR_bicubic_X2.zip 解压出 DIV2K_train_LR_bicubic/X2
        # 或者 DIV2K_train_LR_unknown_X4.zip 解压出 DIV2K_train_LR_unknown/X4
        # 我们期望内容在 target_subdir 下，zip内文件夹可能是 lr_type_base/scale
        return target_subdir, os.path.join(lr_type_base, scale)

    # 尝试匹配其他类型的 LR (difficult, mild, wild, x8)
    # e.g., DIV2K_train_LR_mild or DIV2K_valid_LR_x8
    match_lr_other = re.match(r'(DIV2K_(train|valid)_LR_(difficult|mild|wild|x8))', base_name)
    if match_lr_other:
        target_subdir = os.path.join(TARGET_DIR_BASE, base_name)
        # 这些zip通常解压后得到一个与zip同名的文件夹
        return target_subdir, base_name

    print(f"Warning: Could not determine target subdirectory logic for {zip_filename}. Using default.")
    # 默认行为：目标子目录与zip文件名（无扩展名）相同，期望内容也在同名文件夹下
    target_subdir = os.path.join(TARGET_DIR_BASE, base_name)
    return target_subdir, base_name

def main():
    print(f"Scanning for zip files in: {SOURCE_DIR}")
    zip_files_found = False
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith('.zip') and filename.startswith('DIV2K_'):
            zip_files_found = True
            zip_filepath = os.path.join(SOURCE_DIR, filename)
            print(f"\nProcessing: {filename}")
            
            target_subdir, expected_content_folder_name = get_target_subdir_and_content_folder_name(filename)
            
            if not target_subdir:
                print(f"Skipping {filename} as target subdirectory could not be determined.")
                continue

            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)
                print(f"Created directory: {target_subdir}")
            else:
                print(f"Directory already exists: {target_subdir}. Clearing content if any.")
                # 清理目标目录内容，以防重复运行脚本导致文件冲突
                for item in os.listdir(target_subdir):
                    item_path = os.path.join(target_subdir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

            try:
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    members = zip_ref.namelist()
                    print(f"Extracting to: {target_subdir}")

                    # 检查zip文件内部结构
                    # 理想情况是，zip解压后，所有文件都在一个名为 expected_content_folder_name 的文件夹内
                    # 或者，如果 expected_content_folder_name 是 None (或简单名称)，文件直接在根目录
                    
                    # 创建一个临时解压目录
                    temp_extract_path = os.path.join(TARGET_DIR_BASE, f"_temp_extract_{filename.replace('.zip','')}")
                    if os.path.exists(temp_extract_path):
                        shutil.rmtree(temp_extract_path)
                    os.makedirs(temp_extract_path)

                    zip_ref.extractall(temp_extract_path)
                    print(f"Extracted {filename} to temporary directory {temp_extract_path}")

                    # 确定实际的内容源路径
                    # 检查 temp_extract_path 下是否有名为 expected_content_folder_name 的文件夹
                    # 或者，如果zip内部有多个文件夹，但我们只关心一个特定的 (如 DIV2K_train_HR)
                    
                    actual_content_source_dir = temp_extract_path
                    # 尝试找到期望的文件夹
                    potential_source_dir = os.path.join(temp_extract_path, expected_content_folder_name)
                    if os.path.isdir(potential_source_dir):
                        actual_content_source_dir = potential_source_dir
                        print(f"Found expected content folder: {potential_source_dir}")
                    else:
                        # 如果期望的文件夹不存在，检查zip解压后是否只有一个顶级文件夹
                        extracted_items = os.listdir(temp_extract_path)
                        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract_path, extracted_items[0])):
                            actual_content_source_dir = os.path.join(temp_extract_path, extracted_items[0])
                            print(f"Found single top-level folder: {actual_content_source_dir}")
                        else:
                            print(f"Content seems to be directly in {temp_extract_path} or structure is unexpected.")
                            # 此时 actual_content_source_dir 仍然是 temp_extract_path

                    # 将内容从 actual_content_source_dir 移动到 target_subdir
                    if os.path.isdir(actual_content_source_dir):
                        for item_name in os.listdir(actual_content_source_dir):
                            s = os.path.join(actual_content_source_dir, item_name)
                            d = os.path.join(target_subdir, item_name)
                            # 确保目标不存在，或者先删除
                            if os.path.exists(d):
                                if os.path.isdir(d):
                                    shutil.rmtree(d)
                                else:
                                    os.remove(d)
                            shutil.move(s, d)
                        print(f"Moved contents from {actual_content_source_dir} to {target_subdir}")
                    else:
                        print(f"Error: Source content directory {actual_content_source_dir} not found or not a directory.")

                    # 清理临时解压目录
                    shutil.rmtree(temp_extract_path)
                    print(f"Cleaned up temporary directory {temp_extract_path}")
                    print(f"Successfully processed and organized {filename}")

            except zipfile.BadZipFile:
                print(f"Error: {filename} is a bad zip file.")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
                # 如果出错，也尝试清理临时目录
                if 'temp_extract_path' in locals() and os.path.exists(temp_extract_path):
                    shutil.rmtree(temp_extract_path)
    
    if not zip_files_found:
        print(f"No DIV2K zip files found in {SOURCE_DIR}. Please ensure your .zip files are in this directory and start with 'DIV2K_'.")
    else:
        print("\nAll DIV2K zip files processed.")
        print(f"Data should now be organized in: {TARGET_DIR_BASE}")

if __name__ == '__main__':
    main()
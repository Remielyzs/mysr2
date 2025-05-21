import os
import sys
import glob

# Add the parent directory to the sys.path to be able to import train
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from models.simple_srcnn import SimpleSRCNN # Assuming SimpleSRCNN is the model to train
import torch.nn as nn
from losses import L1Loss, CombinedLoss, EdgeLoss # Assuming losses are defined in losses.py
import torch
from preprocess_edges import preprocess_edge_data # Import the pre-processing function

# Define the base parameters for training
# Define the base parameters for training
# Update data directories to point to the split data
# TRAIN_LR_DIR = '/home/remiel/project/mysr2/data/split_sample/train/lr'
# TRAIN_HR_DIR = '/home/remiel/project/mysr2/data/split_sample/train/hr'
# VAL_LR_DIR = '/home/remiel/project/mysr2/data/split_sample/val/lr'
# VAL_HR_DIR = '/home/remiel/project/mysr2/data/split_sample/val/hr'
TRAIN_LR_DIR = './data/DIV2K/DIV2K_train_LR_bicubic/X2'
TRAIN_HR_DIR = './data/DIV2K/DIV2K_train_HR'
VAL_LR_DIR = './data/DIV2K/DIV2K_valid_LR_bicubic/X2'
VAL_HR_DIR = './data/DIV2K/DIV2K_valid_HR'

BASE_TRAIN_PARAMS = {
    'model_class': SimpleSRCNN,
    'epochs': 1, # Reduced epochs for faster testing
    'batch_size': 128,
    'learning_rate': 0.001,
    'lr_data_dir': TRAIN_LR_DIR,
    'hr_data_dir': TRAIN_HR_DIR,
    'use_text_descriptions': False,
    'criterion': nn.MSELoss(), # Default loss
    'results_base_dir': 'results_edge_experiments',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'upscale_factor': 2, # Example upscale factor
    'lr_patch_size': 48 # Define LR patch size for training
}

# Define different edge detection method combinations to test
EDGE_COMBINATIONS = [
    [], # No edge detection
    ['sobel'], # Sobel edge detection
    ['canny'], # Canny edge detection
    ['laplacian'], # Laplacian edge detection
    ['sobel', 'canny'], # Both Sobel and Canny
    ['sobel', 'laplacian'], # Sobel and Laplacian
    ['canny', 'laplacian'], # Canny and Laplacian
    ['sobel', 'canny', 'laplacian'] # Sobel, Canny, and Laplacian
]

# Define different loss functions to potentially use with edges
# This is just an example, you might want to combine these with MSELoss
LOSS_FUNCTIONS = {
    'mse': nn.MSELoss(),
    'l1': L1Loss(),
    'sobel_edge_loss': EdgeLoss(edge_detector_type='sobel'),
    'canny_edge_loss': EdgeLoss(edge_detector_type='canny'),
    'laplacian_edge_loss': EdgeLoss(edge_detector_type='laplacian'),
}

# Example of how to define training runs with specific loss functions
TRAINING_RUNS = [
    {
        'name': 'no_edge_mse',
        'edge_methods': [],
        'criterion': LOSS_FUNCTIONS['mse'],
    },
    {
        'name': 'sobel_edge_mse',
        'edge_methods': ['sobel'],
        'criterion': LOSS_FUNCTIONS['mse'],
    },
     {
        'name': 'canny_edge_mse',
        'edge_methods': ['canny'],
        'criterion': LOSS_FUNCTIONS['mse'],
    },
    {
        'name': 'sobel_canny_edge_mse',
        'edge_methods': ['sobel', 'canny'],
        'criterion': LOSS_FUNCTIONS['mse'],
    },
    # Example with combined loss (MSE + EdgeLoss)
    {
        'name': 'sobel_edge_combined_loss',
        'edge_methods': ['sobel'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['sobel_edge_loss'], 0.1) # Add Sobel edge loss with weight 0.1
        ]),
    },
     {
        'name': 'canny_edge_combined_loss',
        'edge_methods': ['canny'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['canny_edge_loss'], 0.1) # Add Canny edge loss with weight 0.1
        ]),
    },
     {
        'name': 'sobel_canny_edge_combined_loss',
        'edge_methods': ['sobel', 'canny'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['sobel_edge_loss'], 0.1),
            (LOSS_FUNCTIONS['canny_edge_loss'], 0.1)
        ]),
    },
    {
        'name': 'laplacian_edge_mse',
        'edge_methods': ['laplacian'],
        'criterion': LOSS_FUNCTIONS['mse'],
    },
    {
        'name': 'laplacian_edge_combined_loss',
        'edge_methods': ['laplacian'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
        ]),
    },
    {
        'name': 'sobel_laplacian_edge_combined_loss',
        'edge_methods': ['sobel', 'laplacian'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['sobel_edge_loss'], 0.1),
            (LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
        ]),
    },
    {
        'name': 'canny_laplacian_edge_combined_loss',
        'edge_methods': ['canny', 'laplacian'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['canny_edge_loss'], 0.1),
            (LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
        ]),
    },
    {
        'name': 'sobel_canny_laplacian_edge_combined_loss',
        'edge_methods': ['sobel', 'canny', 'laplacian'],
        'criterion': CombinedLoss([
            (LOSS_FUNCTIONS['mse'], 1.0),
            (LOSS_FUNCTIONS['sobel_edge_loss'], 0.1),
            (LOSS_FUNCTIONS['canny_edge_loss'], 0.1),
            (LOSS_FUNCTIONS['laplacian_edge_loss'], 0.1)
        ]),
    },
]



if __name__ == '__main__':
    print("Starting edge detection experiments...")

    # Collect all unique edge methods required by the training runs
    required_edge_methods = set()
    for run_config in TRAINING_RUNS:
        if run_config['edge_methods']:
            required_edge_methods.update(run_config['edge_methods'])

    required_edge_methods = list(required_edge_methods)

    # Pre-process edge data if needed
    if required_edge_methods:
        print(f"Checking and pre-processing required edge data: {required_edge_methods}")
        
        # Define the directories for training and validation LR data
        train_lr_dir = TRAIN_LR_DIR
        val_lr_dir = VAL_LR_DIR

        # Process training data edges
        print(f"Processing edges for training data in {train_lr_dir}...")
        preprocess_edge_data(
            input_dir=train_lr_dir,
            edge_methods=required_edge_methods,
            device=BASE_TRAIN_PARAMS['device'] # Use the same device as training
        )

        # Process validation data edges if the directory exists
        if os.path.exists(val_lr_dir):
            print(f"Processing edges for validation data in {val_lr_dir}...")
            preprocess_edge_data(
                input_dir=val_lr_dir,
                edge_methods=required_edge_methods,
                device=BASE_TRAIN_PARAMS['device'] # Use the same device as training
            )
        else:
            print(f"Warning: Validation LR directory {val_lr_dir} not found. Skipping edge pre-processing for validation data.")

    for run_config in TRAINING_RUNS:
        run_name = run_config['name']
        edge_methods = run_config['edge_methods']
        criterion = run_config['criterion']

        print(f"\n--- Running experiment: {run_name} with edge methods: {edge_methods} ---")

        # Combine base parameters with run-specific parameters
        current_train_params = BASE_TRAIN_PARAMS.copy()
        current_train_params['model_name'] = run_name
        current_train_params['edge_detection_methods'] = edge_methods
        current_train_params['criterion'] = criterion

        # Pass explicit validation data directories
        current_train_params['lr_data_dir'] = TRAIN_LR_DIR
        current_train_params['hr_data_dir'] = TRAIN_HR_DIR
        current_train_params['val_lr_data_dir'] = VAL_LR_DIR
        current_train_params['val_hr_data_dir'] = VAL_HR_DIR

        # Ensure edge_detection_methods is passed correctly
        current_train_params['edge_detection_methods'] = edge_methods
        current_train_params['lr_patch_size'] = BASE_TRAIN_PARAMS['lr_patch_size'] # Pass patch size
        current_train_params['upscale_factor'] = BASE_TRAIN_PARAMS['upscale_factor'] # Pass upscale factor

        try:
            train_model(**current_train_params)
            print(f"Experiment '{run_name}' completed successfully.")
        except Exception as e:
            print(f"Error running experiment '{run_name}': {e}")

    print("All edge detection experiments finished.")
import torch
import torch.nn as nn
import torch.nn.functional as F
from image_processing import EdgeDetector

class MSELoss(nn.Module):
    """Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)

class L1Loss(nn.Module):
    """L1 Loss (Mean Absolute Error)"""
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)

class CombinedLoss(nn.Module):
    """A loss function that combines multiple loss terms with weights."""
    def __init__(self, losses_and_weights):
        """
        Args:
            losses_and_weights (list of tuples): A list where each tuple contains
                                                 (loss_function_instance, weight).
                                                 Example: [(MSELoss(), 1.0), (L1Loss(), 0.5)]
        """
        super().__init__()
        self.losses_and_weights = losses_and_weights

    def forward(self, prediction, target, **kwargs):
        total_loss = 0
        individual_losses = {}
        for loss_fn, weight in self.losses_and_weights:
            # Some loss functions might require additional arguments (e.g., for physical constraints)
            # Pass kwargs to allow flexibility
            current_loss = loss_fn(prediction, target, **kwargs)
            total_loss += weight * current_loss
            individual_losses[loss_fn.__class__.__name__] = current_loss.item()
        
        # For logging or debugging, you might want to return individual losses too
        # For now, just return the total weighted loss
        return total_loss #, individual_losses

# Example of a custom loss that could incorporate a physical constraint (placeholder)
class EdgeLoss(nn.Module):
    """A placeholder for a loss that penalizes differences in edges (e.g., using Canny or Sobel)."""
    def __init__(self, edge_detector_type='sobel'):
        super().__init__()
        self.edge_detector_type = edge_detector_type
        # In a real implementation, you might initialize Sobel kernels here
        # For Canny, you might use a library function directly in forward

    def get_edges(self, image_batch):
        """Applies the specified edge detection filter."""
        # Convert to grayscale if needed
        if image_batch.shape[1] == 3: # Assuming CHW format
            image_batch_gray = 0.299 * image_batch[:, 0:1, :, :] + \
                               0.587 * image_batch[:, 1:2, :, :] + \
                               0.114 * image_batch[:, 2:3, :, :]
        else:
            image_batch_gray = image_batch

        if self.edge_detector_type == 'sobel':
            # Simple Sobel filter (example - not a full Canny implementation)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image_batch.dtype, device=image_batch.device).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image_batch.dtype, device=image_batch.device).unsqueeze(0).unsqueeze(0)
            
            # Apply Sobel filters
            grad_x = F.conv2d(image_batch_gray, sobel_x, padding=1)
            grad_y = F.conv2d(image_batch_gray, sobel_y, padding=1)
            
            edges = torch.sqrt(grad_x**2 + grad_y**2)
            return edges
        elif self.edge_detector_type == 'laplacian':
            # Simple Laplacian filter
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=image_batch.dtype, device=image_batch.device).unsqueeze(0).unsqueeze(0)
            
            # Apply Laplacian filter
            edges = torch.abs(F.conv2d(image_batch_gray, laplacian_kernel, padding=1))
            return edges
        elif self.edge_detector_type == 'canny':
             # Use the Canny implementation from EdgeDetector
             detector = EdgeDetector(device=image_batch.device)
             # The EdgeDetector expects input in [0, 1] range, which image_batch should be.
             # It returns a tensor in [0, 1] range.
             edges = detector.canny_edge(image_batch)
             return edges
        else:
            raise ValueError(f"Unsupported edge detector type: {self.edge_detector_type}")

    def forward(self, prediction, target, **kwargs):
        pred_edges = self.get_edges(prediction)
        target_edges = self.get_edges(target)
        
        # Use a standard loss (e.g., MSE or L1) on the edge maps
        loss = F.mse_loss(pred_edges, target_edges)
        return loss

if __name__ == '__main__':
    # Example Usage
    mse_loss = MSELoss()
    l1_loss = L1Loss()
    edge_loss_fn = EdgeLoss()

    # Example of combined loss
    combined_loss_fn = CombinedLoss([
        (mse_loss, 1.0), 
        (l1_loss, 0.5),
        (edge_loss_fn, 0.2) # Example of adding the edge loss
    ])

    # Dummy data
    prediction_tensor = torch.rand(2, 3, 64, 64) # Batch_size=2, Channels=3, H=64, W=64
    target_tensor = torch.rand(2, 3, 64, 64)

    # Calculate individual losses
    mse_val = mse_loss(prediction_tensor, target_tensor)
    l1_val = l1_loss(prediction_tensor, target_tensor)
    edge_val = edge_loss_fn(prediction_tensor, target_tensor)
    print(f"MSE Loss: {mse_val.item()}")
    print(f"L1 Loss: {l1_val.item()}")
    print(f"Edge Loss (example): {edge_val.item()}")

    # Calculate combined loss
    total_combined_loss = combined_loss_fn(prediction_tensor, target_tensor)
    print(f"Total Combined Loss: {total_combined_loss.item()}")

    print("\nTesting with single channel (grayscale-like) images for EdgeLoss directly:")
    prediction_gray = torch.rand(2, 1, 64, 64)
    target_gray = torch.rand(2, 1, 64, 64)
    edge_val_gray = edge_loss_fn(prediction_gray, target_gray)
    print(f"Edge Loss (grayscale): {edge_val_gray.item()}")
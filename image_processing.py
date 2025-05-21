import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

class EdgeDetector:
    def __init__(self, device='cpu'):
        """
        初始化边缘检测器。
        Args:
            device (str): 计算设备 ('cpu' 或 'cuda').
        """
        self.device = device

    def _to_numpy(self, image_tensor):
        """
        将PyTorch张量转换为适合OpenCV处理的NumPy数组。
        Args:
            image_tensor (torch.Tensor): 输入图像张量 (C, H, W) 或 (B, C, H, W)，范围 [0, 1]。
        Returns:
            np.ndarray: NumPy数组 (H, W, C)，范围 [0, 255]，数据类型 uint8。
        """
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0] # 取batch中的第一张图
        pil_image = to_pil_image(image_tensor.cpu())
        return np.array(pil_image)

    def _to_tensor(self, image_numpy):
        """
        将NumPy数组 (H, W) 或 (H, W, C) 转换为PyTorch张量 (1, H, W) 或 (C, H, W)。
        Args:
            image_numpy (np.ndarray): 输入图像NumPy数组。
        Returns:
            torch.Tensor: 输出图像张量，范围 [0, 1]。
        """
        return to_tensor(image_numpy).to(self.device)

    def sobel_edge(self, image_tensor, ksize=5):
        """
        使用Sobel算子进行边缘检测。
        Args:
            image_tensor (torch.Tensor): 输入图像张量 (C, H, W) 或 (B, C, H, W)。
            ksize (int): Sobel核的大小。
        Returns:
            torch.Tensor: 边缘图像张量 (1, H, W)。
        """
        img_np = self._to_numpy(image_tensor)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # 计算梯度幅值
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 归一化到 0-255
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max()) * 255
        edge_image = magnitude.astype(np.uint8)
        
        return self._to_tensor(edge_image)

    def canny_edge(self, image_tensor, threshold1=100, threshold2=200):
        """
        使用Canny算子进行边缘检测。
        Args:
            image_tensor (torch.Tensor): 输入图像张量 (C, H, W) 或 (B, C, H, W)。
            threshold1 (int): Canny算法的第一个阈值。
            threshold2 (int): Canny算法的第二个阈值。
        Returns:
            torch.Tensor: 边缘图像张量 (1, H, W)。
        """
        img_np = self._to_numpy(image_tensor)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return self._to_tensor(edges)

    def laplacian_edge(self, image_tensor, ksize=3):
        """
        使用Laplacian算子进行边缘检测。
        Args:
            image_tensor (torch.Tensor): 输入图像张量 (C, H, W) 或 (B, C, H, W)。
            ksize (int): Laplacian核的大小。
        Returns:
            torch.Tensor: 边缘图像张量 (1, H, W)。
        """
        img_np = self._to_numpy(image_tensor)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        # 取绝对值并转换为8位
        abs_laplacian = np.absolute(laplacian)
        edge_image = np.uint8(abs_laplacian)
        # 归一化处理，增强对比度
        if edge_image.max() > 0:
             edge_image = (edge_image / edge_image.max()) * 255
        edge_image = edge_image.astype(np.uint8)
        return self._to_tensor(edge_image)

    def detect(self, image_tensor, method='canny', **kwargs):
        """
        根据指定方法进行边缘检测。
        Args:
            image_tensor (torch.Tensor): 输入图像张量 (C, H, W) 或 (B, C, H, W)。
            method (str): 边缘检测方法 ('sobel', 'canny', 'laplacian')。
            **kwargs: 传递给特定边缘检测方法的参数。
        Returns:
            torch.Tensor: 边缘图像张量 (1, H, W)。
        Raises:
            ValueError: 如果指定了不支持的方法。
        """
        if method == 'sobel':
            return self.sobel_edge(image_tensor, **kwargs)
        elif method == 'canny':
            return self.canny_edge(image_tensor, **kwargs)
        elif method == 'laplacian':
            return self.laplacian_edge(image_tensor, **kwargs)
        else:
            raise ValueError(f"Unsupported edge detection method: {method}. Supported methods are 'sobel', 'canny', 'laplacian'.")

if __name__ == '__main__':
    # 创建一个虚拟输入张量 (batch_size=1, channels=3, height=64, width=64)
    # 确保张量值在 [0, 1] 范围内
    dummy_input_rgb = torch.rand(1, 3, 64, 64) 
    dummy_input_gray = torch.rand(1, 1, 64, 64) # 也可以测试灰度图输入，但内部会转灰度

    detector = EdgeDetector(device='cpu')

    print("Testing with RGB-like input (3 channels):")
    # Sobel
    sobel_edges = detector.detect(dummy_input_rgb, method='sobel', ksize=3)
    print(f"Sobel output shape: {sobel_edges.shape}, dtype: {sobel_edges.dtype}, min: {sobel_edges.min()}, max: {sobel_edges.max()}")

    # Canny
    canny_edges = detector.detect(dummy_input_rgb, method='canny', threshold1=50, threshold2=150)
    print(f"Canny output shape: {canny_edges.shape}, dtype: {canny_edges.dtype}, min: {canny_edges.min()}, max: {canny_edges.max()}")

    # Laplacian
    laplacian_edges = detector.detect(dummy_input_rgb, method='laplacian', ksize=3)
    print(f"Laplacian output shape: {laplacian_edges.shape}, dtype: {laplacian_edges.dtype}, min: {laplacian_edges.min()}, max: {laplacian_edges.max()}")

    # 测试不支持的方法
    try:
        detector.detect(dummy_input_rgb, method='unknown')
    except ValueError as e:
        print(f"Error caught as expected: {e}")

    # 可以选择性地可视化结果，如果环境支持
    # import matplotlib.pyplot as plt
    # def show_tensor_image(image_tensor, title=''):
    #     if image_tensor.ndim == 4:
    #         image_tensor = image_tensor[0]
    #     if image_tensor.shape[0] == 1: # 灰度图
    #         plt.imshow(to_pil_image(image_tensor.cpu()), cmap='gray')
    #     else: # 彩色图
    #         plt.imshow(to_pil_image(image_tensor.cpu()))
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.show()

    # show_tensor_image(dummy_input_rgb, title='Original RGB-like')
    # show_tensor_image(sobel_edges, title='Sobel Edges')
    # show_tensor_image(canny_edges, title='Canny Edges')
    # show_tensor_image(laplacian_edges, title='Laplacian Edges')

    print("\nEdge detection module created and tested.")
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

def get_gaussian_kernel(kernel_size=37, sigma=6):
    """Create a 2D Gaussian kernel."""
    # 1D Gaussian
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    # Outer product for 2D kernel
    kernel_2d = torch.outer(gauss, gauss)
    return kernel_2d

def apply_lowpass_gaussian(img, kernel):
    """Apply Gaussian filtering to BCHW tensor using depthwise conv."""
    B, C, H, W = img.shape
    kernel = kernel.to(img.device)
    kernel = kernel.view(1, 1, *kernel.shape)  # [1,1,k,k]
    kernel = kernel.repeat(C, 1, 1, 1)  # [C,1,k,k]
    
    padding = kernel.shape[-1] // 2
    return F.conv2d(img, kernel, padding=padding, groups=C)
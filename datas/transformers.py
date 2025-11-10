import torch
import torchvision.transforms as transforms
from typing import List

class Normalize3D:
    """3D数据归一化"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

class RandomFlip3D:
    """3D数据随机翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # 随机在深度、高度、宽度维度翻转
        if torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [1])  # 深度翻转
        if torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [2])  # 高度翻转
        if torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [3])  # 宽度翻转
        
        return tensor

class RandomRotate3D:
    """3D数据随机旋转（90度的倍数）"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            # 随机选择旋转角度（0, 90, 180, 270度）
            k = torch.randint(0, 4, (1,)).item()
            # 在高度和宽度维度旋转
            tensor = torch.rot90(tensor, k, dims=[2, 3])
        
        return tensor

def get_train_transforms() -> transforms.Compose:
    """获取训练数据变换"""
    return transforms.Compose([
        RandomFlip3D(p=0.5),
        RandomRotate3D(p=0.3),
        Normalize3D(mean=0.0, std=1.0)
    ])

def get_test_transforms() -> transforms.Compose:
    """获取测试数据变换"""
    return transforms.Compose([
        Normalize3D(mean=0.0, std=1.0)
    ])

def get_no_transforms() -> transforms.Compose:
    """获取无变换（恒等变换）"""
    return transforms.Compose([
        transforms.Lambda(lambda x: x)
    ])
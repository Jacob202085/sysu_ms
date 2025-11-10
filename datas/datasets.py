import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable

class Synthetic3DDataset(Dataset):
    """
    合成3D数据集
    用于训练和测试的虚拟数据生成
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 spatial_size: int = 32,
                 temporal_size: int = 16,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transform = transform
        self.target_transform = target_transform
        
        # 预生成数据（对于小数据集）或动态生成（对于大数据集）
        self.data = None
        self.targets = None
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.data is None:
            # 动态生成数据
            input_data = self._generate_sample()
            target_data = self._generate_target(input_data)
        else:
            # 使用预生成数据
            input_data = self.data[idx]
            target_data = self.targets[idx]
        
        # 应用变换
        if self.transform:
            input_data = self.transform(input_data)
        if self.target_transform:
            target_data = self.target_transform(target_data)
            
        return input_data, target_data
    
    def _generate_sample(self):
        """生成一个输入样本"""
        # 创建随机3D数据 [C, D, H, W]
        data = torch.randn(self.in_channels, self.temporal_size, 
                          self.spatial_size, self.spatial_size)
        return data
    
    def _generate_target(self, input_data):
        """根据输入生成目标数据"""
        # 这里可以定义更复杂的目标生成逻辑
        # 简单示例：添加一些噪声和变换
        target = input_data + 0.1 * torch.randn_like(input_data)
        return target
    
    def generate_all_data(self):
        """预生成所有数据"""
        self.data = []
        self.targets = []
        
        for i in range(self.num_samples):
            input_data = self._generate_sample()
            target_data = self._generate_target(input_data)
            self.data.append(input_data)
            self.targets.append(target_data)
        
        self.data = torch.stack(self.data)
        self.targets = torch.stack(self.targets)


class Custom3DDataset(Dataset):
    """
    自定义3D数据集
    用于加载真实数据（需要用户实现数据加载逻辑）
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        
        # 加载数据 - 需要根据实际数据格式实现
        data = self._load_3d_data(sample_path)
        
        if self.transform:
            data = self.transform(data)
            
        # 对于真实数据，可能需要返回输入和目标
        # 这里假设是自监督学习，输入和目标相同
        return data, data
    
    def _load_samples(self):
        """加载样本列表"""
        # 需要根据实际数据格式实现
        # 示例：返回数据文件路径列表
        data_files = list(self.data_dir.glob('*.npy'))  # 假设是numpy文件
        return data_files
    
    def _load_3d_data(self, file_path):
        """加载单个3D数据文件"""
        # 需要根据实际数据格式实现
        try:
            data = np.load(file_path)
            data = torch.from_numpy(data).float()
            return data
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return torch.randn(1, 16, 32, 32)  # 返回虚拟数据
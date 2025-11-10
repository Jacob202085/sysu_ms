import torch
from torch.utils.data import Dataset

class Synthetic3DDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机3D数据 [C, D, H, W]
        data = torch.randn(1, 16, 32, 32)
        target = data + 0.1 * torch.randn_like(data)  # 添加噪声作为目标
        return data, target
import torch
from torch.utils.data import DataLoader
from .datasets import Synthetic3DDataset, Custom3DDataset

def create_dataloader(dataset, 
                     batch_size: int = 4,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     **kwargs) -> DataLoader:
    """
    创建数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )

def create_synthetic_dataloader(config: dict) -> DataLoader:
    """
    创建合成数据加载器
    """
    dataset = Synthetic3DDataset(
        num_samples=config.get('num_samples', 1000),
        spatial_size=config.get('spatial_size', 32),
        temporal_size=config.get('temporal_size', 16),
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        transform=config.get('transform', None)
    )
    
    return create_dataloader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=config.get('shuffle', True)
    )

def create_train_val_dataloaders(train_config: dict, val_config: dict):
    """
    创建训练和验证数据加载器
    """
    train_loader = create_synthetic_dataloader(train_config)
    val_loader = create_synthetic_dataloader(val_config)
    
    return train_loader, val_loader
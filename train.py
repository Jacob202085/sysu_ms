#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from pathlib import Path

from models import create_model
from utils.config import get_train_args

def create_dummy_data(batch_size=2, spatial_size=32, temporal_size=16):
    """创建虚拟数据用于演示"""
    # 输入: [B, C, D, H, W]
    inputs = torch.randn(batch_size * 10, 1, temporal_size, spatial_size, spatial_size)
    # 目标: 与输入相同维度
    targets = torch.randn_like(inputs)
    
    dataset = TensorDataset(inputs, targets)
    return dataset

def train():
    """训练主函数"""
    args = get_train_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(
        model_name=args.model,
        in_channels=args.in_channels,
        out_channels=args.out_channels
    )
    model.to(device)
    print(f"创建模型: {args.model}")
    
    # 创建数据
    dataset = create_dummy_data(args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch: {epoch+1}/{args.epochs} '
                      f'Batch: {batch_idx}/{len(dataloader)} '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}')
    
    # 保存模型
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f'{args.model}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == '__main__':
    train()
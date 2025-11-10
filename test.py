#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from models import create_model
from utils.config import get_test_args

def test():
    """测试主函数"""
    args = get_test_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = create_model(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"加载模型: {args.model_path}")
    
    # 创建测试数据
    test_inputs = torch.randn(4, 1, 16, 32, 32)  # [B, C, D, H, W]
    test_dataset = TensorDataset(test_inputs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 测试
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            
            print(f"Batch {batch_idx}:")
            print(f"  输入维度: {data.shape}")
            print(f"  输出维度: {output.shape}")
            print(f"  输入输出维度是否相同: {data.shape == output.shape}")

if __name__ == '__main__':
    test()
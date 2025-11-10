import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

def get_train_args():
    """训练参数解析"""
    parser = argparse.ArgumentParser(description='训练3D卷积网络')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='./data', 
                       help='数据目录')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='批次大小')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='数据加载工作进程数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'basic'], help='模型类型')
    parser.add_argument('--in-channels', type=int, default=1,
                       help='输入通道数')
    parser.add_argument('--out-channels', type=int, default=1,
                       help='输出通道数')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='基础通道数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 保存参数
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='保存间隔（轮数）')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    return parser.parse_args()

def get_test_args():
    """测试参数解析"""
    parser = argparse.ArgumentParser(description='测试3D卷积网络')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'basic'], help='模型类型')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='测试设备')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='结果保存目录')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """保存配置到文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def create_default_config():
    """创建默认配置"""
    config = {
        'data': {
            'train': {
                'num_samples': 1000,
                'spatial_size': 32,
                'temporal_size': 16,
                'batch_size': 2
            },
            'val': {
                'num_samples': 200,
                'spatial_size': 32,
                'temporal_size': 16,
                'batch_size': 2
            }
        },
        'model': {
            'name': 'simple',
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 32
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'save_dir': './checkpoints',
            'log_dir': './logs'
        }
    }
    return config
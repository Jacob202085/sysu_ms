import torch
from models import SimpleConv3D
from data import Synthetic3DDataset, create_dataloader

def main():
    # 加载训练好的模型
    model = SimpleConv3D()
    model.load_state_dict(torch.load('checkpoints/simple_model.pth'))
    model.eval()
    
    # 创建测试数据
    test_dataset = Synthetic3DDataset(num_samples=20)
    test_loader = create_dataloader(test_dataset, batch_size=1)
    
    # 测试循环
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            print(f'Sample {i}: Input {data.shape}, Output {output.shape}')

if __name__ == '__main__':
    main()
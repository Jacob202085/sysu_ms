import torch
import torch.nn as nn
from models import SimpleConv3D
from data import Synthetic3DDataset, create_dataloader
from utils.logger import setup_logging, get_logger

def main():
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("开始训练...")
    model = SimpleConv3D()
    dataset = Synthetic3DDataset(num_samples=100)
    dataloader = create_dataloader(dataset, batch_size=2)
    
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        for data, target in dataloader:
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    logger.info("训练完成!")

if __name__ == '__main__':
    main()
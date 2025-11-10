from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
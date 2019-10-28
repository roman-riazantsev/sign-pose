from torch.utils.data import DataLoader
from config import CONFIG
from frei_dataset import FreiDataset
from utils import ToTensor

if __name__ == '__main__':
    dataset = FreiDataset(
        CONFIG['dataset_path'],
        transform=ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )

    dataset[0]

from torch.utils.data import DataLoader
import torch
from config import CONFIG
from frei_dataset import FreiDataset
from network_1 import Network1
from trainer import Trainer
from utils import ToTensor

if __name__ == '__main__':
    dataset = FreiDataset(
        CONFIG['dataset_path'],
        transform=ToTensor()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True
        # num_workers=4
    )

    model = Network1()
    trainer = Trainer(CONFIG['batch_size'], dataloader, model, 'results/network_1.pt')
    trainer.train(epochs=1000, save_rate=100)

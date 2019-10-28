from torch.utils.data import DataLoader
from config import CONFIG
from frei_dataset import FreiDataset
from network_0 import Network0
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
        shuffle=True,
        num_workers=4
    )

    network = Network0()
    trainer = Trainer(CONFIG['batch_size'], dataloader, network)
    trainer.train(5)

from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_v0 import CONFIG_V0
from datasets.frei_dataset_v0 import FreiHandVideoDataset
from models.model_u0 import ModelU0
from models.model_v0.model_v0 import ModelV0
from trainers.trainer_u0 import TrainerU0
from trainers.trainer_v0 import TrainerV0
from transforms.to_tensor_u0 import ToTensor

if __name__ == '__main__':
    config = CONFIG_V0

    build_id = config['build_id']

    dataset = FreiHandVideoDataset(
        config['dataset_path'],
        config['data_version'],
        transform=ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    model = ModelV0()

    trainer = TrainerV0(config['batch_size'], dataloader, model, build_id, config['save_rate'])
    trainer.train(epochs=10000)

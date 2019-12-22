from torch.utils.data import DataLoader

from datasets.frei_dataset_u0 import FreiHandDataset
from transforms.to_tensor_u0 import ToTensor

from configs.config_m0 import CONFIG_M0
from models.model_m0 import ModelM0
from trainers.trainer_m0 import TrainerM0

if __name__ == '__main__':
    build_id = CONFIG_M0['build_id']

    dataset = FreiHandDataset(
        CONFIG_M0['dataset_path'],
        CONFIG_M0['data_version'],
        transform=ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG_M0['batch_size'],
        shuffle=True
    )

    model = ModelM0()
    trainer = TrainerM0(CONFIG_M0['batch_size'], dataloader, model, build_id, CONFIG_M0['save_rate'])
    trainer.train(epochs=10000)
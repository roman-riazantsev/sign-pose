from torch.utils.data import DataLoader

from configs.config_u0 import CONFIG_U0
from datasets.frei_dataset_u0 import FreiHandDataset
from models.model_u0 import ModelU0
from trainers.trainer_u0 import TrainerU0
from transforms.to_tensor_u0 import ToTensor

if __name__ == '__main__':
    build_id = CONFIG_U0['build_id']

    dataset = FreiHandDataset(
        CONFIG_U0['dataset_path'],
        CONFIG_U0['data_version'],
        transform=ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG_U0['batch_size'],
        shuffle=True
    )
    model = ModelU0()
    trainer = TrainerU0(CONFIG_U0['batch_size'], dataloader, model, build_id, CONFIG_U0['save_rate'])
    trainer.train(epochs=10000)

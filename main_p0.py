from torch.utils.data import DataLoader
from configs.config_p0 import CONFIG_P0
from datasets.frei_dataset_p0 import FreiDataset
from models.model_p0 import ModelP0
from trainers.trainer_p0 import TrainerP0
from transforms.to_tensor_p0 import ToTensorP0

if __name__ == '__main__':
    build_id = CONFIG_P0['build_id']

    dataset = FreiDataset(
        CONFIG_P0['dataset_path'],
        transform=ToTensorP0()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG_P0['batch_size'],
        shuffle=True
    )

    model = ModelP0()
    trainer = TrainerP0(CONFIG_P0['batch_size'], dataloader, model, build_id)
    trainer.train(epochs=1000, save_rate=100)

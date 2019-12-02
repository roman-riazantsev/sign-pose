import torch
from torch.utils.data import DataLoader

from configs.config_s2 import CONFIG_S2
from datasets.frei_dataset_s2_0 import FreiHandDatasetS2
from datasets.frei_dataset_u0 import FreiHandDataset
from models.model_s2 import ModelS2
from models.model_u0 import ModelU0
from trainers.trainer_s2 import TrainerS2
from trainers.trainer_u0 import TrainerU0
from transforms.np_to_tensor import NpToTensor
from transforms.to_tensor_u0 import ToTensor

if __name__ == '__main__':
    build_id = CONFIG_S2['build_id']

    dataset = FreiHandDatasetS2(
        CONFIG_S2['dataset_path'],
        CONFIG_S2['data_version'],
        transform=NpToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG_S2['batch_size'],
        shuffle=True
    )

    model = ModelS2()
    sample = next(iter(dataloader))
    # print(sample['X'].shape)
    X = sample['X']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model(X)
    trainer = TrainerS2(CONFIG_S2['batch_size'], dataloader, model, build_id, CONFIG_S2['save_rate'])
    trainer.train(epochs=10000)

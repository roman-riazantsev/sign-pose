import torch

from configs.config_s1 import CONFIG_S1
from datasets.frei_dataset_s1 import FreiDatasetS1
from models.model_s1 import ModelS1
from trainers.trainer_s1 import TrainerS1
from transforms.np_to_tensor import NpToTensor
from utils.data_utils import get_loaders
from utils.utils import init_build_id

if __name__ == '__main__':
    config = init_build_id(CONFIG_S1)
    print(f"INITIATED BUILD WITH ID: {config['build_id']}")
    dataset = FreiDatasetS1(
        config=config,
        transform=NpToTensor()
    )
    train_loader, test_loader = get_loaders(config['batch_size'], dataset)
    model = ModelS1()
    test_in = next(iter(train_loader))
    # print(test_in['img'])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    #
    # hz = model(test_in['img'])

    trainer = TrainerS1(config, train_loader, test_loader, model)
    trainer.train(config['epochs'])

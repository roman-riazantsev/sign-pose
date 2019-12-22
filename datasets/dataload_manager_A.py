import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from transforms.np_to_tensor import NpToTensor


def get_train_test_loaders(is_vid, config, dataset):
    # TODO: REFACTOR LOGIC @!
    batch_size = config['batch_size']
    test_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = 32560
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataset = dataset(
        config=config,
        excluded_indices=test_indices,
        transform=NpToTensor(),
        is_vid=is_vid,
    )

    test_dataset = dataset(
        config=config,
        excluded_indices=train_indices,
        transform=NpToTensor(),
        is_vid=is_vid,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             sampler=test_sampler)

    return train_loader, test_loader


def get_validation_loader(is_vid, config, dataset, augment):
    validation_dataset = dataset(
        config,
        transform=NpToTensor(),
        is_vid=is_vid,
        augment=augment
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    return validation_dataloader


def get_loaders(config, dataset_1, dataset_2):
    train_loader_vid, test_loader_vid = get_train_test_loaders(True, config['train_test'], dataset_1)
    train_loader_pic, test_loader_pic = get_train_test_loaders(False, config['train_test'], dataset_1)

    val_loader_vid = get_validation_loader(True, config['val'], dataset_2, augment=False)
    val_loader_pic = get_validation_loader(False, config['val'], dataset_2, augment=False)

    val_loader_augment_vid = get_validation_loader(True, config['val'], dataset_2, augment=True)
    val_loader_augment_pic = get_validation_loader(False, config['val'], dataset_2, augment=True)

    loaders = {
        'VID': {
            'train': train_loader_vid,
            'test': test_loader_vid,
            'val': val_loader_vid,
            'val_augmented': val_loader_augment_vid
        },
        'PIC': {
            'train': train_loader_pic,
            'test': test_loader_pic,
            'val': val_loader_pic,
            'val_augmented': val_loader_augment_pic
        }
    }

    return loaders

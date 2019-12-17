from datasets.synth_dataset_s1 import SynthDatasetS1
from transforms.np_to_tensor import NpToTensor
from transforms.to_tensor_u0 import ToTensor

if __name__ == '__main__':
    config = {
        'dataset_path': '/media/roman/D/11_Datasets/SynthHands_Release/male_noobject/seq05/cam01/01/'
    }

    dataset = SynthDatasetS1(
        config,
        transform=None,
        is_vid=True
    )

    img = dataset[4]

    # print(img[400,500])

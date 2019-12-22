from torch.utils.data import DataLoader

from configs.config_A_VAL import CONFIG_A_VAL
from datasets.dataload_manager_A import get_loaders
from datasets.frei_dataset_s1 import FreiDatasetS1
from datasets.synth_dataset_s1 import SynthDatasetS1
from evaluators.evaluator_A import EvaluatorA
from models.model_s1_2 import ModelS1_2
from models.model_s1_3 import ModelS1_3
from models.model_u0 import ModelU0

from utils.utils import init_build_id

if __name__ == '__main__':
    loaders = get_loaders(CONFIG_A_VAL, FreiDatasetS1, SynthDatasetS1)
    
    models = {
        "STH_2_PIC": ModelS1_2(3),
        "STH_2_VID": ModelS1_2(28),
        "STH_3_PIC": ModelS1_3(3),
        "STH_3_VID": ModelS1_3(28),
        "UNET_PIC": ModelU0(3),
        "UNET_VID": ModelU0(28)
    }

    evaluator = EvaluatorA(CONFIG_A_VAL, loaders, models)
    evaluator.evaluate()
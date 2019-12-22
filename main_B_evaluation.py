from torch.utils.data import DataLoader

from configs.config_A_VAL import CONFIG_A_VAL
from configs.config_B_VAL import CONFIG_B_VAL
from datasets.dataload_manager_A import get_loaders
from datasets.frei_dataset_B import FreiDatasetB
from datasets.frei_dataset_s1 import FreiDatasetS1
from datasets.synth_dataset_B import SynthDatasetB
from datasets.synth_dataset_s1 import SynthDatasetS1
from evaluators.evaluator_A import EvaluatorA
from evaluators.evaluator_B import EvaluatorB
from models.model_B import ModelB
from models.model_s1_2 import ModelS1_2
from models.model_s1_3 import ModelS1_3
from models.model_u0 import ModelU0

from utils.utils import init_build_id

if __name__ == '__main__':
    config = CONFIG_B_VAL
    loaders = get_loaders(config, FreiDatasetB, SynthDatasetB)

    models = {
        "B0_VID": ModelB(True, False),
        "B0_PIC": ModelB(False, False),
        "B1_VID": ModelB(True, True),
        "B1_PIC": ModelB(False, True),
    }

    evaluator = EvaluatorB(config, loaders, models)
    evaluator.evaluate()

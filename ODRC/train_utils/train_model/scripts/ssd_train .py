import os
from pathlib import Path

def train_ssd(BATCH_SIZE, CONFIG_PATH, DATA_PATH, GPU_COUNT, DATASET):
    """
    Runs SSD training using the parameters specified in the config.

    """
    file = Path(__file__).resolve()
    os.system(
            f'python {file.parents[1]}/models/ssd.pytorch/train.py --dataset {DATASET} --dataset_root {DATA_PATH} \
                       --bs {BATCH_SIZE} --nw {GPU_COUNT} \
                       --save_folder ' + '/'.join(CONFIG_PATH.split("/")[:-1]))
import os
from pathlib import Path

def train_frcnn(BATCH_SIZE, GPU_COUNT, SELECT_GPU, DATA_PATH):
    """
    Runs faster-rccn training using the parameters specified in the config.

    """
    file = Path(__file__).resolve()
    os.system(
        f' CUDA_VISIBLE_DEVICES={SELECT_GPU} python trainval_net.py \
                    --dataset {DATA_PATH} --net res101 \
                    --bs {BATCH_SIZE} --nw {GPU_COUNT} \
                    --cuda --mGPUs')
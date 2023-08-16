import os
from pathlib import Path


def train_frcnn(CONFIG_PATH, EPOCHS, BATCH_SIZE, GPU_COUNT, IMG_SIZE):
    """
    Runs faster-rccn training using the parameters specified in the config.
    """
    file = Path(__file__).resolve()
    command = "python3" if GPU_COUNT == 0 else f"OMP_NUM_THREADS=1 python3 -m torch.distributed.run --nproc_per_node {GPU_COUNT}"
    os.system(
        command +
        f' {file.parents[1]}/models/fastercnn-pytorch-training-pipeline/train.py'
        f" --data {CONFIG_PATH}"
        f" --epochs {EPOCHS}"
        f" --batch {BATCH_SIZE}"
        f" --model fasterrcnn_resnet50_fpn" +
        f" --name {os.path.dirname(CONFIG_PATH)}")

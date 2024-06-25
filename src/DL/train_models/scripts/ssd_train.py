import os
from pathlib import Path
import subprocess


def train_ssd(CONFIG_PATH):
    """
    Runs SSD training using the parameters specified in the config.
    """
    file = Path(__file__).resolve()

    full_command = (
        f'python3 {file.parents[1]}/models/PyTorch-SSD/train.py'
        f" --cfg {CONFIG_PATH}"
        f" --logdir {os.path.dirname(CONFIG_PATH)}/exp")
    
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.decode().strip())

    rc = process.poll()
    return rc


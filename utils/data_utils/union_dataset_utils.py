import os
import shutil
from tqdm import tqdm

DIR = '../Monitoring_photo_отработаны'
SAVE_DIR = '/home/farm/PycharmProjects/dataset_posad_monitoring'

os.makedirs(SAVE_DIR, exist_ok=True)

for dir in tqdm(os.listdir(DIR)):
    # print(dir)
    for file in os.listdir(f'{DIR}/{dir}/'):
        # print(file)
        shutil.copyfile(f'{DIR}/{dir}/{file}', f'{SAVE_DIR}/{file}')

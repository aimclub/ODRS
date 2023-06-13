import os 
from loguru import logger
import shutil
from tqdm import tqdm
from pathlib import Path

#напиши сортировку пузыр

# def split_data(datapath, SPLIT_TRAIN_VALUE, SPLIT_VAL_VALUE, SPLIT_TEST_VALUE):
#     # Create train, test and validation dataset
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[2] #PATH TO ODRC_project
#     name_dir = f'{ROOT}/dataset'


#     if os.path.exists(f'{datapath}/train') or os.path.exists(f'{datapath}/test') or os.path.exists(f'{datapath}/val'):
#         logger.exception('Directory with split dataset already exist')
#         name_dir = datapath

#     else: 
#         os.makedirs(f'{name_dir}')
#         os.makedirs(f'{name_dir}/train')
#         os.makedirs(f'{name_dir}/test')
#         os.makedirs(f'{name_dir}/val')
#         train_size = int(SPLIT_TRAIN_VALUE * len(os.listdir(datapath)))
#         val_size = int(SPLIT_VAL_VALUE * len(os.listdir(datapath)))
#         test_size = int(SPLIT_TEST_VALUE * len(os.listdir(datapath)))
#         count = 0
#         for data in tqdm(os.listdir(datapath)):
#             if count < train_size:
#                 shutil.copy(datapath + '/' + data, f'{name_dir}/train')
#                 count += 1
#             if count >= train_size and count < train_size + val_size:
#                 shutil.copy(datapath + '/' + data, f'{name_dir}/val')
#                 count += 1
#             if count > train_size and count >= train_size + val_size:
#                 shutil.copy(datapath + '/' + data, f'{name_dir}/test')
#                 count += 1

#         logger.info('Count of train example: ' + str(train_size))
#         logger.info('Count of val example: ' + str(val_size))
#         logger.info("Count of test example: " + str(test_size))

#     PATH_SPLIT_TRAIN = f'{name_dir}/train'
#     PATH_SPLIT_VALID = f'{name_dir}/val'

#     return PATH_SPLIT_TRAIN, PATH_SPLIT_VALID

def split_data(datapath, split_train_value, split_val_value, split_test_value):
    # Create train, test and validation datasets
    file = Path(__file__).resolve()
    root = file.parents[2] # Path to ODRC_project
    name_dir = f'{root}/dataset'

    if any(os.path.exists(f'{datapath}/{dir_name}') for dir_name in ['train', 'test', 'val']):
        raise Exception('Directory with split dataset already exists')
    else: 
        os.makedirs(name_dir, exist_ok=True)
        os.makedirs(f'{name_dir}/train', exist_ok=True)
        os.makedirs(f'{name_dir}/test', exist_ok=True)
        os.makedirs(f'{name_dir}/val', exist_ok=True)
        file_list = os.listdir(datapath)
        train_size = int(split_train_value * len(file_list))
        val_size = int(split_val_value * len(file_list))
        test_size = int(split_test_value * len(file_list))
        count = 0
        for data in tqdm(file_list):
            src_file = os.path.join(datapath, data)
            if count < train_size:
                dst_dir = os.path.join(name_dir, 'train')
            elif count < train_size + val_size:
                dst_dir = os.path.join(name_dir, 'val')
            elif count < train_size + val_size + test_size:
                dst_dir = os.path.join(name_dir, 'test')
            else:
                break
            shutil.copy(src_file, dst_dir)
            count += 1

        assert count == train_size + val_size + test_size

        logger.info('Count of train examples:', train_size)
        logger.info('Count of val examples:', val_size)
        logger.info('Count of test examples:', test_size)

    path_split_train = f'{name_dir}/train'
    path_split_valid = f'{name_dir}/val'

    return path_split_train, path_split_valid
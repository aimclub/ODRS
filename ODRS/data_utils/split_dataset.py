import os
import shutil
import glob
import sys
from tqdm import tqdm
from loguru import logger

def split_data(datapath, split_train_value, split_valid_value):
    selected_folders = ['test', 'train', 'valid']

    train_path = os.path.join(datapath, 'train')
    test_path = os.path.join(datapath, 'test')
    val_path = os.path.join(datapath, 'valid')

    if os.path.exists(train_path) and (os.path.exists(val_path)
                                        or os.path.exists(os.path.join(datapath, 'val'))):
        logger.info("Dataset is ready")
        return train_path, val_path if os.path.exists(val_path) else os.path.join(datapath, 'val')
    if os.path.exists(train_path) and not (os.path.exists(val_path)
                                        or os.path.exists(os.path.join(datapath, 'val'))):
        logger.error("Dataset has no validation sample")
        sys.exit()
    if not os.path.exists(train_path) and (os.path.exists(val_path)
                                        or os.path.exists(os.path.join(datapath, 'val'))):
        logger.error("Dataset has no training sample")
        sys.exit()


    images_path = os.path.join(datapath, 'images')
    labels_path = os.path.join(datapath, 'labels')

    if os.path.exists(images_path) and os.path.exists(labels_path):
        image_files = glob.glob(os.path.join(images_path, '*.jpg')) + \
                      glob.glob(os.path.join(images_path, '*.jpeg')) + \
                      glob.glob(os.path.join(images_path, '*.png'))
        label_files = glob.glob(os.path.join(labels_path, '*.txt'))
    else:
        image_files = glob.glob(os.path.join(datapath, '*.jpg')) + \
                      glob.glob(os.path.join(datapath, '*.jpeg')) + \
                      glob.glob(os.path.join(datapath, '*.png'))
        label_files = glob.glob(os.path.join(datapath, '*.txt'))

    image_files.sort()
    label_files.sort()

    total_files = len(image_files) + len(label_files)

    if total_files == 0:
        logger.error("Error: No image or label files found in the datapath.")

    train_split = int(len(image_files) * split_train_value)
    val_split = int(len(image_files) * split_valid_value)

    logger.info(f'Total number of images:{len(image_files)}')
    logger.info(f'Total number of labels:{len(label_files)}')

    train_images = image_files[:train_split]
    train_labels = label_files[:train_split]
    logger.info(f'Number train images:{len(train_images)}')
    logger.info(f'Number train labels:{len(train_labels)}')

    val_images = image_files[train_split:train_split+val_split]
    val_labels = label_files[train_split:train_split+val_split]
    logger.info(f'Number valid images:{len(val_images)}')
    logger.info(f'Number valid labels:{len(val_labels)}')

    test_images = image_files[train_split+val_split:]
    test_labels = label_files[train_split+val_split:]
    logger.info(f'Number test images:{len(test_images)}')
    logger.info(f'Number test labels:{len(test_labels)}')

    for path in [train_path, test_path, val_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        images_subpath = os.path.join(path, 'images')
        labels_subpath = os.path.join(path, 'labels')
        os.makedirs(images_subpath)
        os.makedirs(labels_subpath)

    for image_file in tqdm(train_images, desc="Train images"):
        shutil.copy(image_file, os.path.join(train_path, 'images', os.path.basename(image_file)))
    for image_file in tqdm(val_images, desc="Valid images"):
        shutil.copy(image_file, os.path.join(val_path, 'images', os.path.basename(image_file)))
    for image_file in tqdm(test_images, desc="Test images"):
        shutil.copy(image_file, os.path.join(test_path, 'images', os.path.basename(image_file)))

    for label_file in tqdm(train_labels, desc="Train labels"):
        shutil.copy(label_file, os.path.join(train_path, 'labels', os.path.basename(label_file)))
    for label_file in tqdm(val_labels, desc="Valid labels"):
        shutil.copy(label_file, os.path.join(val_path, 'labels', os.path.basename(label_file)))
    for label_file in tqdm(test_labels, desc="Test labels"):
        shutil.copy(label_file, os.path.join(test_path, 'labels', os.path.basename(label_file)))

    for root, dirs, files in os.walk(datapath, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if file_path.split('/')[-3] not in selected_folders:
                os.remove(file_path)

    logger.info("Dataset was split")
    return train_path, val_path


def remove_folder(path):
    shutil.rmtree(path)


def copy_arch_folder(dataset_path):
    dataset_folder = dataset_path.parent
    dataset_name = f'{dataset_path.name}_voc'
    voc_path = os.path.join(dataset_folder, dataset_name)
    yolo_path = os.path.join(dataset_path)
    if os.path.exists(voc_path):
        remove_folder(voc_path)
    shutil.copytree(yolo_path, voc_path)
    return voc_path

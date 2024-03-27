import os
import shutil
import glob
import sys
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from PIL import Image


def sorted_files(image_files, label_files):
    new_label_files = list()
    new_image_files = list()
    for image_path in tqdm(image_files, desc="Sorting"):
        image_stem = Path(image_path).stem
        for label_path in label_files:
            label_stem = Path(label_path).stem
            if label_stem == image_stem:
                new_image_files.append(image_path)
                new_label_files.append(label_path)
                break
    return new_image_files, new_label_files


def split_data(datapath, split_train_value, split_valid_value):
    selected_folders = ['test', 'train', 'valid']

    train_path = os.path.join(datapath, 'train')
    test_path = os.path.join(datapath, 'test')
    val_path = os.path.join(datapath, 'valid')


    if os.path.exists(train_path) and (os.path.exists(val_path) or
                                       os.path.exists(os.path.join(datapath, 'val'))):
        logger.info("Dataset is ready")
        return train_path, val_path if os.path.exists(val_path) else os.path.join(datapath, 'val')
    if os.path.exists(train_path) and not (os.path.exists(val_path) or
                                           os.path.exists(os.path.join(datapath, 'val'))):
        logger.error("Dataset has no validation sample")
        sys.exit()
    if not os.path.exists(train_path) and (os.path.exists(val_path) or
                                           os.path.exists(os.path.join(datapath, 'val'))):
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

    image_files, label_files = sorted_files(image_files, label_files)

    total_files = len(image_files) + len(label_files)

    if total_files == 0:
        logger.error("Error: No image or label files found in the datapath.")

    train_split = int(len(image_files) * split_train_value)
    val_split = int(len(image_files) * split_valid_value)

    logger.info(f'Total number of images:{len(image_files)}')
    logger.info(f'Total number of labels:{len(label_files)}')

    train_images = image_files[:train_split]
    train_labels = label_files[:len(train_images)]
    logger.info(f'Number train images:{len(train_images)}')
    logger.info(f'Number train labels:{len(train_labels)}')

    val_images = image_files[len(train_images):len(train_images)+val_split]
    val_labels = label_files[len(train_images):len(train_images)+val_split]
    logger.info(f'Number valid images:{len(val_images)}')
    logger.info(f'Number valid labels:{len(val_labels)}')

    test_images = image_files[len(train_images)+len(val_images):]
    test_labels = label_files[len(train_images)+len(val_images):]
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
        shutil.move(image_file, os.path.join(train_path, 'images', os.path.basename(image_file)))
    for image_file in tqdm(val_images, desc="Valid images"):
        shutil.move(image_file, os.path.join(val_path, 'images', os.path.basename(image_file)))
    for image_file in tqdm(test_images, desc="Test images"):
        shutil.move(image_file, os.path.join(test_path, 'images', os.path.basename(image_file)))

    for label_file in tqdm(train_labels, desc="Train labels"):
        shutil.move(label_file, os.path.join(train_path, 'labels', os.path.basename(label_file)))
    for label_file in tqdm(val_labels, desc="Valid labels"):
        shutil.move(label_file, os.path.join(val_path, 'labels', os.path.basename(label_file)))
    for label_file in tqdm(test_labels, desc="Test labels"):
        shutil.move(label_file, os.path.join(test_path, 'labels', os.path.basename(label_file)))

    for item in os.listdir(datapath):
        full_path = os.path.join(datapath, item)
        if os.path.isfile(full_path):
            os.remove(full_path)


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


def resize_images_and_annotations(data_path, img_size):
    if isinstance(img_size, int):
        width = height = img_size
    elif isinstance(img_size, tuple) and len(img_size) == 2:
        width, height = img_size
    else:
        raise ValueError("Invalid img_size format. Please provide either an integer or a tuple of two integers.")

    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]

    for name in folder_names:
        folder_path = path / name
        images_path = os.path.join(folder_path, 'images')
        labels_path = os.path.join(folder_path, 'labels')

        for image_name in tqdm(os.listdir(images_path), desc=f'Resize {name} images'):
            image_path = os.path.join(images_path, image_name)
            label_path = os.path.join(labels_path, image_name.replace('.jpg', '.txt'))

            with Image.open(image_path) as img:
                original_width, original_height = img.size

                if original_width != width or original_height != height:
                    img = img.resize((width, height))

                    if os.path.exists(label_path):
                        with open(label_path, 'r') as file:
                            lines = file.readlines()

                        with open(label_path, 'w') as file:
                            for line in lines:
                                parts = line.split()
                                if len(parts) == 5:
                                    x_center = float(parts[1]) * original_width
                                    y_center = float(parts[2]) * original_height
                                    box_width = float(parts[3]) * original_width
                                    box_height = float(parts[4]) * original_height

                                    x_center *= width / original_width
                                    y_center *= height / original_height
                                    box_width *= width / original_width
                                    box_height *= height / original_height

                                    x_center /= width
                                    y_center /= height
                                    box_width /= width
                                    box_height /= height

                                    file.write(f"{parts[0]} {x_center} {y_center} {box_width} {box_height}\n")

                    img.save(image_path)

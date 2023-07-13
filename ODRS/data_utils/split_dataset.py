import os
import shutil
import glob


def split_data(datapath, split_train_value, split_val_value, split_test_value):
    selected_folders = ['test', 'train', 'valid']
    selected_files = ['classes.txt']

    train_path = os.path.join(datapath, 'train')
    test_path = os.path.join(datapath, 'test')
    val_path = os.path.join(datapath, 'valid')

    if os.path.exists(train_path) and os.path.exists(test_path) and (os.path.exists(val_path) 
                                        or os.path.exists(os.path.join(datapath, 'valid'))):
        return "Dataset is ready"

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

    total_files = len(image_files) + len(label_files)

    if total_files == 0:
        print("Error: No image or label files found in the datapath.")
        return

    train_split = int(len(image_files) * split_train_value)
    val_split = int(len(image_files) * split_val_value)

    print(f'Len_images_files:{len(image_files)}')

    train_images = image_files[:train_split]
    train_labels = label_files[:train_split]
    print(f'train_images:{len(train_images)}')

    val_images = image_files[train_split:train_split+val_split]
    val_labels = label_files[train_split:train_split+val_split]
    print(f'val_labels:{len(val_labels)}')

    test_images = image_files[train_split+val_split:]
    test_labels = label_files[train_split+val_split:]
    print(f'test_labels:{len(test_labels)}')

    for path in [train_path, test_path, val_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        images_subpath = os.path.join(path, 'images')
        labels_subpath = os.path.join(path, 'labels')
        os.makedirs(images_subpath)
        os.makedirs(labels_subpath)

    for image_file in train_images:
        shutil.copy(image_file, os.path.join(train_path, 'images', os.path.basename(image_file)))
    for image_file in val_images:
        shutil.copy(image_file, os.path.join(val_path, 'images', os.path.basename(image_file)))
    for image_file in test_images:
        shutil.copy(image_file, os.path.join(test_path, 'images', os.path.basename(image_file)))

    for label_file in train_labels:
        shutil.copy(label_file, os.path.join(train_path, 'labels', os.path.basename(label_file)))
    for label_file in val_labels:
        shutil.copy(label_file, os.path.join(val_path, 'labels', os.path.basename(label_file)))
    for label_file in test_labels:
        shutil.copy(label_file, os.path.join(test_path, 'labels', os.path.basename(label_file)))

    for root, dirs, files in os.walk(datapath, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if name not in selected_files and file_path.split('/')[-3] not in selected_folders:
                os.remove(file_path)

        for name in dirs:
            dir_path = os.path.join(root, name)
            if name not in selected_folders and dir_path.split('/')[-2] not in selected_folders:
                shutil.rmtree(dir_path)

    return "Dataset was split"


def remove_folder(path):
    shutil.rmtree(path)


def copy_arch_folder(dataset_path):
    folder_name = dataset_path.split('/')[-1]
    dataset_path = os.path.dirname(dataset_path)
    voc_path = os.path.join(os.path.dirname(dataset_path), "voc")
    yolo_path = os.path.join(dataset_path)
    if os.path.exists(voc_path):
        remove_folder(voc_path)
    shutil.copytree(yolo_path, voc_path)
    return f'{voc_path}/{folder_name}'

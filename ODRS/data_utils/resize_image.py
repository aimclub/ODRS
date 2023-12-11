import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image

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


# resize_images_and_annotations('/media/space/ssd_1_tb_evo_sumsung/Work/Warp-D', (640, 480))
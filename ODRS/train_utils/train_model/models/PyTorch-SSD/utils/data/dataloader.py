import torch
import json
import utils.data.transforms as T
from PIL import Image
from utils.constants import BACKGROUND_INDEX
from torch.utils.data import Dataset, DataLoader


def _get_transform(image_size, augment, image_mean, image_stddev):
    if augment:
        return T.Compose(
            [
                T.RandomDistortColor(),
                T.RandomPad(image_mean),
                T.RandomCrop(),
                T.RandomHorizontalFlip(),
                T.Resize(image_size),
                T.PILToTensor(),
                T.Normalize(image_mean, image_stddev),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(image_size),
                T.PILToTensor(),
                T.Normalize(image_mean, image_stddev),
            ]
        )


class _ObjectDetectionDataset(Dataset):
    def __init__(self, json_file, transform):
        with open(json_file) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        image_path = d['image']
        boxes = d['boxes']
        classes = d['classes']
        difficulties = d['difficulties']

        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
            classes = [BACKGROUND_INDEX]
            difficulties = [0]

        image = Image.open(image_path).convert('RGB')
        boxes = torch.FloatTensor(boxes)
        classes = torch.LongTensor(classes)
        difficulties = torch.LongTensor(difficulties)
        image, boxes, classes, difficulties = self.transform(
            image, boxes, classes, difficulties
        )
        return image, boxes, classes, difficulties

    @staticmethod
    def collate_fn(batch):
        images, boxes, classes, difficulties = zip(*batch)
        images = torch.stack(images, axis=0)
        return images, boxes, classes, difficulties


def create_dataloader(json_file, batch_size, image_size, image_mean,
                      image_stddev, augment=False, shuffle=False, seed=None,
                      num_workers=0):
    dataset = _ObjectDetectionDataset(
        json_file,
        _get_transform(image_size, augment, image_mean, image_stddev)
    )

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=dataset.collate_fn,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            generator=g)
    return dataloader

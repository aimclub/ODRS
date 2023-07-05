import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from utils.boxes import xyxy2cxcywh, calculate_ious


class Compose(object):
    def __init__(self, transforms):
        self.ts = transforms

    def __call__(self, *args):
        for t in self.ts:
            args = t(*args)
        return args


class RandomDistortColor(object):
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.t = T.ColorJitter(brightness=brightness,
                               contrast=contrast,
                               saturation=saturation,
                               hue=hue)

    def __call__(self, image, *args):
        image = self.t(image)
        return (image, *args)


class RandomPad(object):
    def __init__(self, image_mean, max_scale=4.):
        self.image_mean = tuple([round(x) for x in image_mean])
        self.max_scale = max_scale

    def __call__(self, image, boxes, *args):
        if random.random() < 0.5:
            s = random.uniform(1, self.max_scale)
            w, h = TF.get_image_size(image)
            dx = round(w * (s - 1))
            dy = round(h * (s - 1))
            pad_left = random.randint(0, dx)
            pad_right = dx - pad_left
            pad_top = random.randint(0, dy)
            pad_bottom = dy - pad_top
            image = TF.pad(image,
                           padding=[pad_left, pad_top, pad_right, pad_bottom],
                           fill=self.image_mean)

            offset = torch.FloatTensor([pad_left, pad_top]).repeat([2])
            boxes = boxes + offset
        return (image, boxes, *args)


class RandomCrop(object):
    def __init__(self,
                 max_attempts=50,
                 min_ious=[float('-inf'), 0.1, 0.3, 0.5, 0.7, 0.9, None],
                 scale_range=[0.3, 1.0],
                 aspect_ratio_range=[0.5, 2.0]):
        self.max_attempts = max_attempts
        self.min_ious = min_ious
        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range

    def _sample_new_roi(self, im_w, im_h):
        """
        Returns:
            roi: int32 tensor. ROI in pixel coordinates and `xyxy` format.
        """
        roi_w = random.uniform(self.scale_range[0], self.scale_range[1])
        roi_h = random.uniform(self.scale_range[0], self.scale_range[1])
        roi_x1 = random.uniform(0, 1 - roi_w)
        roi_y1 = random.uniform(0, 1 - roi_h)
        roi_x2 = roi_x1 + roi_w
        roi_y2 = roi_y1 + roi_h
        roi = torch.FloatTensor(
            [
                roi_x1 * im_w,
                roi_y1 * im_h,
                roi_x2 * im_w,
                roi_y2 * im_h
            ]
        )
        roi = torch.round(roi).int()
        return roi

    def _is_valid_aspect_ratio(self, roi):
        w, h = roi[2:] - roi[:2]
        aspect_ratio = w / h
        return (self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1])

    def __call__(self, image, boxes, classes, difficulties):
        im_w, im_h = TF.get_image_size(image)
        cxcy = xyxy2cxcywh(boxes)[:, :2]   # [num_boxes, 2]
        cx, cy = cxcy[:, 0], cxcy[:, 1]

        # Find a proper ROI
        attempts = -1
        while True:
            attempts = (attempts + 1) % self.max_attempts
            if attempts == 0:
                min_iou = random.choice(self.min_ious)
                if min_iou is None:
                    return image, boxes, classes, difficulties

            roi = self._sample_new_roi(im_w, im_h)   # [4]

            if not self._is_valid_aspect_ratio(roi):
                continue

            is_in_roi = (cx > roi[0]) & (cy > roi[1]) & (cx < roi[2]) & (cy < roi[3])
            if not is_in_roi.any():
                continue

            ious = calculate_ious(roi.unsqueeze(0), boxes)   # [1, num_boxes]
            if ious.max() < min_iou:
                continue

            break

        # Crop the image and adjust boxes according ROI
        w, h = roi[2:] - roi[:2]
        image = TF.crop(image, roi[1].item(), roi[0].item(), h.item(), w.item())

        boxes = torch.clip(
            boxes,
            roi[:2].repeat([2]),
            roi[2:].repeat([2]),
        )
        boxes = boxes - roi[:2].repeat([2])

        boxes = boxes[is_in_roi]
        classes = classes[is_in_roi]
        difficulties = difficulties[is_in_roi]
        return image, boxes, classes, difficulties


class RandomHorizontalFlip(object):
    def __call__(self, image, boxes, *args):
        if random.random() < 0.5:
            image = TF.hflip(image)
            im_w, _ = TF.get_image_size(image)
            xmins, xmaxes = im_w - boxes[:, 2], im_w - boxes[:, 0]  # [num_boxes], [num_boxes]
            boxes[:, 0] = xmins
            boxes[:, 2] = xmaxes
        return (image, boxes, *args)


class Resize(object):
    def __init__(self, image_size):
        self.size = image_size

    def __call__(self, image, boxes, *args):
        w, h = TF.get_image_size(image)
        image = TF.resize(image, (self.size, self.size))
        scales = torch.FloatTensor([self.size / w, self.size / h]).repeat([2])
        boxes = boxes * scales
        return (image, boxes, *args)


class LetterBox(object):
    def __init__(self, image_mean, image_size):
        self.image_mean = tuple([round(x) for x in image_mean])
        self.size = image_size

    def __call__(self, image, boxes, *args):
        w0, h0 = TF.get_image_size(image)
        s = self.size / max(w0, h0)
        w, h = round(w0 * s), round(h0 * s)

        image = TF.resize(image, (h, w))
        dx, dy = self.size - w, self.size - h
        left, top = dx // 2, dy // 2
        right, bottom = dx - left, dy - top
        image = TF.pad(image,
                       padding=[left, top, right, bottom],
                       fill=self.image_mean)
        boxes = boxes * torch.FloatTensor([w / w0, h / h0]).repeat([2])
        boxes = boxes + torch.FloatTensor([left, top]).repeat([2])
        return (image, boxes, *args)


class PILToTensor(object):
    def __call__(self, image, *args):
        image = TF.pil_to_tensor(image)
        return (image, *args)


class Normalize(object):
    def __init__(self, mean, stddev):
        self.mean = torch.FloatTensor(mean).reshape([-1, 1, 1])
        self.stddev = torch.FloatTensor(stddev).reshape([-1, 1, 1])

    def __call__(self, image, *args):
        image = (image.float() - self.mean) / self.stddev
        return (image, *args)

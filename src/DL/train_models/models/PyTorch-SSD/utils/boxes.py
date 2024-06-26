import torch


def xyxy2xywh(x):
    wh = x[..., 2:] - x[..., :2]
    return torch.cat([x[..., :2], wh], axis=-1)


def cxcywh2xyxy(x):
    mins = x[..., :2] - x[..., 2:] / 2
    maxes = x[..., :2] + x[..., 2:] / 2
    return torch.cat([mins, maxes], axis=-1)


def xyxy2cxcywh(x):
    wh = x[..., 2:] - x[..., :2]
    cxcy = x[..., :2] + wh / 2
    return torch.cat([cxcy, wh], axis=-1)


def calculate_intersections(set_1, set_2):
    """
    Args:
        set_1, set_2: float32 tensors. Bounding boxes in `xyxy` format.
            Shape: [n1, 4], [n2, 4].

    Returns:
        areas: float32 tensor. Shape: [n1, n2].
    """
    set_1 = torch.unsqueeze(set_1, axis=1)   # [n1, 1, 2]
    lower_bounds = torch.maximum(set_1[..., :2], set_2[..., :2])   # [n1, n2, 2]
    upper_bounds = torch.minimum(set_1[..., 2:], set_2[..., 2:])   # [n1, n2, 2]
    intersect_rectangle = torch.clip(upper_bounds - lower_bounds, min=0)   # [n1, n2, 2]
    areas = torch.prod(intersect_rectangle, axis=-1)    # [n1, n2]
    return areas


def calculate_ious(set_1, set_2):
    """
    Args:
        set_1, set_2: float32 tensors. Bounding boxes in `xyxy` format.
            Shape: [n1, 4], [n2, 4].

    Returns:
        ious: float32 tensor. Shape: [n1, n2].
    """
    intersections = calculate_intersections(set_1, set_2)   # [n1, n2]
    areas_set_1 = torch.prod(set_1[:, 2:] - set_1[:, :2], axis=-1)  # [n1]
    areas_set_2 = torch.prod(set_2[:, 2:] - set_2[:, :2], axis=-1)  # [n2]

    # Find the union
    areas_set_1 = torch.unsqueeze(areas_set_1, axis=-1)    # [n1, 1]
    unions = (areas_set_1 + areas_set_2 - intersections)   # [n1, n2]
    ious = intersections / unions   # [n1, n2]
    return ious

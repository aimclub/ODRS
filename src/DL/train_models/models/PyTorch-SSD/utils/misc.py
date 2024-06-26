import torch
import torchvision
import utils.models
import utils.models.backbones
from yacs.config import CfgNode
from utils.constants import HEX_COLORS
from torchvision.utils import draw_bounding_boxes


def load_config(fname):
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(fname)
    cfg.freeze()
    return cfg


def build_model(cfg):
    backbone = getattr(utils.models.backbones, cfg.backbone.pop('name'))(**cfg.backbone)
    return getattr(utils.models, cfg.model)(
        backbone,
        num_classes=len(cfg.class_names),
        input_size=cfg.input_size,
        anchor_scales=cfg.anchor_scales,
        anchor_aspect_ratios=cfg.anchor_aspect_ratios
    )


def nms(boxes, scores, classes, score_thres=0.3, iou_thres=0.45, max_dets=200):
    """
    Args:
        boxes: float32 tensor. Bounding boxes in `xyxy` format.
            Shape: [bs, n, 4].
        scores: float32 tensor. Shape: [bs, n].
        classes: int64 tensor. Shape: [bs, n].

    Returns:
        det_boxes, det_scores, det_classes: Lists of tensors.
            Shape: [bs, [*, 4]], [bs, [*]], [bs, [*]].
    """
    bs = boxes.shape[0]
    nms_results = []
    for i in range(bs):
        boxes_over_score_thres = boxes[i][scores[i] > score_thres]
        scores_over_score_thres = scores[i][scores[i] > score_thres]
        classes_over_score_thres = classes[i][scores[i] > score_thres]
        indices = torchvision.ops.nms(
            boxes=(   # offset by class; a trick to separate boxes of different classes.
                boxes_over_score_thres
                + 1e4 * classes_over_score_thres.float().unsqueeze(-1)
            ),
            scores=scores_over_score_thres,
            iou_threshold=iou_thres
        )
        indices = indices[:max_dets]
        nms_results.append(
            [
                boxes_over_score_thres[indices],
                scores_over_score_thres[indices],
                classes_over_score_thres[indices],
            ]
        )

    det_boxes, det_scores, det_classes = list(zip(*nms_results))
    return det_boxes, det_scores, det_classes


def unnormalize(images, mean, stddev):
    mean = torch.FloatTensor(mean).reshape([3, 1, 1])
    stddev = torch.FloatTensor(stddev).reshape([3, 1, 1])
    images = torch.clip(images * stddev + mean, 0, 255)
    images = images.byte()
    return images


def draw_boxes(image, class_names, boxes, classes, scores=None, width=3):
    labels = [class_names[i] for i in classes]
    if scores is not None:
        labels = [label + ' %.2f' % score
                  for (label, score) in zip(labels, scores)]
    colors = [HEX_COLORS[i] for i in classes]
    return draw_bounding_boxes(image, boxes, labels, colors=colors, width=width)

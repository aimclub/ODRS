import torch
import argparse
import os
import json
import tempfile
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.cuda.amp import autocast
from utils.boxes import xyxy2xywh
from utils.misc import load_config, build_model, nms


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--coco_dir', type=str, required=True,
                        help="path to a directory containing COCO 2017 dataset.")
    parser.add_argument('--pth', type=str, required=True,
                        help="checkpoint")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)

    model = build_model(cfg)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.pth)['model_state_dict'])

    preprocessing = Compose(
        [
            Resize((cfg.input_size,) * 2),
            ToTensor(),
            Normalize([x / 255 for x in cfg.image_mean], [x / 255 for x in cfg.image_stddev]),
        ]
    )

    coco = COCO(os.path.join(args.coco_dir, 'annotations/instances_val2017.json'))
    cat_ids = coco.getCatIds()
    results = []
    with torch.no_grad():
        for k, v in tqdm(coco.imgs.items()):
            image_path = os.path.join(args.coco_dir, 'val2017/%s' % v['file_name'])
            image = Image.open(image_path).convert('RGB')
            image = preprocessing(image)
            image = image.unsqueeze(0).to(device)

            with autocast(enabled=(not args.no_amp)):
                preds = model(image)
            det_boxes, det_scores, det_classes = nms(*model.decode(preds))
            det_boxes, det_scores, det_classes = det_boxes[0], det_scores[0], det_classes[0]

            det_boxes = torch.clip(det_boxes / cfg.input_size, 0, 1)
            det_boxes = (
                det_boxes.cpu()
                * torch.FloatTensor([v['width'], v['height']]).repeat([2])
            )
            det_boxes = xyxy2xywh(det_boxes)

            det_boxes, det_scores, det_classes = (
                det_boxes.tolist(),
                det_scores.tolist(),
                det_classes.tolist(),
            )

            det_classes = [cat_ids[c] for c in det_classes]

            for box, score, clss in zip(det_boxes, det_scores, det_classes):
                results.append(
                    {
                        'image_id': k,
                        'category_id': clss,
                        'bbox': box,
                        'score': score
                    }
                )

    _, tmp_json = tempfile.mkstemp('.json')
    with open(tmp_json, 'w') as f:
        json.dump(results, f)
    results = coco.loadRes(tmp_json)
    coco_eval = COCOeval(coco, results, 'bbox')
    coco_eval.params.imgIds = list(coco.imgs.keys())   # image IDs to evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()

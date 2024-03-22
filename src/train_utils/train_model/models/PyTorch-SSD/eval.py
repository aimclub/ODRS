import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, build_model, nms
from utils.metrics import AveragePrecision


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--dataset', type=str, required=True,
                        help="dataset JSON file")
    parser.add_argument('--pth', type=str, required=True,
                        help="checkpoint")
    parser.add_argument('--workers', type=int, default=4,
                        help="number of dataloader workers")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)

    model = build_model(cfg)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.pth)['model_state_dict'])

    dataloader = create_dataloader(args.dataset,
                                   batch_size=cfg.batch_size,
                                   image_size=cfg.input_size,
                                   image_mean=cfg.image_mean,
                                   image_stddev=cfg.image_stddev,
                                   num_workers=args.workers)

    metric = AveragePrecision(len(cfg.class_names), cfg.recall_steps)
    metric.reset()
    pbar = tqdm(dataloader, bar_format="{l_bar}{bar:30}{r_bar}")
    with torch.no_grad():
        for (images, true_boxes, true_classes, difficulties) in pbar:
            images = images.to(device)
            true_boxes = [x.to(device) for x in true_boxes]
            true_classes = [x.to(device) for x in true_classes]
            difficulties = [x.to(device) for x in difficulties]

            with autocast(enabled=(not args.no_amp)):
                preds = model(images)
            det_boxes, det_scores, det_classes = nms(*model.decode(preds))
            metric.update(det_boxes, det_scores, det_classes, true_boxes,
                          true_classes, difficulties)

    APs = metric.result
    print()
    print("     Category            AP@[0.5]     AP@[0.5:0.95]")
    for i, (ap, name) in enumerate(zip(APs, cfg.class_names)):
        print("%-5d%-20s%-13.3f%.3f" % (i, name, ap[0], ap.mean()))
    print("mAP@[0.5]: %.3f" % APs[:, 0].mean())
    print("mAP@[0.5:0.95]: %.3f" % APs.mean())


if __name__ == '__main__':
    main()

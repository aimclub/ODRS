import argparse
import random
import matplotlib.pyplot as plt
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, draw_boxes, unnormalize


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'], help="either `train` or `val`")
    args = parser.parse_args()

    cfg = load_config(args.cfg)

    seed = random.randint(0, 9999)
    dataloader_0 = create_dataloader(cfg[args.split + '_json'],
                                     batch_size=cfg.batch_size,
                                     image_size=cfg.input_size,
                                     image_mean=cfg.image_mean,
                                     image_stddev=cfg.image_stddev,
                                     augment=False,
                                     shuffle=True,
                                     seed=seed)
    dataloader_1 = create_dataloader(cfg[args.split + '_json'],
                                     batch_size=cfg.batch_size,
                                     image_size=cfg.input_size,
                                     image_mean=cfg.image_mean,
                                     image_stddev=cfg.image_stddev,
                                     augment=True,
                                     shuffle=True,
                                     seed=seed)
    dataiter_0 = iter(dataloader_0)
    dataiter_1 = iter(dataloader_1)

    while True:
        plt.figure(figsize=(15, 7))

        images, boxes, classes, _ = next(dataiter_0)
        image = unnormalize(images[0], cfg.image_mean, cfg.image_stddev)
        image = draw_boxes(image, cfg.class_names, boxes[0], classes[0])
        plt.subplot(1, 2, 1)
        plt.title("w/o augmentation")
        plt.imshow(image.permute([1, 2, 0]))

        images, boxes, classes, _ = next(dataiter_1)
        image = unnormalize(images[0], cfg.image_mean, cfg.image_stddev)
        image = draw_boxes(image, cfg.class_names, boxes[0], classes[0])
        plt.subplot(1, 2, 2)
        plt.title("w/ augmentation")
        plt.imshow(image.permute([1, 2, 0]))

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
